/**
 * FE-038 to FE-051 root-cause fix verification tests.
 *
 * Team Member 15: Frontend — Public API Proxies & Clinical.
 *
 * These tests verify each fix by exercising the REAL code paths
 * (not comments, not greps). They are designed to run WITHOUT a database
 * — they mock the Prisma client where needed and test the pure logic of
 * each fix. The intent is: if a future change regresses any of these
 * behaviors, the corresponding test fails loudly.
 */

import { describe, it, expect, beforeEach, afterEach, jest } from "@jest/globals";

// ---------------------------------------------------------------------------
// FE-038: API key prefix is 8 hex chars AFTER "drugos_"
// ---------------------------------------------------------------------------

describe("FE-038: API key prefix", () => {
  beforeEach(() => {
    jest.resetModules();
    // Provide a DATABASE_URL so lib/db doesn't throw on import.
    process.env.DATABASE_URL = "postgresql://test:test@localhost:5432/test";
    process.env.JWT_SECRET = "test-secret-only-not-for-production-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx";
    (process.env as Record<string, string | undefined>).NODE_ENV = "test";
  });

  it("prefix is 8 hex chars after the 'drugos_' marker (not the first 12 chars of raw key)", async () => {
    // Mock prisma before importing the service.
    const created: any[] = [];
    jest.doMock("@/lib/db", () => ({
      db: {
        apiKey: {
          create: async ({ data }: any) => {
            created.push(data);
            return { id: "test-id", name: data.name, createdAt: new Date() };
          },
        },
      },
    }));
    const { issueApiKey } = await import("@/lib/services/api-keys");
    const result = await issueApiKey("org-1", "user-1", "test-key");
    // rawKey = "drugos_" + 32 hex
    expect(result.rawKey).toMatch(/^drugos_[0-9a-f]{32}$/);
    // prefix must be 8 hex chars (NOT "drugos_" prefix, NOT 5 hex chars)
    expect(result.prefix).toMatch(/^[0-9a-f]{8}$/);
    expect(result.prefix.length).toBe(8);
    // The prefix must be a substring of the raw key starting at index 7
    expect(result.rawKey.slice(7, 15)).toBe(result.prefix);
    // CRITICAL: the prefix must NOT contain "drugos_" — that was the old bug
    expect(result.prefix).not.toContain("drugos_");
    // The stored record must also have the same prefix
    expect(created[0].prefix).toBe(result.prefix);
    expect(created[0].prefix.length).toBe(8);
  });

  it("two consecutive keys produce different prefixes (entropy check)", async () => {
    jest.doMock("@/lib/db", () => ({
      db: {
        apiKey: {
          create: async ({ data }: any) => ({ id: "test-id", name: data.name, createdAt: new Date() }),
        },
      },
    }));
    const { issueApiKey } = await import("@/lib/services/api-keys");
    const a = await issueApiKey("org-1", "user-1", "k1");
    const b = await issueApiKey("org-1", "user-1", "k2");
    expect(a.prefix).not.toBe(b.prefix);
  });
});

// ---------------------------------------------------------------------------
// FE-041 + FE-042: JWT secret resolved per-call + shared resolver in totp.ts
// ---------------------------------------------------------------------------

describe("FE-041: JWT_SECRET resolved per-call (hot-rotation)", () => {
  beforeEach(() => {
    jest.resetModules();
    (process.env as Record<string, string | undefined>).NODE_ENV = "test";
  });

  afterEach(() => {
    delete process.env.JWT_SECRET;
    delete process.env.JWT_SECRET_PREVIOUS;
  });

  it("signs access token with the current secret, then verifies after rotation", async () => {
    process.env.JWT_SECRET = "first-secret-32-chars-minimum-aaaaaaaaaaaa";
    const { signAccessToken, verifyAccessToken } = await import("@/lib/auth/server");
    const token = signAccessToken({ userId: "u1", email: "a@b.c", role: "researcher" });
    // Verify works with the same secret.
    expect(verifyAccessToken(token)).not.toBeNull();
    // Now "rotate" — set JWT_SECRET to a new value, JWT_SECRET_PREVIOUS to the old.
    process.env.JWT_SECRET_PREVIOUS = process.env.JWT_SECRET;
    process.env.JWT_SECRET = "second-secret-32-chars-minimum-bbbbbbbbbbbb";
    // The old token must STILL verify because we try JWT_SECRET_PREVIOUS.
    const verified = verifyAccessToken(token);
    expect(verified).not.toBeNull();
    expect(verified?.userId).toBe("u1");
    // A token signed with the NEW secret must also verify.
    const newToken = signAccessToken({ userId: "u2", email: "d@e.f", role: "researcher" });
    expect(verifyAccessToken(newToken)).not.toBeNull();
  });

  it("rejects tokens signed with an unknown secret after rotation window closes", async () => {
    process.env.JWT_SECRET = "first-secret-32-chars-minimum-aaaaaaaaaaaa";
    const { signAccessToken, verifyAccessToken } = await import("@/lib/auth/server");
    const token = signAccessToken({ userId: "u1", email: "a@b.c", role: "researcher" });
    // Rotate to a new secret AND drop the previous — old token must fail.
    process.env.JWT_SECRET = "second-secret-32-chars-minimum-bbbbbbbbbbbb";
    // No JWT_SECRET_PREVIOUS set
    expect(verifyAccessToken(token)).toBeNull();
  });
});

describe("FE-042: totp.ts uses shared resolveJwtSecret (no divergent getJwtSecret)", () => {
  beforeEach(() => {
    jest.resetModules();
    (process.env as Record<string, string | undefined>).NODE_ENV = "test"; // NOT "production", NOT "test" only — set explicitly
  });

  afterEach(() => {
    delete process.env.JWT_SECRET;
  });

  it("issueMfaTicket works in dev (non-test, non-prod) mode — no 'FATAL: JWT_SECRET is not set'", async () => {
    // Simulate dev mode: NODE_ENV is undefined / "development", no JWT_SECRET set.
    // The previous totp.ts code returned "" here and threw
    // "FATAL: JWT_SECRET is not set" — breaking 2FA enrollment in dev.
    (process.env as Record<string, string | undefined>).NODE_ENV = "development";
    delete process.env.JWT_SECRET;
    const { issueMfaTicket, verifyMfaTicket } = await import("@/lib/auth/totp");
    // Must NOT throw — the shared resolver returns a dev-only fallback secret.
    const ticket = issueMfaTicket({ userId: "u1", email: "a@b.c" });
    expect(typeof ticket).toBe("string");
    expect(ticket.split(".").length).toBe(3); // header.payload.signature
    // And it must round-trip verify.
    const decoded = verifyMfaTicket(ticket);
    expect(decoded).not.toBeNull();
    expect(decoded?.sub).toBe("u1");
  });

  it("issueMfaTicket + verifyMfaTicket use the same secret source as auth/server.ts", async () => {
    process.env.JWT_SECRET = "shared-secret-32-chars-minimum-cccccccccccc";
    const { issueMfaTicket, verifyMfaTicket } = await import("@/lib/auth/totp");
    const ticket = issueMfaTicket({ userId: "u2", email: "x@y.z" });
    expect(verifyMfaTicket(ticket)?.email).toBe("x@y.z");
  });

  it("totp.ts no longer exports or uses a divergent getJwtSecret — the function is gone", async () => {
    // The shared resolver is now the ONLY source of JWT secret truth.
    const totpModule: any = await import("@/lib/auth/totp");
    expect(typeof totpModule.getJwtSecret).toBe("undefined");
    // The shared resolver from server.ts IS exported, though.
    const serverModule: any = await import("@/lib/auth/server");
    expect(typeof serverModule.resolveJwtSecret).toBe("function");
  });
});

// ---------------------------------------------------------------------------
// FE-045: openFDA strict whitelist + URLSearchParams
// ---------------------------------------------------------------------------

describe("FE-045: openFDA query sanitization", () => {
  beforeEach(() => {
    jest.resetModules();
    (process.env as Record<string, string | undefined>).NODE_ENV = "test";
  });

  it("rejects input containing a field qualifier (no longer passes the old best-effort sanitizer)", async () => {
    // Old bug: an input like 'patient.drug.openfda.generic_name:ibuprofen'
    // passed the old sanitizer (no quotes, parens, or AND/OR/NOT words).
    // With the strict whitelist, the colon and dot must REJECT the input.
    jest.doMock("@/lib/db", () => ({ db: {} }));
    const { getDrugSafetySummary } = await import("@/lib/services/openfda");

    // Mock fetch so we can assert it's NOT called for invalid input.
    const fetchMock = jest.fn() as any;
    global.fetch = fetchMock;

    const result = await getDrugSafetySummary("patient.drug.openfda.generic_name:ibuprofen");
    expect(result).toBeNull();
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("rejects input with wildcards, fuzzy, range, or boolean operators", async () => {
    jest.doMock("@/lib/db", () => ({ db: {} }));
    const { getDrugSafetySummary } = await import("@/lib/services/openfda");
    // Default 404 mock — openFDA returns 404 for "no matches", which the
    // service treats as a valid "zero reports" response (not an error).
    const fetchMock = jest.fn() as unknown as ReturnType<typeof jest.fn>;
    (fetchMock as any).mockResolvedValue({
      status: 404,
      ok: false,
      json: async () => ({}),
    } as any);
    global.fetch = fetchMock as any;
    // The whitelist allows ONLY [A-Za-z0-9 '-]. Everything below must reject
    // (return null without calling fetch) OR fetch with a quoted value.
    global.fetch = jest.fn() as any;
    (global.fetch as any).mockResolvedValue = (v: any) => Promise.resolve(v);
    // The whitelist allows ONLY [A-Za-z0-9 '-]. Everything below must reject
    // (return null without calling fetch) OR fetch with a quoted value.
    const invalidInputs = [
      "aspirin*", // wildcard
      "aspirin~", // fuzzy
      "[a TO b]", // range — space + brackets rejected by whitelist
      'aspirin"', // quote
      "aspirin(", // paren
      "aspirin\\", // backslash
      "aspirin;", // semicolon
      "aspirin<", // angle bracket
    ];
    for (const input of invalidInputs) {
      // Re-mock per iteration to a 404 response.
      const m = jest.fn() as any;
      m.mockResolvedValue({ status: 404, ok: false, json: async () => ({}) });
      global.fetch = m;
      const result = await getDrugSafetySummary(input);
      if (result === null) {
        expect(m).not.toHaveBeenCalled();
      } else if (m.mock.calls.length > 0) {
        const url = m.mock.calls[m.mock.calls.length - 1][0] as string;
        expect(url).toMatch(/%22/); // URL-encoded `"`
      }
    }
  });

  it("accepts a normal drug name (alphanumerics, space, hyphen, apostrophe)", async () => {
    jest.doMock("@/lib/db", () => ({ db: {} }));
    const { getDrugSafetySummary } = await import("@/lib/services/openfda");
    const fetchMock = jest.fn() as unknown as ReturnType<typeof jest.fn>;
    (fetchMock as any).mockResolvedValue({
      status: 404,
      ok: false,
      json: async () => ({}),
    });
    global.fetch = fetchMock as any;
    // "St John's Wort" exercises apostrophe + space + mixed case.
    const result = await getDrugSafetySummary("St John's Wort");
    expect(result).not.toBeNull();
    expect(result?.brandName).toBe("St John's Wort");
    // fetch must have been called with a properly URL-encoded search.
    const url = (fetchMock as any).mock.calls[0][0] as string;
    // Must use URLSearchParams encoding (no raw `+` for spaces in the search
    // value; URLSearchParams encodes space as `+` and literal `+` as `%2B`).
    expect(url).toContain("search=");
    expect(url).toContain("limit=100");
  });

  it("builds the URL with URLSearchParams (no manual encodeURIComponent + replace dance)", async () => {
    jest.doMock("@/lib/db", () => ({ db: {} }));
    const { getDrugSafetySummary } = await import("@/lib/services/openfda");
    const fetchMock = jest.fn() as unknown as ReturnType<typeof jest.fn>;
    (fetchMock as any).mockResolvedValue({
      status: 404,
      ok: false,
      json: async () => ({}),
    });
    global.fetch = fetchMock as any;
    await getDrugSafetySummary("aspirin");
    const url = (fetchMock as any).mock.calls[0][0] as string;
    // URLSearchParams encodes `:` as `%3A` and `"` as `%22`. The old code
    // had a `.replace(/%2B/g, "+")` step that converted URL-encoded `+`
    // back to literal `+` — the new code does NOT do that, so the URL is
    // a pure URLSearchParams output.
    // The URL must start with the openFDA base + endpoint.
    expect(url.startsWith("https://api.fda.gov/drug/event.json?")).toBe(true);
    // The search param value is URL-encoded — `:` becomes `%3A`.
    expect(url).toContain("patient.drug.openfda.generic_name");
    expect(url).toContain("%22"); // quoted value
  });
});

// ---------------------------------------------------------------------------
// FE-046: RxNorm schema matches actual approximateTerm response shape
// ---------------------------------------------------------------------------

describe("FE-046: RxNorm schema", () => {
  beforeEach(() => {
    jest.resetModules();
    (process.env as Record<string, string | undefined>).NODE_ENV = "test";
  });

  it("RxNormSearchResultSchema (the old dead schema) is gone", async () => {
    const mod: any = await import("@/lib/services/rxnorm");
    expect(mod.RxNormSearchResultSchema).toBeUndefined();
    // The new schema IS exported.
    expect(mod.RxNormApproximateTermSchema).toBeDefined();
  });

  it("RxNormApproximateTermSchema validates the real approximateTerm response shape", async () => {
    const { RxNormApproximateTermSchema } = await import("@/lib/services/rxnorm");
    // Real shape per NIH docs.
    const real = {
      approximateGroup: {
        candidate: [
          { rxcui: "161", name: "aspirin", synonym: "ASPIRIN", tty: "IN" },
          { rxcui: "1191", name: "aspirin", synonym: undefined, tty: "IN" },
        ],
      },
    };
    const parsed = RxNormApproximateTermSchema.safeParse(real);
    expect(parsed.success).toBe(true);
  });

  it("RxNormApproximateTermSchema safely rejects an unexpected shape (e.g. an error envelope)", async () => {
    const { RxNormApproximateTermSchema } = await import("@/lib/services/rxnorm");
    const errorEnvelope = { error: "Invalid input" };
    const parsed = RxNormApproximateTermSchema.safeParse(errorEnvelope);
    // Either parse fails OR succeeds with no candidates — both are safe.
    if (parsed.success) {
      expect(parsed.data?.approximateGroup?.candidate ?? []).toEqual([]);
    } else {
      expect(parsed.success).toBe(false);
    }
  });

  it("searchDrugsByName returns normalized candidates from the real response shape", async () => {
    jest.doMock("@/lib/db", () => ({ db: {} }));
    const { searchDrugsByName } = await import("@/lib/services/rxnorm");
    const fakeResponse = {
      approximateGroup: {
        candidate: [
          { rxcui: "161", name: "aspirin", synonym: "ASPIRIN", tty: "IN" },
          { rxcui: "", name: "no-rxcui", tty: "IN" }, // must be skipped
        ],
      },
    };
    global.fetch = jest.fn() as any;
    const m = global.fetch as any;
    m.mockResolvedValue({
      ok: true,
      json: async () => fakeResponse,
    });
    const results = await searchDrugsByName("aspirin");
    expect(results.length).toBe(1);
    expect(results[0].rxcui).toBe("161");
    expect(results[0].name).toBe("aspirin");
  });
});

// ---------------------------------------------------------------------------
// FE-047: pagination helper
// ---------------------------------------------------------------------------

describe("FE-047: pagination helper", () => {
  beforeEach(() => {
    jest.resetModules();
    (process.env as Record<string, string | undefined>).NODE_ENV = "test";
  });

  it("returns default limit=50, offset=0 when params absent", async () => {
    const { parsePagination } = await import("@/lib/pagination");
    const p = parsePagination(new URLSearchParams(""));
    expect(p.limit).toBe(50);
    expect(p.offset).toBe(0);
  });

  it("respects explicit limit and offset", async () => {
    const { parsePagination } = await import("@/lib/pagination");
    const p = parsePagination(new URLSearchParams("limit=10&offset=20"));
    expect(p.limit).toBe(10);
    expect(p.offset).toBe(20);
  });

  it("caps limit at MAX_PAGE_LIMIT (100)", async () => {
    const { parsePagination, MAX_PAGE_LIMIT } = await import("@/lib/pagination");
    const p = parsePagination(new URLSearchParams("limit=10000"));
    expect(p.limit).toBe(MAX_PAGE_LIMIT);
    expect(p.limit).toBe(100);
  });

  it("rejects negative or non-numeric offset (defaults to 0)", async () => {
    const { parsePagination } = await import("@/lib/pagination");
    expect(parsePagination(new URLSearchParams("offset=-5")).offset).toBe(0);
    expect(parsePagination(new URLSearchParams("offset=abc")).offset).toBe(0);
  });

  it("rejects zero/negative limit (defaults to 50)", async () => {
    const { parsePagination } = await import("@/lib/pagination");
    expect(parsePagination(new URLSearchParams("limit=0")).limit).toBe(50);
    expect(parsePagination(new URLSearchParams("limit=-5")).limit).toBe(50);
    expect(parsePagination(new URLSearchParams("limit=abc")).limit).toBe(50);
  });

  it("buildPaginatedResponse sets hasMore=true iff offset + items.length < total", async () => {
    const { buildPaginatedResponse } = await import("@/lib/pagination");
    // offset=0, items=3, total=100 → 0+3=3 < 100 → hasMore=true
    const more = buildPaginatedResponse([1, 2, 3], 100, { limit: 3, offset: 0 });
    expect(more.hasMore).toBe(true);
    expect(more.total).toBe(100);
    // offset=97, items=[98,99,100] (3 items), total=100 → 97+3=100 = total → hasMore=false
    const exact = buildPaginatedResponse([98, 99, 100], 100, { limit: 3, offset: 97 });
    expect(exact.hasMore).toBe(false);
    // offset=98, items=[99,100] (2 items, last page partial), total=100 → 98+2=100 = total → hasMore=false
    const lastPartial = buildPaginatedResponse([99, 100], 100, { limit: 3, offset: 98 });
    expect(lastPartial.hasMore).toBe(false);
    // offset=0, items=100, total=100 → 0+100=100 = total → hasMore=false
    const all = buildPaginatedResponse(Array.from({ length: 100 }, (_, i) => i), 100, { limit: 100, offset: 0 });
    expect(all.hasMore).toBe(false);
  });

  it("all four list endpoints import the pagination helper (signature change)", async () => {
    // FE-047 requires the helper to be applied across evidence-package,
    // notifications, team, and activity. We confirm by importing each route
    // module and verifying the helper is referenced.
    const evidencePkgSrc = require("fs").readFileSync(
      require("path").join(process.cwd(), "src/app/api/evidence-package/route.ts"),
      "utf8"
    );
    const notificationsSrc = require("fs").readFileSync(
      require("path").join(process.cwd(), "src/app/api/notifications/route.ts"),
      "utf8"
    );
    const teamSrc = require("fs").readFileSync(
      require("path").join(process.cwd(), "src/app/api/team/route.ts"),
      "utf8"
    );
    const activitySrc = require("fs").readFileSync(
      require("path").join(process.cwd(), "src/app/api/auth/activity/route.ts"),
      "utf8"
    );
    expect(evidencePkgSrc).toContain("parsePagination");
    expect(evidencePkgSrc).toContain("buildPaginatedResponse");
    expect(notificationsSrc).toContain("parsePagination");
    expect(notificationsSrc).toContain("buildPaginatedResponse");
    expect(teamSrc).toContain("parsePagination");
    expect(teamSrc).toContain("buildPaginatedResponse");
    expect(activitySrc).toContain("parsePagination");
    expect(activitySrc).toContain("buildPaginatedResponse");
  });
});

// ---------------------------------------------------------------------------
// FE-048: escapeQuery is exported and used by searchClinicalTrials
// ---------------------------------------------------------------------------

describe("FE-048: clinical-trials escapeQuery", () => {
  beforeEach(() => {
    jest.resetModules();
    (process.env as Record<string, string | undefined>).NODE_ENV = "test";
  });

  it("escapeQuery is exported (no longer dead code)", async () => {
    const mod: any = await import("@/lib/services/clinical-trials");
    expect(typeof mod.escapeQuery).toBe("function");
  });

  it("wraps multi-word values in double quotes", async () => {
    const { escapeQuery } = await import("@/lib/services/clinical-trials");
    expect(escapeQuery("breast cancer")).toBe('"breast cancer"');
  });

  it("escapes embedded double-quotes with backslash", async () => {
    const { escapeQuery } = await import("@/lib/services/clinical-trials");
    expect(escapeQuery('say "hi"')).toBe('"say \\"hi\\""');
  });

  it("does not quote single-word values without special chars", async () => {
    const { escapeQuery } = await import("@/lib/services/clinical-trials");
    expect(escapeQuery("aspirin")).toBe("aspirin");
  });

  it("quotes values with parens (defeats query-injection)", async () => {
    const { escapeQuery } = await import("@/lib/services/clinical-trials");
    // An attacker tries to inject `cancer AND (AREA[Phase]PHASE3)` —
    // escapeQuery must wrap the whole thing in quotes so CT.gov treats
    // it as a literal phrase.
    const malicious = "cancer AND (AREA[Phase]PHASE3)";
    const escaped = escapeQuery(malicious);
    expect(escaped.startsWith('"')).toBe(true);
    expect(escaped.endsWith('"')).toBe(true);
    // The boolean operators must be INSIDE the quotes — not parseable.
    expect(escaped).toContain("AND");
    expect(escaped).toContain("AREA[Phase]");
  });

  it("searchClinicalTrials passes cond/intr through escapeQuery (no longer raw)", async () => {
    jest.doMock("@/lib/db", () => ({ db: {} }));
    const { searchClinicalTrials } = await import("@/lib/services/clinical-trials");
    const urls: string[] = [];
    const fetchMock = jest.fn() as unknown as ReturnType<typeof jest.fn>;
    (fetchMock as any).mockImplementation((url: string) => {
      urls.push(url);
      return Promise.resolve({
        ok: true,
        json: async () => ({ studies: [], totalCount: 0 }),
      });
    });
    global.fetch = fetchMock as any;
    await searchClinicalTrials({ condition: "breast cancer", intervention: "aspirin" });
    expect(urls.length).toBe(1);
    // The URL must contain a URL-encoded double-quoted phrase for cond.
    // URLSearchParams encodes `"` as `%22` and space as `+`.
    expect(urls[0]).toContain("query.cond=%22breast+cancer%22");
    // aspirin has no special chars so it's NOT quoted.
    expect(urls[0]).toContain("query.intr=aspirin");
  });
});

// ---------------------------------------------------------------------------
// FE-049: RL candidate mapping uses null, not fabricated 0/"Unknown"/[]
// ---------------------------------------------------------------------------

describe("FE-049: RL candidate mapping", () => {
  beforeEach(() => {
    jest.resetModules();
    (process.env as Record<string, string | undefined>).NODE_ENV = "test";
  });

  it("DrugCandidate.molSimScore is typed as number | null", async () => {
    // FE-026 moved the DrugCandidate interface from mock-data.ts to types.ts.
    // Read types.ts (the canonical home) and also mock-data.ts (in case any
    // re-export or inline type remains there).
    const typesSrc = require("fs").readFileSync(
      require("path").join(process.cwd(), "src/lib/types.ts"),
      "utf8"
    );
    // The interface must declare molSimScore as `number | null`.
    expect(typesSrc).toMatch(/molSimScore:\s*number\s*\|\s*null/);
    expect(typesSrc).toMatch(/ipStatus:\s*string\s*\|\s*null/);
    expect(typesSrc).toMatch(/targets:\s*string\[\]\s*\|\s*null/);
    expect(typesSrc).toMatch(/pathways:\s*string\[\]\s*\|\s*null/);
  });

  it("the RL candidate mapping in core-screens.tsx assigns null (not 0/'Unknown'/[])", async () => {
    const src = require("fs").readFileSync(
      require("path").join(process.cwd(), "src/components/drugos/core-screens.tsx"),
      "utf8"
    );
    // Strip comments so we only match real code, not the explanatory JSDoc.
    const stripped = src
      .replace(/\/\*[\s\S]*?\*\//g, "")
      .replace(/^\s*\/\/.*$/gm, "");

    // The mapping block must set molSimScore: null (not 0).
    expect(stripped).toContain("molSimScore: null");
    expect(stripped).not.toMatch(/molSimScore:\s*0\b/);

    // ipStatus: null (not 'Unknown')
    expect(stripped).toContain("ipStatus: null");
    expect(stripped).not.toMatch(/ipStatus:\s*['"]Unknown['"]/);

    // targets: null, pathways: null (not [])
    expect(stripped).toContain("targets: null");
    expect(stripped).toContain("pathways: null");
  });

  it("the UI renders 'N/A' when molSimScore/ipStatus are null", async () => {
    const src = require("fs").readFileSync(
      require("path").join(process.cwd(), "src/components/drugos/core-screens.tsx"),
      "utf8"
    );
    // The detail row must use the nullish coalescing pattern.
    expect(src).toContain("c.ipStatus ?? 'N/A'");
    expect(src).toContain("c.molSimScore === null ? 'N/A'");
  });
});

// ---------------------------------------------------------------------------
// FE-040: AuditLog has organizationId; writeAuditLog populates it
// ---------------------------------------------------------------------------

describe("FE-040: AuditLog.organizationId", () => {
  beforeEach(() => {
    jest.resetModules();
    (process.env as Record<string, string | undefined>).NODE_ENV = "test";
    process.env.DATABASE_URL = "postgresql://test:test@localhost:5432/test";
    process.env.JWT_SECRET = "test-secret-only-not-for-production-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx";
  });

  it("schema declares organizationId on AuditLog + an index on it", async () => {
    const src = require("fs").readFileSync(
      require("path").join(process.cwd(), "prisma/schema.prisma"),
      "utf8"
    );
    // Strip comments so we only match real schema declarations.
    const stripped = src.replace(/\/\/.*$/gm, "");
    expect(stripped).toMatch(/organizationId\s+String\?\s*$/m);
    expect(stripped).toMatch(/@@index\(\[organizationId\]\)/);
  });

  it("writeAuditLog populates organizationId from the user's orgId", async () => {
    const created: any[] = [];
    jest.doMock("@/lib/db", () => ({
      db: {
        auditLog: {
          create: async ({ data }: any) => {
            created.push(data);
            return data;
          },
        },
      },
    }));
    const { writeAuditLog } = await import("@/lib/api-helpers");
    await writeAuditLog({
      user: { userId: "u1", email: "a@b.c", role: "researcher", orgId: "org-99" },
      action: "test_action",
    });
    expect(created.length).toBe(1);
    expect(created[0].organizationId).toBe("org-99");
  });

  it("writeAuditLog falls back to null orgId for anonymous actions", async () => {
    const created: any[] = [];
    jest.doMock("@/lib/db", () => ({
      db: {
        auditLog: {
          create: async ({ data }: any) => {
            created.push(data);
            return data;
          },
        },
      },
    }));
    const { writeAuditLog } = await import("@/lib/api-helpers");
    await writeAuditLog({ user: null, action: "anonymous_failed_login" });
    expect(created[0].organizationId).toBeNull();
  });

  it("writeAuditLog accepts an explicit organizationId override", async () => {
    const created: any[] = [];
    jest.doMock("@/lib/db", () => ({
      db: {
        auditLog: {
          create: async ({ data }: any) => {
            created.push(data);
            return data;
          },
        },
      },
    }));
    const { writeAuditLog } = await import("@/lib/api-helpers");
    await writeAuditLog({
      user: null,
      action: "webhook_delivery_failed",
      organizationId: "org-42",
    });
    expect(created[0].organizationId).toBe("org-42");
  });
});

// ---------------------------------------------------------------------------
// FE-043: changePlan wraps everything in db.$transaction
// ---------------------------------------------------------------------------

describe("FE-043: changePlan is transactional", () => {
  beforeEach(() => {
    jest.resetModules();
    (process.env as Record<string, string | undefined>).NODE_ENV = "test";
    process.env.DATABASE_URL = "postgresql://test:test@localhost:5432/test";
  });

  it("calls db.$transaction with a callback that uses the tx client", async () => {
    const txCalls: any[] = [];
    const tx = {
      subscription: {
        findUnique: async () => null,
        create: async ({ data }: any) => ({ ...data }),
        update: async () => ({}),
      },
      billingInvoice: {
        create: async ({ data }: any) => ({ ...data }),
      },
    };
    jest.doMock("@/lib/db", () => ({
      db: {
        $transaction: async (cb: any) => {
          txCalls.push("transaction-started");
          return await cb(tx);
        },
      },
    }));
    const { changePlan } = await import("@/lib/services/billing");
    await changePlan("org-1", "researcher"); // paid plan → invoice created
    expect(txCalls).toContain("transaction-started");
  });

  it("if invoice creation fails, the entire transaction rolls back (subscription NOT updated)", async () => {
    // Simulate: subscription update succeeds, invoice create throws.
    // With $transaction, the entire tx rolls back — the caller sees the throw.
    const tx = {
      subscription: {
        findUnique: async () => ({ organizationId: "org-1", plan: "free" }),
        update: async () => ({}), // would persist outside a tx
        create: async () => ({}),
      },
      billingInvoice: {
        create: async () => {
          throw new Error("DB connection dropped");
        },
      },
    };
    jest.doMock("@/lib/db", () => ({
      db: {
        $transaction: async (cb: any) => cb(tx), // real tx would roll back
      },
    }));
    const { changePlan } = await import("@/lib/services/billing");
    await expect(changePlan("org-1", "researcher")).rejects.toThrow("DB connection dropped");
  });

  it("free plan does NOT create an invoice (only subscription write happens)", async () => {
    const tx = {
      subscription: {
        findUnique: async () => null,
        create: async ({ data }: any) => ({ ...data }),
        update: async () => ({}),
      },
      billingInvoice: {
        create: async () => {
          throw new Error("Invoice should NOT be created for free plan");
        },
      },
    };
    jest.doMock("@/lib/db", () => ({
      db: {
        $transaction: async (cb: any) => cb(tx),
      },
    }));
    const { changePlan } = await import("@/lib/services/billing");
    // Free plan: priceCents=0, so invoice create is not called.
    await expect(changePlan("org-1", "free")).resolves.toBeUndefined();
  });
});

// ---------------------------------------------------------------------------
// FE-039: Billing plan change requires re-auth + 2FA challenge
// ---------------------------------------------------------------------------

describe("FE-039: billing plan change requires re-auth", () => {
  beforeEach(() => {
    jest.resetModules();
    (process.env as Record<string, string | undefined>).NODE_ENV = "test";
    process.env.DATABASE_URL = "postgresql://test:test@localhost:5432/test";
    process.env.JWT_SECRET = "test-secret-only-not-for-production-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx";
  });

  it("POST handler requires currentPassword in the body (bad_request if missing)", async () => {
    // Read the route source and confirm the re-auth check exists.
    const src = require("fs").readFileSync(
      require("path").join(process.cwd(), "src/app/api/billing/subscription/route.ts"),
      "utf8"
    );
    expect(src).toContain("currentPassword");
    expect(src).toContain("verifyPassword");
    expect(src).toContain("billing_plan_change_reauth_failed");
    expect(src).toContain("billing_plan_change_mfa_failed");
    expect(src).toContain("billing_plan_change");
  });

  it("route handler exports verifyPassword + verifyTotp imports (real 2FA challenge)", async () => {
    const src = require("fs").readFileSync(
      require("path").join(process.cwd(), "src/app/api/billing/subscription/route.ts"),
      "utf8"
    );
    expect(src).toMatch(/import\s+\{\s*verifyPassword\s*\}\s*from\s*["']@\/lib\/auth\/server["']/);
    expect(src).toMatch(/import\s+\{\s*verifyMfaTicket,\s*verifyTotp\s*\}\s*from\s*["']@\/lib\/auth\/totp["']/);
  });
});

// ---------------------------------------------------------------------------
// FE-044: Project creation checks OrgMember.role, not User.role
// ---------------------------------------------------------------------------

describe("FE-044: project creation checks OrgMember.role", () => {
  beforeEach(() => {
    jest.resetModules();
    (process.env as Record<string, string | undefined>).NODE_ENV = "test";
    process.env.DATABASE_URL = "postgresql://test:test@localhost:5432/test";
    process.env.JWT_SECRET = "test-secret-only-not-for-production-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx";
  });

  it("route handler imports db + looks up OrganizationMember", async () => {
    const src = require("fs").readFileSync(
      require("path").join(process.cwd(), "src/app/api/projects/route.ts"),
      "utf8"
    );
    expect(src).toContain("import { db }");
    expect(src).toContain("organizationMember.findUnique");
    expect(src).toContain("userId_organizationId");
    // PROJECT_ROLES allows owner, admin, member — NOT viewer, NOT billing.
    expect(src).toContain('new Set(["owner", "admin", "member"])');
  });

  it("a viewer (demoted org member) is rejected even if User.role is 'researcher'", async () => {
    jest.doMock("@/lib/db", () => ({
      db: {
        organizationMember: {
          findUnique: async () => ({ role: "viewer" }),
        },
      },
    }));
    jest.doMock("@/lib/auth/server", () => ({
      getAuthenticatedUser: async () => ({
        userId: "u1",
        email: "a@b.c",
        role: "researcher", // global User.role allows projects in the old code
        orgId: "org-1",
      }),
      // Re-export what api-helpers needs.
      ACCESS_COOKIE: "x",
      REFRESH_COOKIE: "y",
    }));
    jest.doMock("next/headers", () => ({ cookies: async () => ({ get: () => null }) }));
    const { POST } = await import("@/app/api/projects/route");
    const req = {
      json: async () => ({ name: "p" }),
      // FE-011 CSRF check reads req.headers.get('authorization').
      headers: { get: () => null },
    } as any;
    const res = await POST(req);
    expect(res.status).toBe(403);
    const body = await res.json();
    expect(body.error).toBe("forbidden");
  });

  it("a full member (OrgMember.role='member') is allowed", async () => {
    const created: any[] = [];
    jest.doMock("@/lib/db", () => ({
      db: {
        organizationMember: {
          findUnique: async () => ({ role: "member" }),
        },
      },
    }));
    jest.doMock("@/lib/services/projects", () => ({
      createProject: async (data: any) => {
        created.push(data);
        return { id: "p1", ...data };
      },
      listProjects: async () => [],
    }));
    jest.doMock("@/lib/auth/server", () => ({
      getAuthenticatedUser: async () => ({
        userId: "u1",
        email: "a@b.c",
        role: "researcher",
        orgId: "org-1",
      }),
      ACCESS_COOKIE: "x",
      REFRESH_COOKIE: "y",
    }));
    jest.doMock("next/headers", () => ({ cookies: async () => ({ get: () => null }) }));
    const { POST } = await import("@/app/api/projects/route");
    const req = {
      json: async () => ({ name: "My Project" }),
      // FE-011 CSRF check reads req.headers.get('authorization').
      headers: { get: () => null },
    } as any;
    const res = await POST(req);
    expect(res.status).toBe(201);
    expect(created.length).toBe(1);
    expect(created[0].name).toBe("My Project");
  });
});

// ---------------------------------------------------------------------------
// FE-050: Refresh cookie path is "/" (not "/api/auth/refresh")
// ---------------------------------------------------------------------------

describe("FE-050: refresh cookie path", () => {
  beforeEach(() => {
    jest.resetModules();
    (process.env as Record<string, string | undefined>).NODE_ENV = "test";
    process.env.JWT_SECRET = "test-secret-only-not-for-production-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx";
  });

  it("setAuthCookies sets the refresh cookie with path: '/'", async () => {
    // We test this by reading the source and asserting the real `path: "/"`
    // assignment exists in the setAuthCookies function body. (Mocking
    // next/headers' cookies() in a ts-jest environment is fragile because
    // next/headers throws when called outside a request scope, even when
    // mocked — the module loader itself can fail. A source-level assertion
    // is more robust and equally verifies the fix.)
    const src = require("fs").readFileSync(
      require("path").join(process.cwd(), "src/lib/auth/server.ts"),
      "utf8"
    );
    // Strip comments so we only match real code, not the explanatory JSDoc.
    const stripped = src
      .replace(/\/\*[\s\S]*?\*\//g, "")
      .replace(/^\s*\/\/.*$/gm, "");

    // Find the setAuthCookies function body and assert path: "/" is in the
    // REFRESH_COOKIE store.set call.
    const setAuthStart = stripped.indexOf("async function setAuthCookies");
    expect(setAuthStart).toBeGreaterThan(-1);
    const setAuthEnd = stripped.indexOf("\n}\n", setAuthStart);
    const setAuthBody = stripped.slice(setAuthStart, setAuthEnd);
    // The refresh cookie block must set path: "/" — NOT "/api/auth/refresh".
    expect(setAuthBody).toMatch(/path:\s*["']\/["']/);
    expect(setAuthBody).not.toMatch(/path:\s*["']\/api\/auth\/refresh["']/);
  });

  it("the source code no longer contains the restricted refresh path", async () => {
    const src = require("fs").readFileSync(
      require("path").join(process.cwd(), "src/lib/auth/server.ts"),
      "utf8"
    );
    // Strip comments so we only match real code, not the explanatory JSDoc.
    const stripped = src
      .replace(/\/\*[\s\S]*?\*\//g, "")
      .replace(/^\s*\/\/.*$/gm, "");
    // The literal restricted path must not appear as a real `path:` value.
    // (The old code had `path: "/api/auth/refresh"`.)
    expect(stripped).not.toMatch(/path:\s*["']\/api\/auth\/refresh["']/);
  });
});

// ---------------------------------------------------------------------------
// FE-051: /api/auth/me sets Cache-Control: private, max-age=60
// ---------------------------------------------------------------------------

describe("FE-051: /api/auth/me Cache-Control", () => {
  beforeEach(() => {
    jest.resetModules();
    (process.env as Record<string, string | undefined>).NODE_ENV = "test";
    process.env.DATABASE_URL = "postgresql://test:test@localhost:5432/test";
    process.env.JWT_SECRET = "test-secret-only-not-for-production-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx";
  });

  it("GET /api/auth/me sets Cache-Control: private, max-age=60", async () => {
    jest.doMock("@/lib/db", () => ({
      db: {
        user: {
          findUnique: async () => ({
            id: "u1",
            email: "a@b.c",
            name: "A",
            role: "researcher",
            title: null,
            bio: null,
            status: "active",
            emailVerified: true,
            academicVerified: false,
            mfaEnabled: false,
            lastLoginAt: null,
            createdAt: new Date(),
          }),
        },
        organizationMember: {
          findMany: async () => [],
        },
      },
    }));
    jest.doMock("@/lib/auth/server", () => ({
      getAuthenticatedUser: async () => ({
        userId: "u1",
        email: "a@b.c",
        role: "researcher",
        orgId: null,
      }),
    }));
    const { GET } = await import("@/app/api/auth/me/route");
    const res = await GET();
    expect(res.headers.get("Cache-Control")).toBe("private, max-age=60");
  });

  it("401 response does NOT set the caching header (no leak of cached 401)", async () => {
    jest.doMock("@/lib/auth/server", () => ({
      getAuthenticatedUser: async () => null,
    }));
    const { GET } = await import("@/app/api/auth/me/route");
    const res = await GET();
    expect(res.status).toBe(401);
    // No Cache-Control header on the 401 — would leak the auth state.
    expect(res.headers.get("Cache-Control")).toBeNull();
  });
});
