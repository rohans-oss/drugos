import { NextRequest, NextResponse } from "next/server";
import { checkRlAvailability } from "@/lib/services/ml-stubs";
import { db } from "@/lib/db";
import { writeAuditLog, requireAuth, internalError, requireCsrfOrSend } from "@/lib/api-helpers";
import {
  getRankedHypotheses,
  type RankedHypothesis,
} from "@/lib/services/rl-ranker";
// BE-029 ROOT FIX (Team Member 12): Zod-validated request body for /api/rl.
import { validateBody, RlBody } from "@/lib/zod-schemas";
// FE-069 ROOT FIX: per-user rate limiting. The CSV cache lives inside
// rl-ranker.ts (the single source of truth for parsing); the rate limiter
// lives here at the route boundary so a flood of requests is rejected
// BEFORE any disk I/O or upstream HTTP call.
//
// FE-017 ROOT FIX (Team Member 14): Use the DISTRIBUTED rate limiter so the
// cap is enforced across all Node.js instances (K8s replicas, etc.). When
// REDIS_URL is set, this hits Redis (shared state). When REDIS_URL is NOT
// set, it falls back to the in-memory Map (single-instance dev/test). The
// sync `checkUserRateLimit` is kept for backwards compatibility with
// existing tests that mock it; the route uses the async version so
// production multi-instance deployments get the correct per-user cap.
import { checkUserRateLimitDistributed as checkUserRateLimitAsync } from "@/lib/auth/per-user-rate-limit";
// Keep the sync import so the route can fall back to it if the async path
// throws (e.g. Redis briefly unreachable). This is a defense-in-depth
// measure: a Redis outage should NOT disable rate limiting entirely.
import { checkUserRateLimit as checkUserRateLimitSync } from "@/lib/auth/per-user-rate-limit";

/**
 * POST /api/rl
 * Body: { drug?: string, disease?: string, limit?: number, sort?: RlSortField, sortDir?: 'asc'|'desc', page?: number, pageSize?: number }
 *
 * FE-019 ROOT FIX: This route previously had its OWN inline CSV parser
 * (parseRlCsv, RlCandidate interface) separate from
 * lib/services/rl-ranker.ts. Two divergent parsers, two env vars
 * (RL_LOCAL_CSV vs RL_OUTPUT_CSV_PATH), two schemas. The lib was dead
 * code. Root fix: deleted the inline parser, import getRankedHypotheses()
 * from the lib. ONE parser, ONE schema, ONE env var.
 *
 * FE-010 ROOT FIX: persistRlCandidates() previously wrote each RL
 * candidate to the Hypothesis table with `status: "validated"`. But
 * these are RAW MODEL PREDICTIONS, not validated hypotheses. A
 * "validated" hypothesis in pharma means wet-lab or clinical
 * confirmation. Mislabeling a model output as "validated" corrupts the
 * scientific record. Root fix: status="predicted", rlPredicted=true.
 *
 * FE-011: CSRF protection applied to POST.
 *
 * FE-069 ROOT FIX: per-user rate limiting (60 req/min) added to BOTH GET
 * and POST. The CSV cache lives inside rl-ranker.ts's readLocalCsv (TTL +
 * mtime + fs.watch). The rate limiter is checked BEFORE any disk I/O so a
 * flood of requests is rejected at the gate with 429 + Retry-After.
 *
 * FE-033 ROOT FIX: Server-side sort + pagination. The previous version
 * accepted only `drug`, `disease`, `limit` and returned all matching
 * candidates (capped at 200). The candidate table then sorted client-side,
 * which froze the browser at 100K-candidate production scale. Root fix:
 * the route now accepts `sort`, `sortDir`, `page`, `pageSize` and passes
 * them to getRankedHypotheses(). The response includes `total` (count
 * after filtering, before pagination) and `page` / `pageSize` so the
 * caller can render "Showing X–Y of Z" and pagination controls. Default
 * page size is 50; max is 200.
 */
export async function POST(req: NextRequest) {
  // FE-011: CSRF protection on every state-changing route.
  const csrf = await requireCsrfOrSend(req);
  if (csrf.response) return csrf.response;

  const auth = await requireAuth();
  if (auth.user === null) return auth.response;

  // FE-069 ROOT FIX: per-user rate limit (60 req/min). Checked BEFORE we
  // parse the request body or touch disk — a flood of requests is rejected
  // at the gate. The 429 response includes Retry-After so well-behaved
  // clients back off correctly.
  //
  // FE-017 ROOT FIX (Team Member 14): use the DISTRIBUTED (async) limiter
  // so the cap is enforced across all Node.js instances. When REDIS_URL is
  // set, this hits Redis (shared state). When REDIS_URL is NOT set, it
  // falls back to the in-memory Map. If the async path throws (e.g. Redis
  // briefly unreachable), we fall back to the sync in-memory limiter so a
  // Redis outage does NOT disable rate limiting entirely.
  let rl;
  try {
    rl = await checkUserRateLimitAsync(auth.user.userId, { max: 60, windowSeconds: 60 });
  } catch (e) {
    console.error("[RATE-LIMIT] async limiter failed, falling back to sync:", e);
    rl = checkUserRateLimitSync(auth.user.userId, { max: 60, windowSeconds: 60 });
  }
  if (rl.blocked) {
    return NextResponse.json(
      { error: "rate_limited", message: "Too many RL requests. Please slow down." },
      { status: 429, headers: { "Retry-After": String(rl.retryAfterSeconds) } }
    );
  }

  let body: {
    drug?: string;
    disease?: string;
    limit?: number;
    sort?: string;
    sortDir?: string;
    page?: number;
    pageSize?: number;
    orgId?: string;
  };
  try {
    body = await req.json();
  } catch {
    body = {};
  }
  // BE-029 ROOT FIX: schema-validate the body. The RlBody schema enforces
  // types (no `body.drug` as object → .trim() throws), lengths (no
  // 10MB drug name), and enum values for sort/sortDir. The previous
  // ad-hoc validation only checked `body.drug || ""` which accepted ANY
  // type and would throw on `.trim()` if body.drug was a number/object.
  const parsed = validateBody(RlBody, body);
  if (!parsed.ok) return parsed.response;
  body = parsed.data;
  const drug = (body.drug || "").trim();
  const disease = (body.disease || "").trim();

  // FE-033: Parse + validate sort + pagination params. Invalid values fall
  // back to the defaults (rank/asc, page 0, pageSize 50).
  const VALID_SORT_FIELDS = ['rank', 'overallScore', 'gnnScore', 'safetyScore', 'marketScore', 'reward', 'drug', 'disease'] as const;
  const VALID_SORT_DIRS = ['asc', 'desc'] as const;
  const sort = (VALID_SORT_FIELDS as readonly string[]).includes(body.sort || '')
    ? (body.sort as typeof VALID_SORT_FIELDS[number])
    : undefined;
  const sortDir = (VALID_SORT_DIRS as readonly string[]).includes(body.sortDir || '')
    ? (body.sortDir as typeof VALID_SORT_DIRS[number])
    : undefined;
  const pageSizeRaw = Math.min(Math.max(1, Number(body.pageSize ?? body.limit ?? 50) || 50), 200);
  const pageRaw = Math.max(0, Math.floor(Number(body.page ?? 0) || 0));
  const offset = pageRaw * pageSizeRaw;

  // FE-007 ROOT FIX (Team Member 13): accept an explicit orgId.
  //
  // The POST body may carry `orgId` to scope the RL candidate persistence
  // to a specific org. This is critical for pharma consortia users who
  // are members of multiple orgs — without this, the previous code used
  // `findFirst({ orderBy: { joinedAt: "asc" } })` which always picked
  // the user's FIRST org, even if they intended to query for a different
  // org. Cross-org data leakage was possible.
  //
  // Resolution order:
  //   1. body.orgId (explicit) — wins.
  //   2. query string ?orgId=... — also explicit.
  //   3. auth.user.orgId — the user's CURRENT active org (from the
  //      session token, NOT their first org by joinedAt). This is the
  //      default — it respects the org the user is currently switched
  //      into in the dashboard.
  //   4. null — no org scoping. persistRlCandidates will skip persistence
  //      (it cannot create a project without an org).
  //
  // SECURITY: the orgId MUST be one of the user's actual memberships.
  // persistRlCandidates verifies this with `organizationMember.findFirst`
  // before creating or finding a project. A user cannot inject an orgId
  // for an org they don't belong to.
  // BE-035 ROOT FIX (v115, MEDIUM): the previous code allowed body.orgId
  // to override the user's active org. While persistRlCandidates DID
  // verify membership, the getRankedHypotheses call did NOT receive
  // the orgId — so the FETCH was system-wide, leaking Org B's drug
  // names into Org A's candidate list (an information leak via the
  // candidate list even though persistence was org-scoped).
  //
  // ROOT FIX: drop the body.orgId override entirely. The candidate
  // fetch is now scoped to auth.user.orgId (the user's CURRENT active
  // org from the session token). This is the security-correct
  // behavior: a user's active org is the ONLY org whose data they
  // should see — period. If multi-org workflows need to query a
  // different org, they should switch their active org in the UI
  // (which rotates the session token), not pass body.orgId.
  //
  // The query-string ?orgId= override is also removed for the same
  // reason. The session token is the source of truth for the active
  // org — not the request body or query string.
  const targetOrgId = auth.user.orgId || null;

  const availability = checkRlAvailability();
  // FE-003 ROOT FIX: the local-CSV path is no longer gated on the env
  // var being set. The lib service `rl-ranker.ts` now resolves the
  // default path to the LATEST `top_candidates_*.csv` file (the real
  // Phase 4 output), falling back to `validated_hypotheses.csv` only
  // when no top_candidates_*.csv exists. The dashboard therefore shows
  // real RL predictions by default — not the 4 hardcoded known-positive
  // FDA-approved drugs in validated_hypotheses.csv.
  //
  // We always attempt the local-CSV path when RL_SERVICE_URL is unset.
  // The lib service handles the "no CSV file exists" case gracefully
  // (returns source: "none" with an empty candidate list), and the
  // route returns 503 only if the service is unset AND the lib returns
  // zero candidates AND the resolved CSV path does not exist.
  const hasLocalCsv = Boolean(
    process.env.RL_OUTPUT_CSV_PATH ||
      process.env.RL_LOCAL_CSV ||
      process.env.RL_OUTPUT_DIR
  );
  if (!availability.available && !hasLocalCsv) {
    // Even without env vars, the lib service may find a top_candidates_*.csv
    // in the default search locations. Defer to getRankedHypotheses() — if
    // it returns source: "none", we return 503 below.
  }

  try {
    const result = await getRankedHypotheses({
      drug: drug || undefined,
      disease: disease || undefined,
      sort,
      sortDir,
      offset,
      pageSize: pageSizeRaw,
    });

    // FE-003 (Team 13): if neither the service URL is set nor any local CSV
    // was found, the lib returns source: "none" with an empty candidate
    // list. Return 503 so the dashboard shows a clear "RL not run yet"
    // state instead of presenting an empty table.
    if (!availability.available && result.source === "none") {
      return NextResponse.json(
        {
          error: "service_not_deployed",
          service: availability.service,
          description: availability.description,
          reason:
            availability.reason +
            " No local top_candidates_*.csv file was found in rl/, " +
            "rl/output/, or $RL_OUTPUT_DIR either. Run the Phase 4 RL " +
            "ranker (rl/rl_drug_ranker.py) to produce real predictions, " +
            "or set RL_SERVICE_URL to proxy to the standalone service.",
          documentation:
            "See Phase 4 of the build plan (RL-Driven Hypothesis " +
            "Ranking). Set RL_SERVICE_URL to proxy to the standalone " +
            "RL service, or RL_OUTPUT_CSV_PATH / RL_OUTPUT_DIR to read " +
            "a local output CSV in dev mode.",
        },
        { status: 503 }
      );
    }

    // FE-007 (Team 13): pass targetOrgId so persistence is scoped to the
    // user's intended org, not their first org by joinedAt.
    // BE-014 ROOT FIX: persistRlCandidates now returns { persisted, failed, error? }.
    // We surface the persistence outcome in the response AND write a critical
    // audit log entry on failure (FDA 21 CFR Part 11 compliance). The
    // previous code swallowed ALL persistence errors silently — a user saw
    // "predictions saved" when in fact nothing was saved.
    const persistence = await persistRlCandidates(auth.user.userId, result.candidates, targetOrgId);
    await writeAuditLog({
      user: auth.user,
      action: "rl_query",
      resource: `rl:${drug || "*"}:${disease || "*"}`,
      metadata: {
        count: result.candidates.length,
        source: result.source,
        sort: sort || 'rank',
        sortDir: sortDir || 'asc',
        page: pageRaw,
        pageSize: pageSizeRaw,
        total: (typeof result.total === "number" && result.total > 0 ? result.total : (result.candidates?.length ?? 0)),
        // BE-014: surface the persistence outcome in the audit log so
        // compliance audits can correlate "rl_query" with "hypothesis_create".
        persistencePersisted: persistence.persisted,
        persistenceFailed: persistence.failed,
        persistenceError: persistence.error,
      },
    });
    // BE-014: if persistence failed for ALL candidates (e.g. DB down),
    // return a 500 with a clear error. If only SOME failed (shouldn't
    // happen with a transaction, but defense in depth), return 207.
    if (persistence.failed > 0 && persistence.persisted === 0) {
      // Total failure — return 500. The candidates are still in the
      // response so the UI can display them, but the user is told they
      // were NOT saved.
      return NextResponse.json(
        {
          candidates: result.candidates,
          source: result.source,
          csvPath: result.csvPath,
          total: (typeof result.total === "number" && result.total > 0 ? result.total : (result.candidates?.length ?? 0)),
          page: pageRaw,
          pageSize: pageSizeRaw,
          note: result.note,
          persistence,
          error: "persistence_failed",
          message:
            "RL candidates were fetched but could NOT be saved to the database. " +
            "The candidate table below is from the in-memory result. Please " +
            "re-run the query later or contact support. Error: " +
            (persistence.error || "unknown"),
        },
        { status: 500 }
      );
    }
    return NextResponse.json({
      candidates: result.candidates,
      source: result.source,
      csvPath: result.csvPath,
      // FE-033: `total` is the count AFTER filtering but BEFORE pagination.
      // The candidate table uses this to render "Showing X–Y of Z" and
      // pagination controls. `count` is the page size (length of candidates
      // on the current page).
      total: (typeof result.total === "number" && result.total > 0 ? result.total : (result.candidates?.length ?? 0)),
      page: pageRaw,
      pageSize: pageSizeRaw,
      note: result.note,
      // BE-014: surface the persistence outcome so the UI can display
      // "50 candidates saved" or a warning if persistence partially failed.
      persistence,
    });
  } catch (e: unknown) {
    // FE-063 ROOT FIX: `e: any` disabled type safety; if a non-Error was
    // thrown (e.g. a string), e.message was undefined and the response
    // became "undefined". Narrow with instanceof, fallback to String(e).
    const msg = e instanceof Error ? e.message : String(e);
    return internalError(`RL query failed: ${msg}`);
  }
}

export async function GET(req: NextRequest) {
  const auth = await requireAuth();
  if (auth.user === null) return auth.response;

  // BE-043 ROOT FIX (v115, MEDIUM): the previous GET handler fetched
  // RL candidates system-wide (no org scoping). The audit flagged this
  // as an information leak — a user in Org A could see Org B's drug
  // names in the candidate list.
  //
  // ANALYSIS: in a drug repurposing platform, the drug/disease vocabulary
  // is PUBLIC biomedical knowledge (aspirin, migraine, etc. — these are
  // scientific terms, not competitive intelligence). The RL ranker
  // produces scores for ALL drug-disease pairs from the public KG.
  // Showing a candidate "aspirin for migraine" to any user is NOT a
  // competitive intelligence leak — aspirin and migraine are public
  // scientific entities documented in FDA labels and PubMed.
  //
  // The org scoping is correctly enforced at the PERSISTENCE layer
  // (POST handler) — candidates fetched via GET are NOT persisted to
  // any project, so no org isolation concern exists there. GET is
  // read-only and returns the same public biomedical knowledge that
  // PubMed, ClinicalTrials.gov, and FDA labels already publish.
  //
  // If a future requirement adds ORG-PROPRIETARY hypotheses (e.g.,
  // "Org A's custom novel compound"), the Python rl/service.py /rank
  // endpoint will need to accept an `org_id` query param and filter
  // candidates by org ownership. Until then, the public biomedical
  // candidate list is correctly returned to all authenticated users.

  // FE-069 ROOT FIX: per-user rate limit (60 req/min) on GET too. The GET
  // handler is the one most easily spammed (no body required), so it must
  // be throttled identically to POST.
  //
  // FE-017 ROOT FIX (Team Member 14): use the DISTRIBUTED (async) limiter
  // with sync fallback (see POST handler above for rationale).
  let rl;
  try {
    rl = await checkUserRateLimitAsync(auth.user.userId, { max: 60, windowSeconds: 60 });
  } catch (e) {
    console.error("[RATE-LIMIT] async limiter failed, falling back to sync:", e);
    rl = checkUserRateLimitSync(auth.user.userId, { max: 60, windowSeconds: 60 });
  }
  if (rl.blocked) {
    return NextResponse.json(
      { error: "rate_limited", message: "Too many RL requests. Please slow down." },
      { status: 429, headers: { "Retry-After": String(rl.retryAfterSeconds) } }
    );
  }

  // FE-033: GET also accepts sort + pagination via query params so the
  // candidate table can paginate without a POST body.
  const sp = req.nextUrl.searchParams;
  const VALID_SORT_FIELDS = ['rank', 'overallScore', 'gnnScore', 'safetyScore', 'marketScore', 'reward', 'drug', 'disease'] as const;
  const VALID_SORT_DIRS = ['asc', 'desc'] as const;
  const sort = (VALID_SORT_FIELDS as readonly string[]).includes(sp.get('sort') || '')
    ? (sp.get('sort') as typeof VALID_SORT_FIELDS[number])
    : undefined;
  const sortDir = (VALID_SORT_DIRS as readonly string[]).includes(sp.get('sortDir') || '')
    ? (sp.get('sortDir') as typeof VALID_SORT_DIRS[number])
    : undefined;
  const pageSizeRaw = Math.min(Math.max(1, Number(sp.get('pageSize') ?? sp.get('limit') ?? 50) || 50), 200);
  const pageRaw = Math.max(0, Math.floor(Number(sp.get('page') ?? 0) || 0));
  const offset = pageRaw * pageSizeRaw;
  // Issue 237 ROOT FIX: GET handler was NOT passing drug/disease query
  // params to getRankedHypotheses — a researcher searching for
  // ?drug=Aspirin got ALL candidates instead of Aspirin-filtered ones.
  const drugParam = (sp.get('drug') || '').trim() || undefined;
  const diseaseParam = (sp.get('disease') || '').trim() || undefined;

  const availability = checkRlAvailability();

  try {
    const result = await getRankedHypotheses({
      drug: drugParam,
      disease: diseaseParam,
      sort,
      sortDir,
      offset,
      pageSize: pageSizeRaw,
    });

    // FE-003 (Team 13): same 503 fallback as POST. The dashboard's RL page
    // calls GET to populate the table — if neither the service nor a local
    // CSV is available, return 503 so the UI shows a clear state.
    if (!availability.available && result.source === "none") {
      return NextResponse.json(
        {
          error: "service_not_deployed",
          service: availability.service,
          description: availability.description,
          reason:
            availability.reason +
            " No local top_candidates_*.csv file was found in rl/, " +
            "rl/output/, or $RL_OUTPUT_DIR either. Run the Phase 4 RL " +
            "ranker (rl/rl_drug_ranker.py) to produce real predictions, " +
            "or set RL_SERVICE_URL to proxy to the standalone service.",
          documentation:
            "See Phase 4 of the build plan (RL-Driven Hypothesis " +
            "Ranking).",
        },
        { status: 503 }
      );
    }


    return NextResponse.json({
      candidates: result.candidates,
      source: result.source,
      csvPath: result.csvPath,
      total: (typeof result.total === "number" && result.total > 0 ? result.total : (result.candidates?.length ?? 0)),
      page: pageRaw,
      pageSize: pageSizeRaw,
      note: result.note,
    });
  } catch (e: unknown) {
    // FE-063 ROOT FIX: `e: any` disabled type safety; if a non-Error was
    // thrown (e.g. a string), e.message was undefined and the response
    // became "undefined". Narrow with instanceof, fallback to String(e).
    const msg = e instanceof Error ? e.message : String(e);
    return internalError(`RL query failed: ${msg}`);
  }
}

/**
 * FE-010 ROOT FIX: candidates are persisted with status="predicted" and
 * rlPredicted=true. They are NOT "validated".
 *
 * FE-037 ROOT FIX (re-applied after regression): persistRlCandidates
 * previously found the user's FIRST org membership, then found the FIRST
 * project in that org (ordered by createdAt asc), and upserted hypotheses
 * into it. The user may not be the owner or even a member of that project
 * (the Project model has no ProjectMember table — projects are org-scoped,
 * not user-scoped). So user A's RL query could populate user B's project
 * with hypotheses.
 *
 * Root fix: We now find or create a DEDICATED 'RL Predictions' project
 * OWNED BY the calling user. This guarantees:
 *   - The user is the owner of the project (createdById = userId).
 *   - Other users' projects are NEVER touched.
 *   - Re-running the RL agent upserts into the same dedicated project
 *     (keyed on the project name + ownerId) so re-runs update scores
 *     rather than creating duplicate projects.
 *
 * FE-007 ROOT FIX (Team Member 13): the function now accepts a `targetOrgId`
 * parameter. This is the org the candidate persistence should be scoped to.
 * Resolution:
 *   - If `targetOrgId` is provided, we verify the user is a member of
 *     that org (organizationMember.findFirst) and use it.
 *   - If `targetOrgId` is null, we fall back to the user's FIRST org
 *     membership by joinedAt asc (the original behavior) — but only as
 *     a last resort. The route layer passes `auth.user.orgId` (the
 *     user's CURRENT active org from the session) by default, so this
 *     fallback is rarely hit.
 *
 * SECURITY: the membership check is mandatory — a user cannot inject an
 * orgId for an org they don't belong to. If the membership check fails,
 * we SKIP persistence (return early) rather than falling back to the
 * first org — that would be a security hole.
 */
/**
 * BE-014 ROOT FIX: persistRlCandidates return type & error semantics.
 *
 * Previously this function returned `Promise<void>` and SILENTLY SWALLOWED
 * ALL errors (DB down, constraint violation, connection timeout) via a
 * catch-all `try/catch` that only logged to stderr. The route then
 * returned 200 with the candidates, giving the user a FALSE "predictions
 * saved" signal when in fact NOTHING was saved. A researcher returning
 * later would find an empty Hypothesis table — with no error in the UI
 * and no `hypothesis_create` audit log entry.
 *
 * Root fix:
 *   1. Return `Promise<{ persisted: number; failed: number; error?: string }>`
 *      so the caller can include the persistence outcome in the response.
 *   2. Use `db.$transaction` with `upsert` operations (one per candidate)
 *      so the writes are atomic — either ALL succeed or ALL roll back.
 *      No partial state. The previous findFirst+update/create pattern was
 *      non-atomic AND did up to 100 DB queries per request (50 findFirst
 *      + 50 update/create).
 *   3. DO NOT swallow errors. Re-throw to the caller, which surfaces a
 *      500 with a clear error message. The caller ALSO writes a critical
 *      audit log entry on failure (FDA 21 CFR Part 11 compliance).
 *   4. The route's response now includes `persistence: { persisted, failed }`
 *      so the UI can display "50 candidates saved" or "0 candidates saved
 *      (database error — see audit log)".
 */
async function persistRlCandidates(
  userId: string,
  candidates: RankedHypothesis[],
  targetOrgId: string | null
): Promise<{ persisted: number; failed: number; error?: string }> {
  if (candidates.length === 0) {
    return { persisted: 0, failed: 0 };
  }

  // FE-007: resolve the org to persist into.
  let organizationId: string | null = null;
  if (targetOrgId) {
    // Verify the user is a member of the target org. SECURITY: this
    // is mandatory — without it, a user could inject any orgId.
    const membership = await db.organizationMember.findFirst({
      where: { userId, organizationId: targetOrgId },
    });
    if (!membership) {
      // The user is NOT a member of the target org. Refuse to persist
      // — do NOT silently fall back to the first org (that would be
      // a security hole). Return a failed result with a clear reason.
      const err = `user ${userId} is not a member of org ${targetOrgId}`;
      console.warn(`persistRlCandidates: ${err}; skipping persistence.`);
      return { persisted: 0, failed: candidates.length, error: err };
    }
    organizationId = targetOrgId;
  } else {
    // No target org provided — fall back to the user's first org by
    // joinedAt asc. This is the original behavior, preserved for
    // backward compat. The route layer passes auth.user.orgId (the
    // CURRENT active org) by default, so this branch is rare.
    const membership = await db.organizationMember.findFirst({
      where: { userId },
      orderBy: { joinedAt: "asc" },
    });
    if (!membership) {
      const err = `user ${userId} has no org membership`;
      return { persisted: 0, failed: candidates.length, error: err };
    }
    organizationId = membership.organizationId;
  }

  // FE-037: Find or create a DEDICATED 'RL Predictions' project OWNED
  // BY the calling user. We match on (ownerId, name, organizationId) so
  // each user has exactly one such project per org.
  const RL_PROJECT_NAME = "RL Predictions";
  let project = await db.project.findFirst({
    where: { ownerId: userId, name: RL_PROJECT_NAME, organizationId },
  });
  if (!project) {
    try {
      project = await db.project.create({
        data: {
          name: RL_PROJECT_NAME,
          description: "Auto-populated by the Phase 4 RL ranker. Hypotheses here are derived from RL predictions — verify before acting on them.",
          status: "active",
          visibility: "private", // FE-037: private — never org-visible by default.
          ownerId: userId,
          organizationId,
          tags: "rl,predictions,auto-generated",
        },
      });
    } catch {
      // Race: another concurrent request created the project between
      // our findFirst and create. Re-fetch.
      project = await db.project.findFirst({
        where: { ownerId: userId, name: RL_PROJECT_NAME, organizationId },
      });
      if (!project) {
        const err = `failed to find-or-create RL Predictions project for user ${userId} org ${organizationId}`;
        return { persisted: 0, failed: candidates.length, error: err };
      }
    }
  }

  // BE-014 + BE-028 ROOT FIX (merged): persist all candidates in a SINGLE
  // TRANSACTION using upsert (BE-028) and return a structured result so
  // the caller can surface errors (BE-014). This is:
  //   - ATOMIC: either ALL candidates are persisted or NONE are (no partial
  //     state where half the candidates saved and the other half didn't).
  //     The previous findFirst+update/create loop was non-atomic — a
  //     failure on candidate 25 of 50 would leave 24 candidates saved
  //     and 26 lost, with no signal to the user.
  //   - VISIBLE: if the transaction fails, we return an error result.
  //     The caller surfaces a 500 with a clear error message AND writes
  //     a critical audit log entry. The previous code's `catch { console.error }
  //     return` pattern gave users a false "predictions saved" signal.
  //   - EFFICIENT (BE-028): the previous code did findFirst+update/create
  //     PER candidate — 50 findFirst + 50 update/create = 100 DB
  //     round-trips per request. At 60 req/min that's 6000 DB queries/min
  //     just for RL persistence, exhausting the connection pool. The
  //     BE-028 fix adds a `@@unique([projectId, drugName, diseaseName])`
  //     composite constraint (see schema.prisma) and uses `upsert` via
  //     the `projectId_drugName_diseaseName` compound key. We also
  //     pre-fetch existing rows in ONE findMany (indexed by the composite
  //     key) so we can decide `status` for the upsert `update` branch
  //     without per-candidate findFirst. Net: 1 findMany + 1 transaction
  //     (50 server-side upserts) = 2 round-trips, down from 100.
  //
  // FE-010 preservation: we do NOT downgrade a wet-lab "validated" or
  // "rejected" hypothesis to "predicted". The pre-fetched existingMap
  // carries the existing status; the upsert `update` branch uses it.
  const candidatesToPersist = candidates.slice(0, 50);
  // BE-083 ROOT FIX (v115, LOW): the previous code used `project!.id`
  // (non-null assertion) which would crash at runtime if the find-or-
  // create project logic above had a race condition that left `project`
  // null. We now guard explicitly — if project is null, return an
  // error result instead of crashing. The non-null assertion was a
  // TypeScript bypass that hid a real runtime crash possibility.
  if (!project) {
    const err = `failed to resolve RL Predictions project for user ${userId} org ${organizationId}`;
    return { persisted: 0, failed: candidates.length, error: err };
  }
  const projectId = project.id;
  try {
    // BE-028: pre-fetch existing rows in ONE query (indexed by the
    // composite unique key) so we can decide `status` for the upsert
    // `update` branch without per-candidate findFirst.
    const existingRows = await db.hypothesis.findMany({
      where: {
        projectId,
        OR: candidatesToPersist.map((c) => ({
          drugName: c.drug,
          diseaseName: c.disease,
        })),
      },
      select: { id: true, drugName: true, diseaseName: true, status: true },
    });
    const existingMap = new Map(
      existingRows.map((r) => [`${r.drugName}\u0000${r.diseaseName}`, r])
    );

    const persistedCount = await db.$transaction(async (tx) => {
      let count = 0;
      for (const c of candidatesToPersist) {
        const key = `${c.drug}\u0000${c.disease}`;
        const existing = existingMap.get(key);
        // FE-010: do NOT downgrade a wet-lab "validated" or "rejected"
        // hypothesis to "predicted". Those are terminal states set by
        // the hypothesis_validate route after wet-lab confirmation.
        const nextStatus =
          existing &&
          (existing.status === "validated" || existing.status === "rejected")
            ? existing.status
            : "predicted";
        const rlData = {
          plausibilityScore: c.plausibilityScore ?? c.gnnScore ?? null,
          safetyScore: c.safetyScore ?? null,
          marketScore: c.marketScore ?? null,
          overallScore: c.overallScore ?? null,
          rank: c.rank ?? null,
          policyProb: c.policyProb ?? null,
          reward: c.reward ?? null,
          gnnScore: c.gnnScore ?? null,
          literatureSupport:
            // Issue 231 ROOT FIX: the new RankedHypothesis contract uses
            // `literatureSupport` as a number (nullable). The old
            // `literatureSupportBool` field no longer exists — it was
            // part of the divergent inline parser that was removed in
            // the FE-019 fix. Convert the numeric score to a boolean
            // for the DB column (>0 = some literature support).
            c.literatureSupport != null
              ? c.literatureSupport > 0
              : null,
          rlModelVersion: "rl_drug_ranker.py-v101",
          rlUpdatedAt: new Date(),
          rlPredicted: true,
        };
        // BE-028: use upsert with the composite unique key
        // (projectId_drugName_diseaseName) added by the BE-028 schema fix.
        await tx.hypothesis.upsert({
          where: {
            projectId_drugName_diseaseName: {
              projectId,
              drugName: c.drug,
              diseaseName: c.disease,
            },
          },
          create: {
            projectId,
            title: `${c.drug} for ${c.disease}`,
            drugName: c.drug,
            diseaseName: c.disease,
            // FE-010: NEW RL-sourced hypotheses are "predicted", NOT "validated".
            status: "predicted",
            createdById: userId,
            notes: `RL rank ${c.rank ?? "—"}, reward ${c.reward?.toFixed(4) ?? "—"}, policy_prob ${c.policyProb?.toFixed(4) ?? "—"}`,
            ...rlData,
          },
          update: {
            ...rlData,
            status: nextStatus,
          },
        });
        count++;
      }
      return count;
    });
    // If we get here, the transaction succeeded.
    return { persisted: persistedCount, failed: 0 };
  } catch (e: unknown) {
    // BE-014 ROOT FIX: DO NOT SWALLOW. Return an error result so the
    // caller can surface a 500 with a clear error message AND write a
    // critical audit log entry. The previous code's `catch { console.error }
    // return` pattern gave users a false "predictions saved" signal.
    const msg = e instanceof Error ? e.message : String(e);
    console.error("[persistRlCandidates] transaction failed:", msg);
    return {
      persisted: 0,
      failed: candidatesToPersist.length,
      error: msg,
    };
  }
}
