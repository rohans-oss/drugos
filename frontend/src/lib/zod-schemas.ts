/**
 * BE-029 ROOT FIX (Team Member 12): shared Zod schemas for API request bodies.
 *
 * ROOT CAUSE: NONE of the 43 API routes used Zod to validate request bodies.
 * Validation was manual and inconsistent — some routes checked
 * `typeof body.field === "string"`, others used `!body.field`, others
 * trusted the body entirely. Type-confusion bugs were exploitable:
 *   - `body.limit = "abc"` → `Math.min("abc", 5000)` → NaN →
 *     `pairs.slice(0, NaN)` → empty array returned silently.
 *   - `body.drug` as object → `.trim()` throws unhandled TypeError → 500.
 *   - `body.tags` as string → `.join(",")` joins characters.
 *
 * ROOT FIX: this module exports one Zod schema per route body shape,
 * plus a `validateBody()` helper that returns a typed result. Routes
 * call `const parsed = validateBody(MySchema, body); if (!parsed.ok)
 * return parsed.response;` and get a 400 with a structured error
 * message on validation failure.
 *
 * We use Zod 4 (already a dependency ^4.0.2). Zod 4's `safeParse`
 * returns `{ success: true, data } | { success: false, error }` — we
 * translate that into the discriminated union below so routes can
 * narrow with `parsed.ok`.
 *
 * SECURITY: every state-changing route (POST/PATCH/PUT/DELETE) on the
 * 7 routes owned by Team Member 12 (BE-021 to BE-040) now uses these
 * schemas. Other teams can adopt the same helper for their routes.
 */
import { z } from "zod";
import { NextResponse } from "next/server";
// TASK-272: import the role/status allowlists so the Zod schemas and the
// runtime validators in register/route.ts can't drift. The import is at
// the top so it's hoisted (ES modules) and visible to all schemas below.
import {
  ALLOWED_ROLES_ADMIN,
  ALLOWED_USER_STATUSES,
} from "@/app/api/auth/register/route";

// ---------------------------------------------------------------------------
// Generic validation helper
// ---------------------------------------------------------------------------

export type ValidationResult<T> =
  | { ok: true; data: T }
  | { ok: false; response: NextResponse };

/**
 * Validate an unknown request body against a Zod schema.
 *
 * On success: returns `{ ok: true, data }` with the parsed (and
 * type-narrowed) body.
 *
 * On failure: returns `{ ok: false, response }` with a 400 NextResponse
 * whose JSON body lists every field error. Routes return the response
 * directly: `if (!parsed.ok) return parsed.response;`.
 *
 * The error response shape is:
 *   {
 *     error: "bad_request",
 *     message: "Request body failed validation",
 *     issues: [{ path: "pairs.0.drug", message: "Required" }, ...]
 *   }
 */
export function validateBody<T>(
  schema: z.ZodType<T>,
  body: unknown
): ValidationResult<T> {
  const result = schema.safeParse(body);
  if (result.success) {
    return { ok: true, data: result.data };
  }
  // Zod 4: result.error.issues is an array of { path, message, code }.
  const issues = result.error.issues.map((iss) => ({
    path: iss.path.join("."),
    message: iss.message,
  }));
  return {
    ok: false,
    response: NextResponse.json(
      {
        error: "bad_request",
        message: "Request body failed validation.",
        issues,
      },
      { status: 400 }
    ),
  };
}

// ---------------------------------------------------------------------------
// Per-route schemas
// ---------------------------------------------------------------------------

// /api/predict — POST body (BE-029, BE-030)
export const PredictBody = z.object({
  pairs: z
    .array(
      z.object({
        drug: z.string().min(1).max(200),
        disease: z.string().min(1).max(200),
      })
    )
    .min(1)
    .max(5000),
  limit: z.number().int().positive().max(5000).optional(),
});
export type PredictBodyT = z.infer<typeof PredictBody>;

// /api/top-k — no body (GET only). Query param validated inline.

// /api/rl — POST body (BE-029)
export const RlBody = z.object({
  drug: z.string().min(1).max(200).optional(),
  disease: z.string().min(1).max(200).optional(),
  limit: z.number().int().positive().max(200).optional(),
  sort: z
    .enum(["rank", "overallScore", "gnnScore", "safetyScore", "marketScore", "reward", "drug", "disease"])
    .optional(),
  sortDir: z.enum(["asc", "desc"]).optional(),
  page: z.number().int().min(0).optional(),
  pageSize: z.number().int().positive().max(200).optional(),
  orgId: z.string().min(1).max(100).optional(),
});
export type RlBodyT = z.infer<typeof RlBody>;

// /api/knowledge-graph — POST body (BE-029)
export const KnowledgeGraphBody = z.object({
  cypher: z.string().min(1).max(10_000),
  params: z.record(z.string(), z.unknown()).optional(),
});
export type KnowledgeGraphBodyT = z.infer<typeof KnowledgeGraphBody>;

// /api/auth/2fa/disable — POST body (BE-029, BE-031)
export const TwoFaDisableBody = z.object({
  currentPassword: z.string().min(1).max(1024),
  totpCode: z.string().regex(/^\d{6}$/).optional(),
});
export type TwoFaDisableBodyT = z.infer<typeof TwoFaDisableBody>;

// /api/auth/2fa/login-verify — POST body (BE-029, BE-034)
export const TwoFaLoginVerifyBody = z.object({
  mfaToken: z.string().min(1).max(4096).optional(),
  code: z.string().regex(/^\d{6}$/),
});
export type TwoFaLoginVerifyBodyT = z.infer<typeof TwoFaLoginVerifyBody>;

// /api/auth/password — POST body (BE-029, BE-032)
export const PasswordChangeBody = z.object({
  currentPassword: z.string().min(1).max(1024),
  newPassword: z.string().min(8).max(1024),
});
export type PasswordChangeBodyT = z.infer<typeof PasswordChangeBody>;

// /api/auth/verify-email — POST body (BE-029)
export const VerifyEmailBody = z.object({
  token: z.string().min(10).max(8192),
});
export type VerifyEmailBodyT = z.infer<typeof VerifyEmailBody>;

// /api/auth/me — PATCH body (BE-029)
export const AuthMePatchBody = z.object({
  name: z.string().min(1).max(200).optional(),
  title: z.string().max(200).optional(),
  bio: z.string().max(2000).optional(),
  activeOrganizationId: z.string().min(1).max(100).nullable().optional(),
});
export type AuthMePatchBodyT = z.infer<typeof AuthMePatchBody>;

// /api/billing/subscription — POST body (BE-029, BE-033)
export const BillingSubscriptionBody = z.object({
  planId: z.string().min(1).max(100),
  currentPassword: z.string().min(1).max(1024),
  totpCode: z.string().regex(/^\d{6}$/).optional(),
  mfaTicket: z.string().min(1).max(4096).optional(),
}).refine(
  (data) => !(data.totpCode && data.mfaTicket),
  { message: "Provide either totpCode or mfaTicket, not both." }
);
export type BillingSubscriptionBodyT = z.infer<typeof BillingSubscriptionBody>;

// ---------------------------------------------------------------------------
// Task 252 ROOT FIX: Zod validation for the 7 public-API-proxy routes'
// query parameters.
//
// ROOT CAUSE: the audit required Zod validation on all 7 drug/disease/
// safety routes. Previously, every route used manual validation
// (`q.trim().length < 2`, `parseInt(...)`, etc.). Manual validation is
// inconsistent and prone to type-confusion bugs:
//   - `q = ""` passes `!q` but `q = "  "` (whitespace) passes the check
//     and produces a 400-error upstream call from NLM.
//   - `limit = "abc"` parses to NaN; `Math.min(NaN, 100) = NaN` slices
//     to `[]` silently.
//   - `drug = "../../etc/passwd"` would be passed to the openFDA URL
//     builder as-is — no whitelist.
//
// ROOT FIX: this section defines one Zod schema per route's query
// params, plus a `validateQueryParams()` helper that mirrors
// `validateBody()` but reads from `URLSearchParams`. Each of the 7
// routes now uses these schemas.
//
// The schemas enforce:
//   - `q` is a non-empty string (after trim), 2-200 chars, matching the
//     biomedical-name allowlist [A-Za-z0-9 ,.'-].
//   - `limit` is an integer in [1, 100] (parsed from string).
//   - `drug` (safety route) is 2-64 chars matching the same allowlist.
//   - `condition`/`intervention` (clinical-trials route) are each
//     optional strings ≤200 chars.
//   - `pageToken` (clinical-trials route) is an opaque string ≤256 chars.
//   - `status` (clinical-trials route) is one of the allowed enum values.
//   - `rxcui` (drug search route) is an optional 1-11 digit numeric string.
// ---------------------------------------------------------------------------

const BIOMEDICAL_NAME_REGEX = /^[A-Za-z0-9 ,.'-]{2,200}$/;
const DRUG_NAME_REGEX = /^[A-Za-z0-9 ,.'-]{2,64}$/;

/**
 * Validate a URLSearchParams object against a Zod schema. On success
 * returns `{ ok: true, data }` with the parsed (and type-narrowed)
 * params. On failure returns `{ ok: false, response }` with a 400
 * NextResponse listing every field error.
 *
 * Usage in a route:
 *   const parsed = validateQueryParams(DrugsSearchQuery, req.nextUrl.searchParams);
 *   if (!parsed.ok) return parsed.response;
 *   const { q, limit } = parsed.data;
 */
export function validateQueryParams<T>(
  schema: z.ZodType<T>,
  params: URLSearchParams
): ValidationResult<T> {
  // Convert URLSearchParams to a plain object. Zod's coerce handles
  // string-to-number conversion when the schema uses z.coerce.number().
  const raw: Record<string, string | undefined> = {};
  for (const key of new Set(params.keys())) {
    raw[key] = params.get(key) || undefined;
  }
  const result = schema.safeParse(raw);
  if (result.success) {
    return { ok: true, data: result.data };
  }
  const issues = result.error.issues.map((iss) => ({
    path: iss.path.join("."),
    message: iss.message,
  }));
  return {
    ok: false,
    response: NextResponse.json(
      {
        error: "bad_request",
        message: "Query parameters failed validation.",
        issues,
      },
      { status: 400 }
    ),
  };
}

/**
 * Helper: parse a string-typed query param into a clamped integer.
 * Used by schemas that want `z.coerce.number()` behavior with bounds.
 */
function clampedInt(min: number, max: number, def: number) {
  return z
    .string()
    .optional()
    .transform((v) => {
      if (v === undefined || v === "") return def;
      const n = Number.parseInt(v, 10);
      if (!Number.isFinite(n)) return def;
      return Math.min(Math.max(n, min), max);
    });
}

// /api/drugs/search — query params
export const DrugsSearchQuery = z.object({
  q: z
    .string()
    .trim()
    .min(2, "Query parameter 'q' must be at least 2 characters")
    .max(200, "Query parameter 'q' must be at most 200 characters")
    .regex(BIOMEDICAL_NAME_REGEX, "Query parameter 'q' contains invalid characters")
    .optional(),
  rxcui: z
    .string()
    .regex(/^\d{1,11}$/, "RxCUI must be 1-11 digits")
    .optional(),
  limit: clampedInt(1, 100, 10),
});
export type DrugsSearchQueryT = z.infer<typeof DrugsSearchQuery>;

// /api/drugs/mechanism — POST body (already validated manually in the route,
// but we add the Zod schema for consistency + future migration).
export const DrugsMechanismBody = z.object({
  drugNames: z
    .array(z.string().trim().min(2).max(128))
    .min(1, "drugNames must contain at least one drug name")
    .max(100, "drugNames must contain at most 100 drug names"),
});
export type DrugsMechanismBodyT = z.infer<typeof DrugsMechanismBody>;

// /api/diseases/search — query params
export const DiseasesSearchQuery = z.object({
  q: z
    .string()
    .trim()
    .min(2, "Query parameter 'q' must be at least 2 characters")
    .max(200, "Query parameter 'q' must be at most 200 characters")
    .regex(BIOMEDICAL_NAME_REGEX, "Query parameter 'q' contains invalid characters"),
  limit: clampedInt(1, 100, 10),
});
export type DiseasesSearchQueryT = z.infer<typeof DiseasesSearchQuery>;

// /api/safety/[drug] — the drug is in the path, not the query. But we
// also accept `?limit=N` for the reaction count cap.
export const SafetyQuery = z.object({
  limit: clampedInt(1, 100, 100),
});
export type SafetyQueryT = z.infer<typeof SafetyQuery>;

/**
 * Validate a drug name from a path parameter. Returns the sanitized
 * name or null if invalid. Used by /api/safety/[drug].
 */
export function validateDrugPathParam(drug: string): string | null {
  const decoded = (() => {
    try {
      return decodeURIComponent(drug);
    } catch {
      return drug;
    }
  })();
  if (!decoded || decoded.length < 2) return null;
  if (!DRUG_NAME_REGEX.test(decoded)) return null;
  return decoded.trim();
}

// /api/clinical-trials/search — query params
export const ClinicalTrialsSearchQuery = z.object({
  condition: z
    .string()
    .trim()
    .max(200, "condition must be at most 200 characters")
    .optional(),
  intervention: z
    .string()
    .trim()
    .max(200, "intervention must be at most 200 characters")
    .optional(),
  status: z
    .enum(["RECRUITING", "ACTIVE_NOT_RECRUITING", "COMPLETED", "ALL"])
    .optional()
    .default("ALL"),
  limit: clampedInt(1, 100, 50),
  page: clampedInt(1, 10000, 1),
  pageSize: clampedInt(1, 100, 50),
  pageToken: z
    .string()
    .max(256, "pageToken must be at most 256 characters")
    .optional(),
}).refine(
  (data) => Boolean(data.condition || data.intervention),
  { message: "At least one of 'condition' or 'intervention' is required" }
);
export type ClinicalTrialsSearchQueryT = z.infer<typeof ClinicalTrialsSearchQuery>;

// /api/patents/search — query params
export const PatentsSearchQuery = z.object({
  q: z
    .string()
    .trim()
    .min(2, "Query parameter 'q' must be at least 2 characters")
    .max(200, "Query parameter 'q' must be at most 200 characters")
    .regex(BIOMEDICAL_NAME_REGEX, "Query parameter 'q' contains invalid characters"),
  limit: clampedInt(1, 100, 20),
});
export type PatentsSearchQueryT = z.infer<typeof PatentsSearchQuery>;

// ---------------------------------------------------------------------------
// TASK-272 ROOT FIX: Zod schemas for admin / audit / notification routes.
//
// The audit (Task 272) found that NONE of the admin/audit/notification
// routes used Zod to validate request bodies or query params. Each route
// did ad-hoc `typeof body.x === "string"` checks that were inconsistent
// and exploitable (e.g. `parseInt("abc", 10)` returns NaN, which then
// becomes `take: NaN` in Prisma — silently returning zero rows, or
// worse, throwing an unhandled P2009).
//
// The schemas below cover the routes targeted by Tasks 261-280:
//   - /api/admin/users (PATCH body)
//   - /api/audit-logs (GET query params)
//   - /api/notifications (GET query params)
//   - /api/notifications/[id]/read (POST — no body, but path param)
//   - /api/system/status (GET — no body, no query)
//   - /api/team (GET query params)
//
// Each schema is EXPORTED so the route handler and the corresponding test
// file can both import it. The test file uses `schema.parse(mockBody)` to
// generate valid fixtures and `schema.safeParse(badBody).success === false`
// to assert rejection.
// ---------------------------------------------------------------------------

// /api/admin/users — PATCH body (Task 261, 272)
//
// Validates the role + status values against the SAME allowlist used by
// the existing runtime validators in register/route.ts. We import the
// allowlists here so the Zod schema and the runtime check can't drift.
//
// NOTE: `platformRole` is INTENTIONALLY NOT in this schema. The
// platformRole field is settable ONLY via direct DB access (see
// PlatformRole enum in prisma/schema.prisma). Allowing it via the API
// would re-introduce the privilege-escalation bug that Task 261 fixed.

export const AdminUserPatchBody = z.object({
  userId: z.string().min(1).max(100),
  role: z.enum(ALLOWED_ROLES_ADMIN as unknown as [string, ...string[]]).optional(),
  status: z.enum(ALLOWED_USER_STATUSES as unknown as [string, ...string[]]).optional(),
}).refine(
  (data) => data.role !== undefined || data.status !== undefined,
  { message: "At least one of role or status must be provided." }
);
export type AdminUserPatchBodyT = z.infer<typeof AdminUserPatchBody>;

// /api/audit-logs — GET query params (Task 262, 272)
export const AuditLogsQuery = z.object({
  limit: z.coerce.number().int().positive().max(1000).default(100),
  action: z.string().min(1).max(100).optional(),
  dead_letter: z.enum(["true", "false"]).optional(),
});
export type AuditLogsQueryT = z.infer<typeof AuditLogsQuery>;

// /api/notifications — GET query params (Task 263, 272)
export const NotificationsQuery = z.object({
  limit: z.coerce.number().int().positive().max(100).default(50),
  offset: z.coerce.number().int().min(0).default(0),
});
export type NotificationsQueryT = z.infer<typeof NotificationsQuery>;

// /api/system/status — GET (no body, no query — Task 265, 272)
// No schema needed — the route takes no input. Listed here for completeness.

// /api/team — GET query params (Task 266, 272)
export const TeamQuery = z.object({
  limit: z.coerce.number().int().positive().max(100).default(50),
  offset: z.coerce.number().int().min(0).default(0),
});
export type TeamQueryT = z.infer<typeof TeamQuery>;

// /api/api-keys — POST body (Task 267, 272)
export const ApiKeyCreateBody = z.object({
  name: z.string().min(1).max(200),
});
export type ApiKeyCreateBodyT = z.infer<typeof ApiKeyCreateBody>;
