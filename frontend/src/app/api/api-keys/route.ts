import { NextRequest, NextResponse } from "next/server";
import { requireAuthRole, badRequest, requireCsrfOrSend, writeAuditLog, internalError } from "@/lib/api-helpers";
import { issueApiKey, listApiKeys } from "@/lib/services/api-keys";
// TASK-272: Zod validation on the POST body.
import { validateBody, ApiKeyCreateBody } from "@/lib/zod-schemas";

/**
 * GET /api/api-keys
 *
 * FE-014 ROOT FIX: Previously GET /api/api-keys called `listApiKeys(orgId)`
 * with NO userId argument, so a developer could see EVERY other developer's
 * and admin's API key prefixes/names/last-used timestamps. The revoke
 * endpoint correctly passed userId for non-admins — the list endpoint was
 * inconsistent and leaked information.
 *
 * Root fix: pass the caller's userId as the owner filter UNLESS the caller
 * is an admin/owner (org-wide oversight). This matches the revoke endpoint's
 * posture exactly.
 *
 * FE-011: CSRF protection applied to POST.
 */
export async function GET() {
  const auth = await requireAuthRole("developer");
  if (auth.user === null) return auth.response;
  if (!auth.user.orgId) return badRequest("No active organization");
  // Non-admins only see their own keys. Admin/owner see all keys in the org.
  const ownerFilter =
    auth.user.role === "admin" || auth.user.role === "owner"
      ? undefined
      : auth.user.userId;
  const keys = await listApiKeys(auth.user.orgId, ownerFilter);
  return NextResponse.json({ items: keys });
}

/**
 * POST /api/api-keys
 *
 * TASK-267 ROOT FIX: Add audit logging for API key creation. The previous
 * code created an API key with NO audit trail — a developer could create
 * a key, exfiltrate data via the API, then revoke the key, and there
 * would be NO record of who created it or when. For a pharma platform
 * where API access is a billable enterprise feature, this is a financial
 * compliance gap (SOC 2 CC6.1: "logical access controls").
 *
 * The audit log entry is CRITICAL — if it fails, the request is ABORTED
 * (the API key creation MUST be auditable, otherwise it could be created
 * silently and used for data exfiltration).
 *
 * TASK-272: Zod validation on the POST body. The schema rejects empty
 * names, names >200 chars, and non-string names.
 */
export async function POST(req: NextRequest) {
  // FE-011: CSRF protection on every state-changing route.
  const csrf = await requireCsrfOrSend(req);
  if (csrf.response) return csrf.response;

  const auth = await requireAuthRole("developer");
  if (auth.user === null) return auth.response;
  if (!auth.user.orgId) return badRequest("No active organization");

  let body: unknown;
  try {
    body = await req.json();
  } catch {
    return badRequest("Invalid JSON");
  }

  // TASK-272: validate the body with Zod.
  const parsed = validateBody(ApiKeyCreateBody, body);
  if (!parsed.ok) return parsed.response;
  const name = parsed.data.name.trim();

  const created = await issueApiKey(auth.user.orgId, auth.user.userId, name);

  // TASK-267: critical audit log — API key creation MUST be auditable.
  const auditResult = await writeAuditLog({
    user: auth.user,
    action: "api_key_create",
    resource: `api-key:${created.id}`,
    metadata: {
      keyName: name,
      keyPrefix: created.prefix,
      organizationId: auth.user.orgId,
    },
    critical: true,
  });
  if (!auditResult.ok) {
    // Best-effort cleanup — revoke the key we just issued so it can't
    // be used without an audit trail. If this also fails, the key is
    // "orphaned" but at least the operator can detect the audit-log
    // failure via the dead-letter table.
    try {
      const { revokeApiKey } = await import("@/lib/services/api-keys");
      // We already checked `auth.user.orgId` is truthy at the top of the
      // handler (the `if (!auth.user.orgId) return badRequest(...)` guard).
      // Use a non-null assertion alternative — cast to string.
      await revokeApiKey(auth.user.orgId as string, created.id, undefined);
    } catch (e) {
      console.error("[API-KEYS] Failed to revoke unaudited key:", e);
    }
    return internalError("Audit log write failed. API key creation aborted for compliance.");
  }

  return NextResponse.json(created, { status: 201 });
}
