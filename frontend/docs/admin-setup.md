# Platform Admin Setup Guide

**Tasks 261, 269, 271, 279 — root-forensic implementation.**

This document explains the new platform-admin architecture introduced by
Tasks 261-280, how to grant platform-admin access to SaaS operator staff,
and how the audit/notification/monitoring systems work together.

---

## 1. The Two-Field Authorization Model

The platform now has TWO independent authorization fields on every `User`:

| Field           | Type        | Purpose                                         | Grantable via API? |
|-----------------|-------------|-------------------------------------------------|--------------------|
| `role`          | `UserRole`  | Functional role in the user's org (researcher, pi, admin, owner, etc.). Drives in-app RBAC. | Yes — `PATCH /api/admin/users` (admin-only). |
| `platformRole`  | `PlatformRole` | SaaS-operator flag (`none` \| `admin`). Gates `/api/admin/*` (platform console). | **NO** — direct DB access only. |

### Why two fields?

The previous architecture overloaded `role` for both concerns. The audit
(Task 261) found that any user promoted to `owner` could:

- Enumerate every user in every org via `GET /api/admin/users`.
- Read every audit log across all tenants via `GET /api/audit-logs`.
- Suspend any user system-wide.

The prior `platformOwner` enum-value patch reduced the blast radius but
kept the coupling — granting platform access required changing `role`,
which changed in-app permissions as a side effect.

The clean fix is the **separate `platformRole` field**. The two fields
are independently grantable and independently revocable — the OWASP ASVS
V1.2 "Separation of Duties" pattern for multi-tenant SaaS.

### Where each field is checked

| Route namespace        | Gate                                | Notes |
|------------------------|-------------------------------------|-------|
| `/api/admin/*`         | `requirePlatformAdmin()` (platformRole === "admin") | The new middleware in `lib/auth/require-platform-admin.ts`. |
| `/api/audit-logs`      | `requireAdmin()` + `isPlatformAdmin()` for cross-tenant access | Org-scoped admins see their own org's logs; platform admins see system-wide. |
| `/api/team`            | `requireAuth()` (any logged-in user) | Scoped to the user's own org via `auth.user.orgId`. |
| `/api/billing/*`       | `requireAuthRole("billing")`        | Org-scoped. |
| `/api/system/status`   | `requirePlatformAdmin()`            | Returns cross-tenant infra status — operator-only. |

---

## 2. Granting Platform-Admin Access

Platform-admin access is settable **ONLY via direct DB access**. There is
no API route that can grant it — this is fail-closed (the only way to
become a platform admin is to already have DB access, which is the SaaS
operator's staff).

### PostgreSQL

```sql
-- Promote a user to platform admin.
UPDATE "User" SET "platformRole" = 'admin' WHERE email = 'operator@your-saas.com';

-- Verify the change.
SELECT id, email, role, "platformRole" FROM "User" WHERE email = 'operator@your-saas.com';

-- Revoke platform admin access (e.g. when an operator leaves the company).
UPDATE "User" SET "platformRole" = 'none' WHERE email = 'operator@your-saas.com';
```

### After granting, force a re-login

The user's `platformRole` is stamped into their JWT access token at login
time. Existing tokens issued BEFORE the DB change still have
`platformRole = "none"` and will be denied access until the user
re-authenticates.

The token refresh path (`rotateRefreshToken`) also reads `platformRole`
from the DB on every refresh (every 15 minutes), so the change takes
effect within 15 minutes even if the user doesn't manually re-login.

For IMMEDIATE revocation (e.g. a compromised operator account), revoke
all their refresh tokens to force a re-login on the next request:

```sql
UPDATE "RefreshToken" SET "revokedAt" = NOW()
  WHERE "userId" = (SELECT id FROM "User" WHERE email = 'operator@your-saas.com')
  AND "revokedAt" IS NULL;
```

---

## 3. The `requirePlatformAdmin` Middleware

All `/api/admin/*` routes MUST use this middleware instead of the old
`requireAdmin()`. The middleware enforces:

1. **Authentication** — 401 if not logged in.
2. **Platform-admin gate** — 403 if `platformRole !== "admin"`. The 403
   body does NOT leak the user's current platformRole (just "forbidden").
3. **Rate limiting** — 1 req/sec per platform admin on state-changing
   methods (Task 273). GET requests are exempt (the admin console polls
   `/api/system/status` every 5s).
4. **CSRF protection** — double-submit cookie pattern on POST/PATCH/PUT/DELETE.
5. **Audit logging of denials** — every 403 writes an audit log entry
   with action `platform_admin_denied` so the SaaS operator can detect
   probing.

### Usage

```typescript
import { requirePlatformAdmin } from "@/lib/auth/require-platform-admin";

export async function GET(req: NextRequest) {
  const auth = await requirePlatformAdmin(req);
  if (auth.user === null) return auth.response;
  // ... route handler ...
}
```

---

## 4. Audit Logging (Task 267)

Every privileged action now writes a CRITICAL audit log entry. If the
audit log write fails, the request is ABORTED with a 500 — the action
MUST be auditable (FDA 21 CFR Part 11).

### Privileged actions that are audit-logged

| Action                          | Route                                  | Critical? |
|---------------------------------|----------------------------------------|-----------|
| User role/status change         | `PATCH /api/admin/users`               | Yes |
| User deletion (soft-delete)     | `DELETE /api/admin/users`              | Yes |
| API key creation                | `POST /api/api-keys`                   | Yes |
| API key revocation              | `POST /api/api-keys/[id]/revoke`       | Yes |
| Hypothesis validation writeback | `POST /api/hypothesis/validate`        | Yes |
| Project comment creation        | `POST /api/projects/[id]/comments`     | Yes |
| Billing plan change             | `POST /api/billing/subscription`       | Yes |
| Platform-admin denial (403)     | (any /api/admin/* route)               | Best-effort |

### Reading audit logs

```bash
# Org-scoped (admin sees their own org's logs):
curl -b cookies.txt https://your-saas.com/api/audit-logs?limit=100

# System-wide (platform admin only):
curl -b cookies.txt https://your-saas.com/api/audit-logs?limit=1000

# Dead-letter entries (platform admin only):
curl -b cookies.txt https://your-saas.com/api/audit-logs?dead_letter=true
```

---

## 5. Notification Triggers (Task 268)

Three notification triggers are now wired into the platform:

| Trigger                          | Fired by                                  | Recipients |
|----------------------------------|-------------------------------------------|------------|
| New project comment              | `POST /api/projects/[id]/comments`        | All project members except the commenter. |
| Billing invoice ready            | `changePlan()` in `lib/services/billing.ts` | Org members with `billing` or `owner` role. |
| Hypothesis validation complete   | `POST /api/hypothesis/validate`           | The submitter + the org's PIs. |

Each trigger is **best-effort (non-blocking)** — a notification failure
does not break the user action that triggered it. Failures are logged to
stderr so operators can monitor for systemic issues.

Users read their notifications via `GET /api/notifications` and mark
them as read via `POST /api/notifications/[id]/read`.

---

## 6. System Status & Monitoring (Tasks 265, 280)

`GET /api/system/status` aggregates REAL connectivity checks from every
backend service:

| Service                        | Check                                            | Critical? |
|--------------------------------|--------------------------------------------------|-----------|
| PostgreSQL                     | `SELECT 1` via Prisma                            | Yes — platform down if unreachable. |
| Neo4j (Phase 2 KG)             | HTTP ping with 2s timeout                        | Yes — platform down if unreachable. |
| Apache Airflow (Phase 1)       | HTTP ping `/api/v1/health`                       | No — degraded if unreachable. |
| Graph Transformer (Phase 3)    | HTTP ping `/health` or model-artifact check      | No — degraded if unreachable. |
| RL Agent (Phase 4)             | HTTP ping `/health`                              | No — degraded if unreachable. |
| MLflow (experiment tracking)   | HTTP ping `/health`                              | No — degraded if unreachable. |

### Response shape

```json
{
  "overall": "operational" | "degraded" | "down",
  "health": {
    "overall": "...",
    "services": { "postgres": {...}, "neo4j": {...}, ... },
    "generatedAt": "2026-07-16T..."
  },
  "services": { ... },  // legacy per-service keys (backwards compat)
  "generatedAt": "..."
}
```

### Monitoring alert (Task 280)

The route returns **HTTP 503** when `overall === "down"`. Configure your
uptime monitor (Pingdom, Datadog, etc.) to alert on:

- HTTP status 503 from `/api/system/status` — a critical service is down.
- HTTP status 5xx from `/api/audit-logs` — the audit log DB is unreachable.
- HTTP status 5xx from `/api/notifications` — the notifications DB is unreachable.

The 503 response body still contains the per-service breakdown so the
on-call operator can see WHICH critical service is down.

---

## 7. Zod Validation (Task 272)

All admin/audit/notification routes now validate their input with Zod
schemas defined in `lib/zod-schemas.ts`:

| Schema                 | Used by                                |
|------------------------|----------------------------------------|
| `AdminUserPatchBody`   | `PATCH /api/admin/users`               |
| `AuditLogsQuery`       | `GET /api/audit-logs`                  |
| `NotificationsQuery`   | `GET /api/notifications`               |
| `TeamQuery`            | `GET /api/team`                        |
| `ApiKeyCreateBody`     | `POST /api/api-keys`                   |

On validation failure, the route returns 400 with a structured error
envelope listing every field error.

---

## 8. Testing (Tasks 274-278)

The test suite is in `tests/api/`:

| Test file                          | Verifies                                                       |
|------------------------------------|----------------------------------------------------------------|
| `admin.security.test.ts`           | Non-admins get 403 on `/api/admin/*`. Platform admins pass.    |
| `audit-logs.test.ts`               | Privileged actions are logged; org-scoped filtering works.     |
| `notifications.test.ts`            | Notification triggers fire correctly.                          |
| `system-status.test.ts`            | Status aggregates from all services; 503 when critical down.   |
| `team.test.ts`                     | Team membership is real (wired to OrganizationMember table).   |

Run the tests:

```bash
cd frontend
npm test
```

---

## 9. Migration Notes

If you're upgrading from a prior version that used `role === "platformOwner"`
for platform-superuser access:

1. Run the migration `20260716000001_task261_269_add_platform_role` to
   add the `platformRole` column (default `none`).
2. Grant `platformRole = 'admin'` to existing SaaS operator staff:
   ```sql
   UPDATE "User" SET "platformRole" = 'admin' WHERE role = 'platformOwner';
   ```
3. The `role === "platformOwner"` check in `requireAdmin()` is KEPT for
   backwards compat (so existing operator accounts don't lose access
   until step 2 completes). It can be removed once all operators are
   migrated to `platformRole = 'admin'`.
4. Existing JWT access tokens do NOT carry `platformRole` — users must
   re-login to get a token with the new claim. The `verifyAccessToken`
   function treats missing `platformRole` as `"none"` (fail-closed), so
   existing tokens CANNOT access `/api/admin/*` until re-login.
