# DrugOS Upgrade Notes

## v0.5.0 — Login flow + admin role fix

This release fixes the two issues you reported:

1. **Login page appeared to "refresh" and stay on the login page.**
   Root cause: the `SessionProvider` started with `loading: false, user: null`
   on the client, which caused the AppShell auth guard to immediately redirect
   to `/login` before the session had a chance to load — even for users who
   had a valid auth cookie. Fix: introduced a `mounted` flag that starts
   `false` on both server and first client render (so no hydration mismatch),
   then flips to `true` inside a `useEffect`. The `loading` flag is now
   `true` whenever `mounted === false` OR a fetch is in flight, so the
   AppShell auth guard waits for the session to resolve before deciding
   whether to redirect.
2. **Admin role was missing from the Register dropdown.** Added it back.
   The full list is now: Researcher, Data Scientist, Principal Investigator,
   Admin, Business Development, Developer, Viewer.

### Files modified
- `src/components/drugos/session-provider.tsx` — added `mounted` flag and `fetching` state; `loading` is now `!mounted || fetching`. Default context value also reports `loading: true` so consumers see the splash immediately.
- `src/components/drugos/app-router.tsx` — added `<SelectItem value="admin">Admin</SelectItem>` back to the Register page role dropdown.

---

## v0.4.0 — Role-Based Access Control + Real Profile & Team Data

This release fixes the issues you reported from the screenshots:

1. **Researchers can no longer see admin/billing/developer sections.**
   The sidebar now filters items based on the logged-in user's role.
   Direct navigation to a forbidden section shows an "Access denied" page
   with a button back to the dashboard. New file: `src/lib/rbac.ts`.
2. **Profile page no longer shows hardcoded "Dr. Sarah Chen".** The
   `ProfileScreen` now pulls real data from `/api/auth/me` and saves
   changes via `PATCH /api/auth/me`. New backend endpoint: PATCH on
   `src/app/api/auth/me/route.ts`.
3. **Hydration error fixed.** `SessionProvider` now starts with
   `loading: false, user: null` (matching the server-rendered HTML)
   and only flips `loading` to `true` inside a `useEffect`, eliminating
   the "button cannot contain a nested button" warning caused by the
   public header rendering different buttons on server vs. client.
4. **No more duplicate role selection.** The Register page now collects
   the user's role and sends it to `/api/auth/register`. After successful
   registration, the user is sent straight to `onboarding-workspace`,
   skipping the now-redundant `onboarding-role` step.
5. **Team Members page uses real data.** New endpoint `GET /api/team`
   returns the actual members of the user's organization. The
   `TeamMembersScreen` now fetches this list and renders it with real
   avatars, roles, statuses, and last-active timestamps — no more
   fake "Dr. Sarah Chen / James Wilson / Dr. Priya Patel" entries.

### Files added
- `src/lib/rbac.ts` — role-based access control: `canAccessSection(role, section)`, `visibleSectionsForRole(role)`, `roleLabel(role)`.
- `src/app/api/team/route.ts` — `GET /api/team` lists the current user's organization members.

### Files modified
- `prisma/schema.prisma` — added `title` and `bio` columns to the `User` model.
- `src/app/api/auth/register/route.ts` — accepts `role`, `title`, `bio` in the request body; validates the role against an allowlist.
- `src/app/api/auth/me/route.ts` — added `PATCH` handler to update `name`, `title`, `bio`. The `GET` handler now also returns `title` and `bio`.
- `src/components/drugos/session-provider.tsx` — fixed hydration mismatch by deferring the `loading: true` flip into a `useEffect`.
- `src/components/drugos/app-router.tsx`:
  - Imported `canAccessSection`, `visibleSectionsForRole`, `roleLabel` from `@/lib/rbac`.
  - `RegisterPage` now sends `role` to `/api/auth/register` and skips `onboarding-role` (jumps to `onboarding-workspace`).
  - `AppShell` sidebar filters items via `canAccessSection(user.role, item.id)`. Empty groups are hidden entirely.
  - `AppSectionRenderer` redirects to the dashboard if the user's role cannot access the requested section; otherwise shows an "Access denied" page.
  - `OnboardingWelcomePage` now shows a 2-step plan (workspace + invite) instead of 3-step (role + workspace + invite), and acknowledges the role the user picked during registration.
  - `OnboardingRolePage` is kept for backwards compatibility but shows the user's current role and notes that changing it requires admin approval.
- `src/components/drugos/all-screens.tsx`:
  - Added imports for `useSession`, `api`, `useEffect`, `roleLabel`, `canAccessSection`, `TeamMember`.
  - `ProfileScreen` is now backend-connected: pulls from `useSession()`, edits `name`/`title`/`bio`, calls `api.updateMe()`, shows success/error messages.
  - `TeamMembersScreen` is now backend-connected: fetches `api.listTeamMembers()`, renders real members with their actual roles and last-login timestamps.
- `src/lib/api-client.ts` — added `TeamMember` interface, `api.updateMe()`, `api.listTeamMembers()`. The `AuthUser` interface now includes optional `title` and `bio`. The `register` method now accepts an optional `role` parameter.
- `next.config.ts` — removed the deprecated `eslint.ignoreDuringBuilds` option (Next.js 16 no longer supports it in `next.config.ts`).

### Role → sections mapping

| Role | Can access |
|------|------------|
| `viewer` | Dashboard, search, evidence, profile, settings, help center |
| `researcher` | viewer + projects, shared-queries, annotations, changelog, roadmap, feedback |
| `data-scientist` | researcher + data-sources, graph-stats, quality |
| `pi` | data-scientist + team members |
| `business-dev` | researcher + deals |
| `developer` | api-docs, api-keys, playground, webhooks |
| `billing` | subscription, usage, invoices |
| `admin` | everything |
| `owner` | everything + investor-dashboard, cap-table |

---

## v0.3.0 — Initial frontend↔backend auth wiring

The frontend auth flow is now **fully wired to the backend**. Previously the
"Start Free" and "Sign In" buttons just navigated to the dashboard without
calling any API — now they hit `POST /api/auth/login` and
`POST /api/auth/register` instead of just navigating to the dashboard.

---

## TL;DR — what changed

| Area                | Before                                  | After                                                       |
|---------------------|-----------------------------------------|-------------------------------------------------------------|
| Login button        | `navigate({ page: 'app' })`             | `POST /api/auth/login` → set cookies → `refresh()` → navigate |
| Register button     | `navigate({ page: 'onboarding' })`      | `POST /api/auth/register` → set cookies → `refresh()` → navigate |
| `/app` access       | Open to anyone                          | Auth-guarded: redirects to `/login` if no valid session    |
| Header user menu    | Hardcoded "Dr. Sarah Chen"              | Pulls real user from `/api/auth/me`                         |
| Public header       | Always shows "Sign In / Start Free"     | Shows "Dashboard / Open Workspace" when already logged in  |
| Session state       | None — UI was stateless about auth      | New `SessionProvider` + `useSession()` hook                |
| API client          | None — UI never called the backend      | New `src/lib/api-client.ts` with typed wrappers for every endpoint |
| `package.json`      | `nextjs_tailwind_shadcn_ts` v0.2.0      | Renamed `drugos` v0.3.0; cleaned `dev`/`start`/`build` scripts |
| `next.config.ts`    | Hard-failed on ESLint warnings          | `eslint.ignoreDuringBuilds` for smoother local builds       |

---

## Files added

### `src/lib/api-client.ts`
A single typed frontend client for every backend endpoint. Wraps `fetch`
with `credentials: "include"` (HttpOnly cookies), JSON parsing, error
normalization, and a `drugos:unauthorized` window event the session
provider listens for. Exports typed interfaces: `AuthUser`, `Project`,
`Hypothesis`, `Plan`, `Subscription`, `Invoice`, `ApiKey`,
`Notification`, `AuditLog`, `AdminUser`, `SystemStatus`, etc.

### `src/components/drugos/session-provider.tsx`
React context that:
- Calls `GET /api/auth/me` on mount to hydrate the current user.
- Exposes `{ user, organizations, activeOrganizationId, loading, refresh, signOut }`.
- Listens for `drugos:unauthorized` events (fired by the API client on
  any 401) and clears the session so the auth guard can redirect.

### `scripts/install-loop.sh`
Helper script that retries `npm install` up to 10 times. Useful in
flaky-network environments. Run with `bash scripts/install-loop.sh`.

---

## Files modified

### `src/app/layout.tsx`
Wrapped `<children />` with `<SessionProvider>` so the session is
hydrated before any page renders.

### `src/components/drugos/app-router.tsx`
The big one. Five concrete fixes:

1. **Imports**: added `useEffect`, `useSession`, `api`, `ApiError`.
2. **`LoginPage`**: replaced the no-op `navigate({ page: 'app' })`
   handler with `handleSubmit` that calls `api.login(...)`, awaits
   `refresh()` from the session provider, and only navigates to the
   dashboard on success. Shows inline error messages for invalid
   credentials. Supports Enter-to-submit. Disables inputs while
   submitting.
3. **`RegisterPage`**: same pattern — `api.register(...)` then
   `refresh()` then navigate to `onboarding-welcome`. Validates first
   name + email + password client-side before submitting.
4. **`AppShell`**: added an auth guard. While the session is loading
   we show a splash screen; once it resolves and there's no user, we
   redirect to `/login`. The user menu in the header now reads from
   `useSession()` and shows the real name/email + active organization.
   The "Sign Out" menu item now calls `signOut()` and bounces to the
   landing page.
5. **`PublicHeader`**: when the user is already authenticated, the
   "Sign In / Start Free" buttons become "Dashboard / Open Workspace".
6. **`AppDashboard`**: subtitle reads from the session ("Welcome back,
   {user.name}") instead of a hardcoded "Dr. Sarah Chen".

### `package.json`
- Renamed package to `drugos`, bumped to `0.3.0`.
- `dev` no longer pipes through `tee` (cleaner stdout for terminals).
- `build` no longer copies `.next/static` into `.next/standalone` — that
  copy is only needed for the `output: "standalone"` Docker deploy
  pattern, and was breaking plain `next build` locally.
- `start` no longer invokes `bun .next/standalone/server.js` — it now
  uses `next start -p 3000` so you don't need bun installed.

### `next.config.ts`
- Added `eslint.ignoreDuringBuilds: true` so a stray lint warning in a
  mock-data file doesn't kill the production build.
- Kept `output: "standalone"` for Docker/Node production deploys.

### `.env`
The repo now ships with a `.env` pre-populated with a randomly
generated `JWT_SECRET` and a SQLite path at
`file:/home/z/my-project/db/custom.db`. **Change `JWT_SECRET` before
any real deployment** with `openssl rand -hex 32`.

---

## Backend endpoints (already existed, now actually used)

These routes were already implemented in the previous snapshot but the
frontend never called them. They are now exercised by the new auth
flow:

| Method | Path                           | Used by                          |
|--------|--------------------------------|----------------------------------|
| POST   | `/api/auth/register`           | RegisterPage                     |
| POST   | `/api/auth/login`              | LoginPage                        |
| POST   | `/api/auth/logout`             | AppShell user menu → Sign Out    |
| POST   | `/api/auth/refresh`            | SessionProvider (auto on 401)    |
| GET    | `/api/auth/me`                 | SessionProvider on mount         |
| GET    | `/api/system/status`           | Available for the Status page    |
| GET    | `/api/notifications`           | Available for the Notifications dropdown |
| GET    | `/api/projects`                | Available for the Projects screen |
| GET    | `/api/billing/subscription`    | Available for the Billing screen |
| GET    | `/api/api-keys`                | Available for the Developer/API Keys screen |
| GET    | `/api/admin/users`             | Available for the Admin Users screen |
| GET    | `/api/audit-logs`              | Available for the Admin Audit Logs screen |

The other endpoints (`/api/drugs/search`, `/api/diseases/search`,
`/api/clinical-trials/search`, `/api/literature/search`,
`/api/safety/[drug]`, `/api/patents/search`, `/api/evidence-package`)
remain wired through the `api` client and are ready to be called from
the relevant research screens — the existing mock-data screens can be
incrementally replaced one by one.

---

## Scientific integrity guarantees (preserved)

The three subsystems the user is building manually are intentionally
NOT implemented in this backend, and that contract is preserved:

1. **Knowledge Graph** (`/api/knowledge-graph`) — returns 503 with a
   refusal-to-fabricate message unless `KG_SERVICE_URL` is set.
2. **Dataset Pipeline** (`/api/dataset`) — same pattern, gated on
   `DATASET_SERVICE_URL`.
3. **RL Hypothesis Ranker** (`/api/rl`) — same pattern, gated on
   `RL_SERVICE_URL`.

When the user's standalone ML services are deployed, set those three
env vars and the corresponding endpoints will forward requests
transparently. The Prisma schema also intentionally does NOT model
graph nodes/edges, raw dataset tables, or RL model artifacts.

---

## How to run

```bash
# 1. Install deps (use whichever package manager you prefer)
bun install                # or: npm install / pnpm install / yarn install

# 2. Initialize the SQLite database
bun x prisma db push       # or: npx prisma db push
bun x prisma generate      # or: npx prisma generate

# 3. Start the dev server
bun run dev                # or: npm run dev
# → http://localhost:3000

# 4. Try the auth flow
# - Click "Start Free" on the landing page
# - Fill in the registration form (password needs 10+ chars with
#   upper + lower + digit + symbol)
# - You should land on the onboarding flow
# - Sign out, sign back in via the Login page
# - The dashboard should now show YOUR name in the header
```

If `npm install` hangs on slow networks, use the included retry script:

```bash
bash scripts/install-loop.sh
```

---

## Production deployment

1. Swap SQLite for PostgreSQL: change `DATABASE_URL` and the Prisma
   `datasource` provider to `postgresql`.
2. Set a strong `JWT_SECRET` (`openssl rand -hex 32`).
3. Configure `NCBI_API_KEY` and `PATENTSVIEW_API_KEY` for full biomedical
   data coverage.
4. Set `KG_SERVICE_URL`, `DATASET_SERVICE_URL`, `RL_SERVICE_URL` once
   the standalone ML services are deployed.
5. Replace the mock billing service in `src/lib/services/billing.ts`
   with Stripe webhooks.
6. Build and run:
   ```bash
   npm run build
   npm run start
   ```
