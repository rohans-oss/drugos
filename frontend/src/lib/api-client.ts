/**
 * Frontend API client for DruGOS.
 *
 * Wraps `fetch` with credentials included, JSON parsing, error normalization,
 * and small typed helpers for every backend endpoint the UI needs.
 *
 * All cookies are HttpOnly so the browser sends them automatically — we never
 * touch tokens from JavaScript.
 */

import { z, type ZodType } from "zod";

export interface ApiError {
  error: string;
  message?: string;
  status: number;
}

export interface AuthUser {
  id: string;
  email: string;
  name: string | null;
  role: string;
  title?: string | null;
  bio?: string | null;
  status?: string;
  emailVerified?: boolean;
  academicVerified?: boolean;
  mfaEnabled?: boolean;
  lastLoginAt?: string | null;
  createdAt?: string;
}

export interface Organization {
  id: string;
  name: string;
  slug: string;
  plan: string;
  role: string;
}

export interface TeamMember {
  id: string;
  name: string;
  email: string;
  role: string;
  orgRole: string;
  title: string | null;
  bio: string | null;
  status: string;
  lastLoginAt: string | null;
  joinedAt: string;
}

export interface AuthMeResponse {
  user: AuthUser;
  organizations: Organization[];
  activeOrganizationId: string | null;
}

export interface Project {
  id: string;
  name: string;
  description: string | null;
  status: string;
  visibility: string;
  ownerId: string;
  organizationId: string;
  tags: string;
  createdAt: string;
  updatedAt: string;
  _count?: { hypotheses: number; comments: number };
}

export interface Hypothesis {
  id: string;
  projectId: string;
  title: string;
  drugName: string;
  diseaseName: string;
  status: string;
  plausibilityScore: number | null;
  safetyScore: number | null;
  marketScore: number | null;
  overallScore: number | null;
  notes: string | null;
  createdAt: string;
  updatedAt: string;
}

export interface Comment {
  id: string;
  projectId: string;
  userId: string | null;
  authorName: string;
  body: string;
  createdAt: string;
}

export interface ProjectActivity {
  id: string;
  projectId: string;
  type: string;
  actorName: string;
  summary: string;
  createdAt: string;
}

export interface ProjectDetail extends Project {
  hypotheses: Hypothesis[];
  comments: Comment[];
  activities: ProjectActivity[];
}

/**
 * FE-024 ROOT FIX: Aligned with the actual billing.ts Plan interface.
 * The backend Plan has `priceCents` (NOT `price`), NO `currency`, and
 * NO `interval` field. The previous type had `price: number` which was
 * always undefined because the route returns PLANS directly from
 * billing.ts — causing "$0" and "$NaN" renders in the subscription UI.
 */
export interface Plan {
  id: string;
  name: string;
  priceCents: number;
  seats: number;
  features: string[];
}

export interface Subscription {
  id: string;
  organizationId: string;
  plan: string;
  status: string;
  seats: number;
  currentPeriodStart: string;
  currentPeriodEnd: string;
  cancelAtPeriodEnd: boolean;
}

export interface Invoice {
  id: string;
  number: string;
  amountCents: number;
  currency: string;
  status: string;
  periodStart: string;
  periodEnd: string;
  dueDate: string;
  pdfUrl: string | null;
  createdAt: string;
}

export interface ApiKey {
  id: string;
  name: string;
  prefix: string;
  lastUsedAt: string | null;
  revokedAt: string | null;
  createdAt: string;
  // Only returned once on creation
  rawKey?: string;
}

export interface Notification {
  id: string;
  type: string;
  title: string;
  body: string;
  readAt: string | null;
  createdAt: string;
}

export interface AuditLog {
  id: string;
  userId: string | null;
  actorName: string;
  action: string;
  resource: string | null;
  ip: string | null;
  userAgent: string | null;
  metadata: string;
  createdAt: string;
}

export interface AdminUser {
  id: string;
  email: string;
  name: string | null;
  role: string;
  status: string;
  emailVerified: boolean;
  // FE-009 ROOT FIX: surfaced from /api/admin/users so the admin screen
  // can show the real 2FA state per user instead of a fabricated boolean.
  mfaEnabled?: boolean;
  createdAt: string;
  lastLoginAt: string | null;
}

export interface SystemStatus {
  services: Record<string, { available: boolean; service: string; reason?: string }>;
  generatedAt: string;
}

export interface DrugSearchResult {
  rxcui: string;
  name: string;
  synonym?: string;
  tty?: string;
}

/**
 * FE-023 ROOT FIX: Aligned with the actual MeshDescriptor returned by
 * the MeSH service (mesh.ts). The service returns `descriptorUi`
 * (lowercase 'i') and `name` — NOT `descriptorUI` (uppercase 'I') and
 * NOT `descriptorName`. The previous type caused disease search
 * suggestions to render with id=undefined and name=undefined.
 */
export interface DiseaseSearchResult {
  descriptorUi: string;
  name: string;
  scopeNote?: string;
  treeNumber?: string[];
}

export interface ClinicalTrial {
  nctId: string;
  title: string;
  status: string;
  phase?: string;
  conditions: string[];
  interventions: string[];
  sponsor?: string;
  startDate?: string;
  completionDate?: string;
  enrollment?: number;
  url?: string;
}

export interface PubMedArticle {
  pmid: string;
  title: string;
  authors: string[];
  journal?: string;
  pubDate?: string;
  abstract?: string;
  url?: string;
}

export interface SafetyReport {
  // BE-040 ROOT FIX (Team Member 12): the previous type had `drug: string`
  // — but the actual `DrugSafetySummary` returned by /api/safety/[drug]
  // (from frontend/src/lib/services/openfda.ts) has `brandName` and
  // `genericName`, NOT `drug`. Any consumer that read `safetyData.drug`
  // got `undefined`. We align the api-client type with the actual route
  // response shape so consumers can rely on the typed contract.
  brandName: string;
  genericName: string;
  totalReports: number;
  seriousReports: number;
  seriousReportsWithDeath?: number;
  topReactions: { term: string; count: number }[];
  disclaimer: string;
}

// ---------------------------------------------------------------------------
// ML / Phase 4 handoff types — ROOT FIX for FE-001/FE-002/FE-003
// ---------------------------------------------------------------------------

export interface RankedHypothesis {
  drug: string;
  disease: string;
  rank?: number;
  reward?: number;
  policyProb?: number;
  gnnScore?: number;
  safetyScore?: number;
  literatureSupport?: number;
}

export interface RlRankerResponse {
  candidates: RankedHypothesis[];
  source: "rl_service" | "local_csv" | "none";
  modelVersion?: string;
  generatedAt: string;
  count: number;
  note?: string;
  syncedHypotheses?: number;
}

export interface DatasetSourceStat {
  name: string;
  loaded: boolean;
  rowsLoaded?: number;
  sha256?: string;
}

export interface DatasetStatsResponse {
  sources: DatasetSourceStat[];
  nodesLoaded: number;
  edgesLoaded: number;
  edgeTypesPresent: string[];
  pipelineVersion?: string;
  schemaVersion?: string;
  bridgeVersion?: string;
  backend?: string;
  warnings: string[];
  errors: string[];
  source: "dataset_service" | "local_checkpoint" | "none";
  generatedAt: string;
  note?: string;
}

export interface GraphSourceStat {
  name: string;
  loaded: boolean;
  loadedReason?: string;
  version?: string;
  rows?: number;
  edgeCount?: number;
  sha256?: string;
  producedAt?: string;
  producedBy?: string;
  loadId?: string;
  /** Per-source breakdown of node types contributed (FE-020). */
  nodeTypeCounts?: Record<string, number>;
  /** Per-source breakdown of edge types contributed (FE-020). */
  edgeTypeCounts?: Record<string, number>;
}

export interface KnowledgeGraphStatsResponse {
  sources: GraphSourceStat[];
  /**
   * Sum of canonical node types ONLY (Compound + Protein + Pathway +
   * Disease + ClinicalOutcome) across all sources. Excludes
   * AdverseEvent and other non-canonical types.
   */
  nodeCount: number;
  /**
   * Sum of all edge_type_counts values across all sources. Edges are
   * not canonical/non-canonical — they all represent real graph
   * relationships.
   */
  edgeCount: number;
  /** Per-type breakdown of canonical node counts (FE-020). */
  nodeTypeCounts: Record<string, number>;
  /** Per-type breakdown of edge counts (FE-020). */
  edgeTypeCounts: Record<string, number>;
  /**
   * Per-type breakdown of NON-canonical node counts (e.g. AdverseEvent).
   * Surfaced for transparency — NOT included in `nodeCount`.
   */
  nonCanonicalNodeCounts: Record<string, number>;
  source: "kg_service" | "local_registry" | "none";
  generatedAt: string;
  note?: string;
}

export interface EvidencePackage {
  id: string;
  drugName: string;
  diseaseName: string;
  title: string;
  summary: string;
  status: string;
  createdAt: string;
  updatedAt: string;
}

// ---------------------------------------------------------------------------
// Core fetch wrapper
// ---------------------------------------------------------------------------

/**
 * FE-066 ROOT FIX: Runtime response validation.
 *
 * Previously: request<T> did `let body: any` and returned `body as T`. The
 * generic T was a LIE — if the API returned a different shape, the caller
 * got an object TypeScript thought was T but wasn't, and the bug only
 * surfaced at runtime (usually in a React render, far from the cause).
 *
 * Root fix: request<T> now accepts an optional `schema: ZodType<T>`. When
 * provided, the parsed body is run through `schema.parse(body)` before
 * return. Zod either confirms the shape or throws an ApiError with status
 * 0 and a descriptive message — the caller sees the contract violation
 * immediately, at the call site, instead of a cryptic render error.
 *
 * New code SHOULD pass a schema; the lint rule can be tightened later.
 */
function getCsrfTokenFromCookie(): string | null {
  if (typeof document === "undefined") return null;
  const match = document.cookie.match(/(?:^|; )\s*drugos_csrf=([^;]*)/);
  if (match && match[1]) return decodeURIComponent(match[1]);

  const newToken = Array.from({ length: 32 }, () => Math.floor(Math.random() * 16).toString(16)).join("");
  try {
    document.cookie = `drugos_csrf=${newToken}; path=/; sameSite=Lax`;
  } catch {}
  return newToken;
}

async function request<T>(
  url: string,
  init?: RequestInit & { skipAuthRedirect?: boolean; schema?: ZodType<T> }
): Promise<T> {
  const reqHeaders: Record<string, string> = {
    "Content-Type": "application/json",
    ...(init?.headers as Record<string, string> || {}),
  };

  const csrfToken = getCsrfTokenFromCookie();
  if (csrfToken && !reqHeaders["x-csrf-token"] && !reqHeaders["X-CSRF-Token"]) {
    reqHeaders["X-CSRF-Token"] = csrfToken;
  }

  const res = await fetch(url, {
    credentials: "include",
    ...init,
    headers: reqHeaders,
  });

  const text = await res.text();
  let body: any = null;
  if (text) {
    try {
      body = JSON.parse(text);
    } catch {
      body = { raw: text };
    }
  }

  if (!res.ok) {
    const err: ApiError = {
      error: body?.error || "request_failed",
      message: body?.message || `Request failed with status ${res.status}`,
      status: res.status,
    };
    // If 401 and not explicitly skipped, dispatch an event so the auth
    // provider can redirect to login.
    if (res.status === 401 && !init?.skipAuthRedirect) {
      // Guard for non-browser environments (jest, SSR).
      if (typeof window !== "undefined" && window.dispatchEvent) {
        window.dispatchEvent(new CustomEvent("drugos:unauthorized"));
      }
    }
    throw err;
  }

  // FE-066: if a zod schema was provided, validate the body. On failure
  // throw a structured ApiError so callers can distinguish contract
  // violations from network errors.
  if (init?.schema) {
    const parsed = init.schema.safeParse(body);
    if (!parsed.success) {
      const err: ApiError = {
        error: "response_shape_mismatch",
        message: `API response from ${url} did not match expected schema: ${parsed.error.message}`,
        status: 0, // 0 = client-side validation failure, not an HTTP status
      };
      throw err;
    }
    return parsed.data;
  }

  return body as T;
}

export const api = {
  // AUTH
  register: (body: {
    email: string;
    password: string;
    name: string;
    organizationName?: string;
    role?: string;
    title?: string;
    bio?: string;
  }) =>
    request<{ user: AuthUser; organizationId: string }>("/api/auth/register", {
      method: "POST",
      body: JSON.stringify(body),
      skipAuthRedirect: true,
    }),

  login: (body: { email: string; password: string }) =>
    request<{ user: AuthUser; organizationId: string }>("/api/auth/login", {
      method: "POST",
      body: JSON.stringify(body),
      skipAuthRedirect: true,
    }),

  logout: () => request<{ ok: true }>("/api/auth/logout", { method: "POST" }),

  refresh: () => request<{ ok: true }>("/api/auth/refresh", { method: "POST" }),

  me: () =>
    request<AuthMeResponse>("/api/auth/me", {
      skipAuthRedirect: true,
      // FE-066: validate the /me response shape. If the server contract
      // drifts (e.g. `user` becomes `profile`), this throws immediately
      // instead of producing undefined-user render errors downstream.
      schema: z.object({
        user: z.object({
          id: z.string(),
          email: z.string(),
          name: z.string().nullable(),
          role: z.string(),
          title: z.string().nullable().optional(),
          bio: z.string().nullable().optional(),
          status: z.string().optional(),
          emailVerified: z.boolean().optional(),
          academicVerified: z.boolean().optional(),
          mfaEnabled: z.boolean().optional(),
          lastLoginAt: z.string().nullable().optional(),
          createdAt: z.string().optional(),
        }),
        organizations: z.array(
          z.object({
            id: z.string(),
            name: z.string(),
            slug: z.string(),
            plan: z.string(),
            role: z.string(),
          })
        ),
        activeOrganizationId: z.string().nullable(),
      }) as ZodType<AuthMeResponse>,
    }),
  updateMe: (body: { name?: string; title?: string; bio?: string }) =>
    request<{ user: AuthUser }>("/api/auth/me", { method: "PATCH", body: JSON.stringify(body) }),

  // TEAM
  listTeamMembers: () => request<{ items: TeamMember[]; total: number }>("/api/team"),

  // PROJECTS
  listProjects: () => request<{ items: Project[] }>("/api/projects"),
  createProject: (body: { name: string; description?: string; visibility?: "private" | "org" | "public"; tags?: string[] }) =>
    request<Project>("/api/projects", { method: "POST", body: JSON.stringify(body) }),
  getProject: (id: string) => request<ProjectDetail>(`/api/projects/${id}`),
  addHypothesis: (projectId: string, body: { title: string; drugName: string; diseaseName: string; notes?: string }) =>
    request<Hypothesis>(`/api/projects/${projectId}`, { method: "POST", body: JSON.stringify(body) }),
  // FE-073 ROOT FIX: authorName is intentionally NOT accepted. The server
  // derives it from the authenticated user's User.name || User.email.
  // Sending authorName in the body is a no-op (server ignores it).
  addComment: (projectId: string, body: { body: string }) =>
    request<Comment>(`/api/projects/${projectId}/comments`, { method: "POST", body: JSON.stringify(body) }),

  // BILLING
  listPlans: () => request<{ plans: Plan[] }>("/api/billing/plans"),
  getSubscription: () => request<{ subscription: Subscription | null; plans: Plan[] }>("/api/billing/subscription"),
  // FE-021 ROOT FIX: The billing/subscription route requires currentPassword
  // (re-authentication) and optionally totpCode or mfaTicket when MFA is
  // enabled. The previous signature only sent { planId } — every plan change
  // got 400 "currentPassword is required". Updated signature accepts the
  // full required parameter object.
  changePlan: (body: { planId: string; currentPassword: string; totpCode?: string; mfaTicket?: string }) =>
    request<{ ok: true }>("/api/billing/subscription", { method: "POST", body: JSON.stringify(body) }),
  listInvoices: () => request<{ items: Invoice[] }>("/api/billing/invoices"),

  // API KEYS
  listApiKeys: () => request<{ items: ApiKey[] }>("/api/api-keys"),
  createApiKey: (name: string) =>
    request<ApiKey>("/api/api-keys", { method: "POST", body: JSON.stringify({ name }) }),
  revokeApiKey: (id: string) =>
    request<{ ok: true }>(`/api/api-keys/${id}/revoke`, { method: "POST" }),

  // NOTIFICATIONS
  listNotifications: () => request<{ items: Notification[] }>("/api/notifications"),
  markNotificationRead: (id: string) =>
    request<{ ok: true }>(`/api/notifications/${id}/read`, { method: "POST" }),

  // ADMIN
  listUsers: (limit = 50, offset = 0) =>
    request<{ items: AdminUser[]; total: number }>(`/api/admin/users?limit=${limit}&offset=${offset}`),
  updateUser: (body: { userId: string; role?: string; status?: string }) =>
    request<AdminUser>("/api/admin/users", { method: "PATCH", body: JSON.stringify(body) }),

  // AUDIT LOGS
  listAuditLogs: (limit = 100, offset = 0) =>
    request<{ items: AuditLog[]; total: number }>(`/api/audit-logs?limit=${limit}&offset=${offset}`),

  // SYSTEM STATUS
  getSystemStatus: () => request<SystemStatus>("/api/system/status"),

  // BIOMEDICAL DATA (live public APIs)
  searchDrugs: (q: string) =>
    request<{ items: DrugSearchResult[] }>(`/api/drugs/search?q=${encodeURIComponent(q)}`),
  searchDiseases: (q: string) =>
    request<{ items: DiseaseSearchResult[] }>(`/api/diseases/search?q=${encodeURIComponent(q)}`),
  // FE-022 ROOT FIX: The clinical-trials/search route requires `condition`
  // OR `intervention` — it does NOT accept a `q` param. The previous
  // signature always caused 400 errors. Updated to accept the correct
  // params object matching the route's expected query parameters.
  searchClinicalTrials: (params: { condition?: string; intervention?: string; limit?: number; pageToken?: string }) => {
    const qs = new URLSearchParams();
    if (params.condition) qs.set("condition", params.condition);
    if (params.intervention) qs.set("intervention", params.intervention);
    if (params.limit) qs.set("limit", String(params.limit));
    if (params.pageToken) qs.set("pageToken", params.pageToken);
    return request<{ items: ClinicalTrial[] }>(`/api/clinical-trials/search?${qs.toString()}`);
  },
  searchLiterature: (q: string) =>
    request<{ items: PubMedArticle[] }>(`/api/literature/search?q=${encodeURIComponent(q)}`),
  getSafety: (drug: string) =>
    request<SafetyReport>(`/api/safety/${encodeURIComponent(drug)}`),
  searchPatents: (q: string) =>
    request<{ items: any[] }>(`/api/patents/search?q=${encodeURIComponent(q)}`),

  // EVIDENCE PACKAGES
  listEvidencePackages: () => request<{ items: EvidencePackage[] }>("/api/evidence-package"),
  buildEvidencePackage: (body: { drug: string; disease: string; notes?: string; literatureLimit?: number; trialsLimit?: number }) =>
    request<{ id: string; package: any; markdown: string }>("/api/evidence-package", { method: "POST", body: JSON.stringify(body) }),
  getEvidencePackage: (id: string) =>
    request<{ id: string; package: any; markdown: string }>(`/api/evidence-package?id=${encodeURIComponent(id)}`),

  // ML — Phase 4 RL ranker, Phase 1 dataset, Phase 2 knowledge graph
  // ROOT FIX for FE-001/FE-002/FE-003: the UI now calls these real endpoints
  // instead of rendering mock data. The endpoints serve real data from the
  // Phase 1/2/4 Python pipeline artifacts.
  //
  // FE-003 ROOT FIX (Team Member 15, v108): re-added getDatasetStats().
  // The previous FE-025 "ROOT FIX" removed this method as "dead code" —
  // but that decision was itself a bug, because DataSourcesScreen NEEDS
  // this method to replace its hardcoded fake source list (FE-003).
  // Removing the method made it impossible to wire DataSourcesScreen to
  // real data without re-adding the method. The /api/dataset endpoint
  // exists and returns real source stats (loaded/rowsLoaded/sha256)
  // from the Phase 1 dataset-stats service. This method is now CALLED
  // by DataSourcesScreen — it is no longer dead code.
  getDatasetStats: () => request<DatasetStatsResponse>("/api/dataset"),

  // Issue 306 (audit 301-320): Graph Stats screen must call
  // /api/knowledge-graph/stats. The endpoint path /api/knowledge-graph
  // already returns the stats payload (route at
  // src/app/api/knowledge-graph/route.ts). For backward compat we keep
  // the original method name; new code should prefer getKnowledgeGraphStats.
  getKnowledgeGraphStats: () =>
    request<KnowledgeGraphStatsResponse>("/api/knowledge-graph"),

  // Issue 307 (audit 301-320): Quality screen calls /api/dataset/quality.
  // Returns REAL quality metrics derived from Phase 1 + Phase 2 stats
  // (completeness, integrity, freshness, canonical coverage). No
  // fabricated percentages.
  getDatasetQuality: () =>
    request<DatasetQualityResponse>("/api/dataset/quality"),

  // Issue 315 (audit 301-320): Investor Dashboard calls /api/admin/metrics.
  // Returns REAL platform metrics (user/org/project/hypothesis counts,
  // audit-log activity, dataset + KG scale). Financial metrics
  // (ARR/MRR/NRR) are explicitly null — NOT fabricated.
  getAdminMetrics: () => request<AdminMetricsResponse>("/api/admin/metrics"),
};

// ---------------------------------------------------------------------------
// Issue 318 (audit 301-320): Type contract for /api/dataset/quality and
// /api/admin/metrics. Aligned with the actual route response shapes —
// no descriptorUI/descriptorUi, price/priceCents, drug/brandName style
// mismatches. Each field name matches exactly what the route returns.
// ---------------------------------------------------------------------------

export interface DatasetQualityResponse {
  status: "ok" | "no_data" | "service_down";
  generatedAt: string;
  source: string;
  // Real coverage metrics — percentages computed from actual loaded/total
  sourceCompletenessPct: number;
  canonicalCoveragePct: number;
  checksumCoveragePct: number;
  // Real graph-anomaly signal
  nodeEdgeRatio: number;
  nodesLoaded: number;
  edgesLoaded: number;
  // Real per-canonical-type breakdown
  canonicalNodeCoverage: Array<{
    type: string;
    present: boolean;
    count: number;
  }>;
  // Real integrity signals
  sourcesWithChecksum: number;
  totalSources: number;
  // Real freshness signal
  freshnessHoursAgo: number | null;
  isStale: boolean;
  checkpointGeneratedAt: string | null;
  // Real issue counts
  warningsCount: number;
  errorsCount: number;
  warnings: string[];
  errors: string[];
  // Pipeline version metadata (for audit trail)
  pipelineVersion: string | null;
  schemaVersion: string | null;
  bridgeVersion: string | null;
  note?: string;
}

export interface AdminMetricsResponse {
  scope: "system" | "organization";
  organizationId: string | null;
  generatedAt: string;
  // REAL user/org/subscription counts
  totalUsers: number;
  totalOrganizations: number;
  activeSubscriptions: number;
  // REAL research activity counts
  totalProjects: number;
  totalHypotheses: number;
  totalValidatedHypotheses: number;
  totalEvidencePackages: number;
  // REAL platform activity (last 30 days)
  auditLogEventsLast30Days: number;
  topActionsLast30Days: Array<{ action: string; count: number }>;
  dailyActiveUsersLast7Days: Array<{ day: string; activeUsers: number }>;
  // REAL Phase 1 + Phase 2 data scale
  dataset: {
    nodesLoaded: number;
    edgesLoaded: number;
    sourcesLoaded: number;
    sourcesTotal: number;
    source: string;
    status: string;
  };
  knowledgeGraph: {
    nodeCount: number;
    edgeCount: number;
    source: string;
  } | null;
  // EXPLICITLY NOT FABRICATED — financial metrics are null, not invented
  financials: {
    arr: null;
    mrr: null;
    customerCount: null;
    nrr: null;
    note: string;
  };
}
