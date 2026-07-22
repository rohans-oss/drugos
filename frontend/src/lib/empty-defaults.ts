/**
 * Empty default collections for screens that have not yet been migrated to
 * real API hooks.
 *
 * WHY THIS FILE EXISTS (FE-034 root fix):
 *   The previous `mock-data.ts` file was 339 lines of empty arrays plus type
 *   re-exports. Its NAME was the problem: a future engineer seeing
 *   `mock-data.ts` would reasonably assume "this is where I add sample data
 *   for development" — and re-introduce fabricated data into production.
 *   That exact mistake already happened once (the file was originally 621
 *   lines of fake diseases, drugs, trials, users, audit logs, billing
 *   history, etc.). The audit caught it, the data was emptied, but the
 *   dangerous name remained.
 *
 *   This file is the production-safe replacement. The name `empty-defaults`
 *   makes the intent unambiguous: every export here is an EMPTY default,
 *   and NO ONE should ever push real or sample data into this file. If you
 *   need real data, call the API (see `use-api-data.tsx` and
 *   `use-account-data.tsx` for hooks). If you need a type, import from
 *   `@/lib/types`.
 *
 * CONTRACT:
 *   - Every export here MUST remain empty (array `[]` or object with zeroed
 *     fields). A CI test (fe-029-to-036-team16.test.ts) asserts this.
 *   - No fabricated data. No "sample" data. No "demo" data. Empty only.
 *   - Types are NOT re-exported here — import types from `@/lib/types`.
 */

import type {
  Disease,
  DrugCandidate,
  ClinicalTrial,
  GraphNode,
  GraphEdge,
  User,
  AppNotification,
  AuditLogEntry,
  Patent,
  EvidenceItem,
  ADMETProfile,
  OffTargetPrediction,
  DrugInteraction,
  DashboardStats,
  RecentActivityItem,
  Milestone,
  MonthlyQueryTrend,
  SafetyTierDistribution,
  PathwayData,
} from "@/lib/types";

// ---------------------------------------------------------------------------
// Empty data exports. Every entry is `[]` or a zeroed object. Do NOT add
// real data here — see file header.
// ---------------------------------------------------------------------------

export const diseases: Disease[] = [];
export const drugCandidates: DrugCandidate[] = [];
export const clinicalTrials: ClinicalTrial[] = [];
export const graphNodes: GraphNode[] = [];
export const graphEdges: GraphEdge[] = [];
export const knowledgeGraphNodes: GraphNode[] = graphNodes;
export const knowledgeGraphEdges: GraphEdge[] = graphEdges;
export const users: User[] = [];
export const teamMembers: User[] = [];
export const notifications: AppNotification[] = [];
export const auditLogs: AuditLogEntry[] = [];
export const patents: Patent[] = [];
export const evidenceItems: EvidenceItem[] = [];
export const admetProfiles: ADMETProfile[] = [];
export const offTargetPredictions: OffTargetPrediction[] = [];
export const drugInteractions: DrugInteraction[] = [];
export const pathwayData: PathwayData = { nodes: [], edges: [] };

// BE-075 ROOT FIX (v115, LOW): the previous type used `price: string`
// but billing.ts `Plan` uses `priceCents: number`. The type mismatch
// meant any code that consumed subscription plans from empty-defaults
// would crash at runtime if it tried to do arithmetic on the price.
// The fix aligns the type with billing.ts — `priceCents: number` and
// `currency: string`. The array is still empty (per the file's
// contract: NO real data lives here), but the TYPE is now correct.
export const subscriptionPlans: Array<{
  id: string;
  name: string;
  priceCents: number;
  currency: string;
  interval: string;
  features: string[];
}> = [];

export const billingHistory: Array<{
  id: string;
  number: string;
  amount: number;
  date: string;
  status: string;
  plan: string;
}> = [];

export const apiKeys: Array<{
  id: string;
  name: string;
  prefix: string;
  createdAt: string;
  lastUsedAt: string | null;
  revokedAt: string | null;
}> = [];

export const webhooks: Array<{
  id: string;
  url: string;
  events: string;
  secret: string;
  enabled: boolean;
  lastTriggeredAt: string | null;
  createdAt: string;
}> = [];

export const usageMetrics: {
  apiCalls: { used: number; limit: number };
  storage: { used: number; limit: number };
  projects: { used: number; limit: number };
  hypotheses: { used: number; limit: number };
  queries: { used: number; limit: number };
  reports: { used: number; limit: number };
} = {
  apiCalls: { used: 0, limit: 0 },
  storage: { used: 0, limit: 0 },
  projects: { used: 0, limit: 0 },
  hypotheses: { used: 0, limit: 0 },
  queries: { used: 0, limit: 0 },
  reports: { used: 0, limit: 0 },
};

export const dataSources: Array<{
  id: string;
  name: string;
  version: string;
  lastUpdated: string;
  records: number;
  status: string;
}> = [];

export const trendingDiseases: Array<{
  id: string;
  name: string;
  queries: number;
  candidates: number;
  trend: string;
  change?: string;
}> = [];

export const recentQueries: Array<{
  id: string;
  disease: string;
  date: string;
  candidates: number;
  topScore: number;
}> = [];

export const projects: Array<{
  id: string;
  name: string;
  description: string;
  status: string;
  hypotheses: number;
  collaborators: number;
  updatedAt: string;
}> = [];

export const dealPipeline: Array<{
  id: string;
  partner: string;
  drug: string;
  disease: string;
  stage: string;
  value: number;
  probability: number;
  expectedClose: string;
}> = [];

export const organization: {
  id: string;
  name: string;
  plan: string;
  seats: number;
  createdAt: string;
} = {
  id: "",
  name: "",
  plan: "",
  seats: 0,
  createdAt: "",
};

export const featureFlags: Array<{
  id: string;
  name: string;
  description: string;
  enabled: boolean;
  rolloutPercentage: number;
}> = [];

export const systemStatus: Array<{
  id: string;
  service: string;
  status: string;
  latency: number;
  uptime: number;
}> = [];

export const savedQueries: Array<{
  id: string;
  name: string;
  query: string;
  filters: string;
  createdAt: string;
  disease?: string;
  results?: number;
  created?: string;
}> = [];

export const blogPosts: Array<{
  id: string;
  title: string;
  excerpt: string;
  category: string;
  date: string;
  author: string;
  readTime: string;
}> = [];

export const careers: Array<{
  id: string;
  title: string;
  location: string;
  type: string;
  department: string;
  postedAt: string;
}> = [];

export const dashboardStats: DashboardStats = {
  totalCandidates: 0,
  totalDrugs: 0,
  totalDiseases: 0,
  knowledgeGraphNodes: 0,
  knowledgeGraphEdges: 0,
  literatureSupported: 0,
  novelCandidates: 0,
  avgConfidence: 0,
};

export const recentActivity: RecentActivityItem[] = [];
export const milestones: Milestone[] = [];
export const monthlyQueryTrend: MonthlyQueryTrend[] = [];
export const safetyTierDistribution: SafetyTierDistribution[] = [];
