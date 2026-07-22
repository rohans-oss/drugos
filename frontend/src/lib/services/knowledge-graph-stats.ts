/**
 * DEPRECATED — use `@/lib/services/kg-service` instead.
 *
 * This file is kept ONLY as a backward-compat re-export shim so existing
 * imports (`from "@/lib/services/knowledge-graph-stats"`) continue to
 * work after Issue 232 consolidated the KG service into `kg-service.ts`.
 *
 * Issue 232 ROOT FIX: the previous implementation had a local-registry
 * fallback that read `../phase2/data/registry.json` directly, bypassing
 * the Python Phase 2 service. This caused the dashboard to display
 * stale registry data. The new `kg-service.ts` is HTTP-only — no local
 * file reads.
 *
 * All exports below are re-exported from `kg-service.ts`. Do not add new
 * code to this file — add it to `kg-service.ts` instead.
 */

export {
  getKnowledgeGraphStats,
  exploreKnowledgeGraph,
  queryKnowledgeGraph,
  executeCypher,
  validateEntityInKg,
  checkKgHealth,
  CANONICAL_NODE_TYPES,
  CANONICAL_NODE_TYPE_SET,
} from "./kg-service";
export type {
  KgStatsResponse,
  KgQueryResponse,
  KgCypherResponse,
  GraphSourceStat,
  KnowledgeGraphStatsResponse,
  CanonicalNodeType,
} from "./kg-service";
