/**
 * DrugOS shared domain types.
 *
 * FE-026 ROOT FIX: Previously these types lived in `src/lib/mock-data.ts`
 * alongside 600+ lines of fabricated data (fake diseases, drugs, clinical
 * trials, audit logs, billing history, etc.). 11 components imported from
 * `mock-data.ts` — most of them used the types, but several also imported
 * the fabricated data constants, which meant the UI was rendering fake
 * candidates, fake trials, fake audit logs, fake billing history.
 *
 * This file is the canonical home for the domain types. Components MUST
 * import types from `@/lib/types` and MUST NOT import data from
 * `mock-data.ts`. Loading/empty/error states are rendered with skeletons
 * and "no data" messages — NEVER with fabricated data.
 *
 * If you need a type that doesn't exist here, add it. Do not re-introduce
 * a `mock-data.ts` file.
 */

export interface Disease {
  id: string;
  name: string;
  icdCode: string;
  meshTerm: string;
  therapeuticArea: string;
  prevalence: string;
  description: string;
  geneticBasis: boolean;
  /** Optional fields used by DiseaseSearchBar and dashboard. */
  synonyms?: string[];
  category?: string;
  candidateCount?: number;
  clinicalTrialCount?: number;
  icd10?: string;
}

// FE-023 ROOT FIX: 'unknown' tier added. RL model predictions do NOT get a
// green/yellow/red badge — those thresholds (>=0.7 green, >=0.4 yellow)
// were not clinically validated and gave a false sense of safety. Real
// safety tiering must come from openFDA label data (black-box warning →
// red, etc.) or FAERS adverse-event counts. Until that integration is in
// place, RL candidates show 'unknown' with a disclaimer.
export type SafetyTier = 'green' | 'yellow' | 'red' | 'unknown';

export interface DrugCandidate {
  id: string;
  drugName: string;
  brandNames: string[];
  genericName: string;
  /**
   * 0-100 weighted blend of KG/MolSim/Safety/Clinical sub-scores.
   * This is a MODEL OUTPUT, NOT a statistical confidence interval.
   * See FE-025: the UI column header must say "Composite Score", never
   * "Confidence", and a tooltip must explain the blend.
   */
  compositeScore: number;
  kgScore: number;
  molSimScore: number | null;
  safetyScore: number;
  clinicalScore: number;
  safetyTier: SafetyTier;
  /**
   * Drug mechanism of action (e.g. "NMDA receptor antagonist").
   * FE-024 ROOT FIX: This field MUST hold a real mechanism fetched from
   * ChEMBL/DrugBank — NEVER RL debug values like "RL reward: 0.234".
   * Empty string or "—" means the mechanism is unknown / not yet fetched.
   */
  mechanism: string;
  clinicalPhase: string;
  ipStatus: string | null;
  diseaseId: string;
  /** Optional disease name for candidates returned by the RL ranker. */
  diseaseName?: string;
  /** Optional rank assigned by the RL agent. */
  rank?: number;
  targets: string[] | null;
  pathways: string[] | null;
  /**
   * FE-024: RL debug info — model output that helps ML engineers debug
   * the ranker but is meaningless to a researcher. Surfaced ONLY in a
   * tooltip on the candidate row, NEVER in a table column.
   */
  rlDebugInfo?: {
    reward?: number;
    policyProb?: number;
    gnnScore?: number;
    rank?: number;
    source?: string;
  };
}

export interface ClinicalTrial {
  id: string;
  nctId: string;
  title: string;
  phase: string;
  status: string;
  enrollment: number;
  startDate: string;
  completionDate: string;
  drugName: string;
  disease: string;
  outcome: string;
}

export interface GraphNode {
  id: string;
  label: string;
  type: "drug" | "disease" | "gene" | "protein" | "pathway";
  x: number;
  y: number;
  size?: number;
  description?: string;
}

export interface GraphEdge {
  source: string;
  target: string;
  type: string;
  weight?: number;
  // Backward-compat alias used by knowledge-graph-viewer.
  relation?: string;
}

export interface User {
  id: string;
  name: string;
  email: string;
  role: string;
  status: string;
  lastLoginAt?: string;
  createdAt: string;
}

export interface AppNotification {
  id: string;
  type: "info" | "success" | "warning" | "error";
  title: string;
  body: string;
  readAt: string | null;
  createdAt: string;
  /**
   * Backward-compat aliases used by some components. `read` is derived
   * from `readAt` (read === readAt !== null). `message` is an alias for
   * `body`. These exist so components that imported the old mock-data
   * Notification type keep compiling during the FE-026 migration.
   */
  read?: boolean;
  message?: string;
}

export interface AuditLogEntry {
  id: string;
  userId: string | null;
  actorName: string;
  action: string;
  resource: string | null;
  ip: string | null;
  metadata: string;
  createdAt: string;
}

export interface Patent {
  id: string;
  patentNumber: string;
  title: string;
  assignee: string;
  filingDate: string;
  grantDate: string;
  abstract: string;
  drugName: string;
  // Backward-compat optional fields used by some components.
  status?: string;
  jurisdiction?: string;
  claims?: number;
  expirationDate?: string;
}

export interface EvidenceItem {
  id: string;
  type: string; // Backward-compat: was union, but components compare with various values like 'clinical', 'preclinical'.
  title: string;
  source: string;
  url: string;
  date: string;
  summary: string;
  // Backward-compat optional fields used by some components.
  drugName?: string;
  disease?: string;
  quality?: string;
  year?: number;
}

export interface ADMETProfile {
  drugName: string;
  absorption: number;
  distribution: number;
  metabolism: number;
  excretion: number;
  toxicity: number;
  bioavailability: number;
  bloodBrainBarrier: number;
}

export interface OffTargetPrediction {
  drugName: string;
  target: string;
  affinity: number;
  probability: number;
  adverseEventRisk: "low" | "medium" | "high";
  // Backward-compat optional fields used by some components.
  severity?: string;
  organSystem?: string;
}

export interface DrugInteraction {
  drugA: string;
  drugB: string;
  severity: string; // Backward-compat: was union, but components compare with various values.
  mechanism: string;
  clinicalEffect: string;
  // Backward-compat aliases used by some components.
  drug1?: string;
  drug2?: string;
  description?: string;
}

// SafetyTier is defined above (line ~37) to avoid a circular reference
// with DrugCandidate.safetyTier.

export interface DashboardStats {
  totalCandidates: number;
  totalDrugs: number;
  totalDiseases: number;
  knowledgeGraphNodes: number;
  knowledgeGraphEdges: number;
  literatureSupported: number;
  novelCandidates: number;
  avgConfidence: number;
}

export interface RecentActivityItem {
  id: string;
  type: string;
  actor: string;
  action: string;
  target: string;
  timestamp: string;
}

export interface Milestone {
  id: string;
  title: string;
  description: string;
  dueDate: string;
  status: "completed" | "in_progress" | "upcoming" | "blocked";
  progress: number;
}

export interface MonthlyQueryTrend {
  month: string;
  queries: number;
  candidates: number;
}

export interface SafetyTierDistribution {
  tier: SafetyTier;
  count: number;
  percentage: number;
}

export type KnowledgeGraphNode = GraphNode;
export type KnowledgeGraphEdge = GraphEdge;

export interface PathwayNode {
  id: string;
  label: string;
  type: "drug" | "protein" | "pathway" | "disease";
  // Backward-compat fields used by pathway-viz for positioning.
  x?: number;
  y?: number;
}

export interface PathwayEdge {
  source: string;
  target: string;
  label: string;
  // Backward-compat field used by pathway-viz.
  type?: string;
}

export interface PathwayData {
  nodes: PathwayNode[];
  edges: PathwayEdge[];
  // Backward-compat field used by pathway-viz.
  name?: string;
}
