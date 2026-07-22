'use client';

/**
 * useKnowledgeGraph — React hook that fetches the knowledge graph for a
 * given drug (and optional disease) from the REAL `/api/knowledge-graph`
 * route, which proxies to the Phase 2 Neo4j service.
 *
 * Root fix for audit issue #290: the previous "use-knowledge-graph" hook
 * (and the in-screen code in `core-screens.tsx`) returned EMPTY data
 * because it:
 *   1. Never called `/api/knowledge-graph` at all — it just used the
 *      hardcoded `graphNodes` / `graphEdges` mock arrays from mock-data.
 *   2. Even if it had called the API, the response shape from the Neo4j
 *      service is different from the mock-data shape — the mock uses
 *      `{ x, y }` on every node, but the real API returns `{ id, label,
 *      type, properties }` with NO pre-computed positions.
 *
 * Root fix:
 *   - Always calls `/api/knowledge-graph?drug=<name>&disease=<name>&limit=<n>`.
 *   - Maps the API response into the same `GraphNode` / `GraphEdge`
 *     shape used by the rest of the UI, computing positions with a
 *     deterministic circular layout (so two renders of the same graph
 *     are visually stable).
 *   - When the API returns 503 (service_not_deployed), falls back to
 *     the mock data AND sets `isDemo: true` so the UI can show a
 *     "DEMO DATA" banner — never silently fabricating real data.
 *   - Never throws. Errors are captured into `error` and the hook
 *     returns `{ nodes: [], edges: [], isDemo: true }`.
 *
 * Acceptance criteria:
 *   - Returns real graph data when KG_SERVICE_URL is set.
 *   - Returns mock data with a DEMO flag when the service is down.
 *   - Returns an empty graph (not a crash) when the drug has no edges.
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import {
  graphNodes as MOCK_NODES,
  graphEdges as MOCK_EDGES,
  type GraphNode,
  type GraphEdge,
} from '@/lib/mock-data';

export interface KnowledgeGraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
  /** True when the data came from mock data, not the real service. */
  isDemo: boolean;
  /** True when the request is in flight. */
  loading: boolean;
  /** Error message if the request failed (and we couldn't fall back). */
  error: string | null;
  /** Source of the current data: 'api' | 'mock' | 'empty'. */
  source: 'api' | 'mock' | 'empty';
  /** Refresh callback. */
  refresh: () => void;
}

export interface UseKnowledgeGraphOptions {
  /** Drug name to center the graph on. */
  drug?: string;
  /** Optional disease name to filter the graph. */
  disease?: string;
  /** Max number of nodes to fetch. Default 500. */
  limit?: number;
  /** Disable the hook (skip the fetch). Default false. */
  enabled?: boolean;
  /** Auto-refresh interval in ms. 0 = no auto refresh. Default 0. */
  refreshIntervalMs?: number;
}

/** Canonical node types produced by the Phase 2 Neo4j service. */
type ApiNodeType = 'drug' | 'protein' | 'gene' | 'pathway' | 'disease' | 'outcome' | string;

interface ApiNode {
  id: string;
  label?: string;
  type?: ApiNodeType;
  properties?: Record<string, unknown>;
  x?: number;
  y?: number;
}

interface ApiEdge {
  source: string;
  target: string;
  relation?: string;
  label?: string;
  type?: string;
  evidence?: number;
  weight?: number;
  properties?: Record<string, unknown>;
}

interface ApiResponse {
  nodes?: ApiNode[];
  edges?: ApiEdge[];
  // Some proxies wrap the data under `graph`.
  graph?: { nodes?: ApiNode[]; edges?: ApiEdge[] };
  count?: number;
  error?: string;
}

/**
 * Deterministic circular layout — produces stable, predictable positions
 * for any node set without requiring a physics engine. For very large
 * graphs (>200 nodes) we switch to a multi-ring layout so labels don't
 * overlap.
 */
function computeLayout(nodes: ApiNode[], width = 900, height = 600): Map<string, { x: number; y: number }> {
  const out = new Map<string, { x: number; y: number }>();
  const n = nodes.length;
  if (n === 0) return out;
  const cx = width / 2;
  const cy = height / 2;

  if (n <= 200) {
    // Single ring.
    const r = Math.min(width, height) / 2 - 60;
    for (let i = 0; i < n; i++) {
      const angle = (2 * Math.PI * i) / n - Math.PI / 2;
      out.set(nodes[i].id, {
        x: cx + r * Math.cos(angle),
        y: cy + r * Math.sin(angle),
      });
    }
  } else {
    // Multi-ring (concentric) layout for large graphs.
    const perRing = 80;
    const ringCount = Math.ceil(n / perRing);
    const baseR = Math.min(width, height) / (ringCount * 2 + 1);
    let idx = 0;
    for (let ring = 0; ring < ringCount && idx < n; ring++) {
      const r = baseR * (ring + 1);
      const count = Math.min(perRing, n - idx);
      for (let i = 0; i < count; i++) {
        const angle = (2 * Math.PI * i) / count - Math.PI / 2 + (ring * 0.3);
        out.set(nodes[idx].id, {
          x: cx + r * Math.cos(angle),
          y: cy + r * Math.sin(angle),
        });
        idx++;
      }
    }
  }
  return out;
}

/** Map an arbitrary API node into our internal `GraphNode` shape. */
function mapApiNode(node: ApiNode, layout: Map<string, { x: number; y: number }>): GraphNode {
  const pos = layout.get(node.id) ?? { x: 0, y: 0 };
  const label =
    node.label ??
    (typeof node.properties?.['name'] === 'string' ? (node.properties['name'] as string) : node.id);
  const type = (node.type ?? 'gene') as GraphNode['type'];
  return {
    id: node.id,
    label,
    type,
    x: typeof node.x === 'number' ? node.x : pos.x,
    y: typeof node.y === 'number' ? node.y : pos.y,
  };
}

/** Map an arbitrary API edge into our internal `GraphEdge` shape. */
function mapApiEdge(edge: ApiEdge, i: number): GraphEdge {
  const evidence =
    typeof edge.evidence === 'number'
      ? edge.evidence
      : typeof edge.weight === 'number'
        ? edge.weight
        : 0.5;
  return {
    source: edge.source,
    target: edge.target,
    relation: edge.relation ?? edge.label ?? edge.type ?? 'related_to',
    evidence,
  };
}

export function useKnowledgeGraph(opts: UseKnowledgeGraphOptions = {}): KnowledgeGraphData {
  const { drug, disease, limit = 500, enabled = true, refreshIntervalMs = 0 } = opts;
  const [nodes, setNodes] = useState<GraphNode[]>([]);
  const [edges, setEdges] = useState<GraphEdge[]>([]);
  const [isDemo, setIsDemo] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [source, setSource] = useState<'api' | 'mock' | 'empty'>('empty');
  const [tick, setTick] = useState(0);
  const lastDrug = useRef<string | undefined>(undefined);

  const refresh = useCallback(() => setTick((t) => t + 1), []);

  useEffect(() => {
    if (!enabled) return;
    // Always fetch when drug/disease/limit/tick changes.
    const controller = new AbortController();
    // eslint-disable-next-line react-hooks/set-state-in-effect -- standard data-fetching pattern; setLoading must be called synchronously so the UI shows a loading indicator before the fetch resolves. Same pattern as the existing useRealCandidates hook in core-screens.tsx.
    setLoading(true);
    // eslint-disable-next-line react-hooks/set-state-in-effect -- see above
    setError(null);

    const params = new URLSearchParams();
    if (drug) params.set('drug', drug);
    if (disease) params.set('disease', disease);
    params.set('limit', String(limit));
    const url = `/api/knowledge-graph?${params.toString()}`;

    fetch(url, { signal: controller.signal, headers: { Accept: 'application/json' } })
      .then(async (res) => {
        if (res.status === 503) {
          // Service not deployed — fall back to mock data with a DEMO flag.
          // Filter mock data by drug/disease if provided.
          let mn = MOCK_NODES;
          let me = MOCK_EDGES;
          if (drug) {
            const drugLower = drug.toLowerCase();
            const matchingDrugNodeIds = new Set(
              MOCK_NODES.filter((n) => n.label.toLowerCase() === drugLower).map((n) => n.id),
            );
            if (matchingDrugNodeIds.size > 0) {
              // Keep nodes within 2 hops of the matched drug node.
              const hop1 = new Set<string>(matchingDrugNodeIds);
              MOCK_EDGES.forEach((e) => {
                if (matchingDrugNodeIds.has(e.source)) hop1.add(e.target);
                if (matchingDrugNodeIds.has(e.target)) hop1.add(e.source);
              });
              const hop2 = new Set<string>(hop1);
              MOCK_EDGES.forEach((e) => {
                if (hop1.has(e.source)) hop2.add(e.target);
                if (hop1.has(e.target)) hop2.add(e.source);
              });
              mn = MOCK_NODES.filter((n) => hop2.has(n.id));
              me = MOCK_EDGES.filter((e) => hop2.has(e.source) && hop2.has(e.target));
            }
          }
          setNodes(mn);
          setEdges(me);
          setIsDemo(true);
          setSource(mn.length === 0 ? 'empty' : 'mock');
          return;
        }
        if (!res.ok) {
          throw new Error(`knowledge-graph API returned ${res.status}`);
        }
        const data: ApiResponse = await res.json();
        const rawNodes = data.nodes ?? data.graph?.nodes ?? [];
        const rawEdges = data.edges ?? data.graph?.edges ?? [];
        if (rawNodes.length === 0) {
          setNodes([]);
          setEdges([]);
          setIsDemo(false);
          setSource('empty');
          return;
        }
        const layout = computeLayout(rawNodes);
        setNodes(rawNodes.map((n) => mapApiNode(n, layout)));
        setEdges(rawEdges.map((e, i) => mapApiEdge(e, i)));
        setIsDemo(false);
        setSource('api');
      })
      .catch((err) => {
        if (err?.name === 'AbortError') return;
        // Network/parse error — fall back to mock so the UI is usable.
        setNodes(MOCK_NODES);
        setEdges(MOCK_EDGES);
        setIsDemo(true);
        setSource('mock');
        setError(String(err?.message ?? err));
      })
      .finally(() => {
        setLoading(false);
      });

    return () => controller.abort();
  }, [drug, disease, limit, enabled, tick]);

  // Optional auto-refresh.
  useEffect(() => {
    if (!refreshIntervalMs || refreshIntervalMs < 5000) return;
    const id = setInterval(() => setTick((t) => t + 1), refreshIntervalMs);
    return () => clearInterval(id);
  }, [refreshIntervalMs]);

  return { nodes, edges, isDemo, loading, error, source, refresh };
}

export default useKnowledgeGraph;
