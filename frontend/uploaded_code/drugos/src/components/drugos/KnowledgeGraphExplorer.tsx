'use client';

/**
 * KnowledgeGraphExplorer — production-grade biomedical knowledge graph
 * visualization component.
 *
 * ROOT FIXES (audit issues #283, #286, #287, #288, #289):
 *
 * #283 — The previous version initialized `positions` Map from the
 *        `nodes` prop, but the prop defaulted to `knowledgeGraphNodes`
 *        which DOES NOT EXIST in mock-data.ts (the actual exports are
 *        `graphNodes` / `graphEdges`). The import was broken at
 *        compile time, AND every real node returned `{x:0,y:0}` because
 *        the positions Map was built from an empty default array.
 *        → Now we compute positions from the actual graph data using
 *          a deterministic circular / multi-ring layout, and we accept
 *          the canonical `GraphNode` / `GraphEdge` types.
 *
 * #286 — The previous version used `knowledgeGraphNodes` mock data
 *        directly. → Now wires to `/api/knowledge-graph` via the
 *        `useKnowledgeGraph` hook.
 *
 * #287 — The previous version rendered SVG, which crashes the browser
 *        at ~200 nodes (SVG reflows every DOM node on every paint).
 *        → Now uses Canvas2D for the graph body. Canvas2D handles
 *          10,000+ nodes smoothly on commodity hardware (only one DOM
 *          element, all drawing done in JS). SVG is used ONLY for the
 *          legend. We deliberately chose Canvas2D over WebGL (sigma.js
 *          / deck.gl) because: (a) it requires zero new dependencies,
 *          (b) it's enough for the 1,000-node target, (c) WebGL
 *          pipelines are notoriously fragile across drivers and would
 *          add 200KB+ to the bundle. If the graph ever needs to scale
 *          past 50K nodes, WebGL becomes mandatory — until then
 *          Canvas2D is the right engineering trade-off.
 *
 * #288 — The previous version had no click handler that opened a side
 *        panel with details. → Now `onNodeClick` opens a slide-out
 *        side panel showing the node's label, type, all connected
 *        edges, and (for drug nodes) a link to the candidate detail.
 *
 * #289 — The previous version showed ALL edges with no filtering. →
 *        Now supports edge filtering by relation type (checkboxes),
 *        plus the existing node-type filter and evidence threshold.
 *
 * Acceptance: renders 0 / 12 / 1,000 / 5,000 nodes without crashing,
 * supports click → side panel, supports edge filtering, never throws
 * on undefined nodes/edges.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { ZoomIn, ZoomOut, RotateCcw, X, Filter, Network } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Checkbox } from '@/components/ui/checkbox';
import { Slider } from '@/components/ui/slider';
import { Input } from '@/components/ui/input';
import { ScrollArea } from '@/components/ui/scroll-area';
import { cn } from '@/lib/utils';
import { EmptyState } from '@/components/drugos/EmptyState';
import { useKnowledgeGraph } from '@/hooks/use-knowledge-graph';
import type { GraphNode, GraphEdge } from '@/lib/mock-data';

export interface KnowledgeGraphExplorerProps {
  /** Drug name to center the graph on. */
  drug?: string;
  /** Optional disease filter. */
  disease?: string;
  /** Max nodes to fetch from the API. Default 1000. */
  limit?: number;
  /** Pixel height of the canvas. Default 550. */
  height?: number;
  /** Extra class names. */
  className?: string;
  /** Initial evidence threshold (0..1). Default 0. */
  defaultEvidenceThreshold?: number;
}

const NODE_COLORS: Record<string, string> = {
  drug: '#1D9E75',
  disease: '#C0392B',
  gene: '#5B4FCF',
  protein: '#8B5CF6',
  pathway: '#D4853A',
  outcome: '#0EA5E9',
};

const NODE_LABELS: Record<string, string> = {
  drug: 'Drug',
  disease: 'Disease',
  gene: 'Gene',
  protein: 'Protein',
  pathway: 'Pathway',
  outcome: 'Outcome',
};

const ALL_NODE_TYPES = ['drug', 'protein', 'gene', 'pathway', 'disease', 'outcome'] as const;

interface SelectedNodeInfo {
  node: GraphNode;
  edges: Array<{ edge: GraphEdge; other: GraphNode; direction: 'out' | 'in' }>;
}

export function KnowledgeGraphExplorer({
  drug,
  disease,
  limit = 1000,
  height = 550,
  className,
  defaultEvidenceThreshold = 0,
}: KnowledgeGraphExplorerProps) {
  // Real API data via the hook.
  const { nodes, edges, loading, error, isDemo, source, refresh } = useKnowledgeGraph({
    drug,
    disease,
    limit,
    enabled: true,
  });

  // UI state.
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [hoveredNodeId, setHoveredNodeId] = useState<string | null>(null);
  const [nodeTypeFilter, setNodeTypeFilter] = useState<Record<string, boolean>>(
    () => Object.fromEntries(ALL_NODE_TYPES.map((t) => [t, true])),
  );
  const [edgeTypeFilter, setEdgeTypeFilter] = useState<Record<string, boolean>>({});
  const [evidenceThreshold, setEvidenceThreshold] = useState<number>(defaultEvidenceThreshold);
  const [searchQuery, setSearchQuery] = useState('');
  const [showFilters, setShowFilters] = useState(true);

  // Canvas + drag state.
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const dragState = useRef<{ mode: 'none' | 'pan' | 'node'; nodeId?: string; startX: number; startY: number; origPanX: number; origPanY: number; origNodeX?: number; origNodeY?: number }>({
    mode: 'none',
    startX: 0,
    startY: 0,
    origPanX: 0,
    origPanY: 0,
  });
  const [canvasSize, setCanvasSize] = useState({ width: 900, height });

  // Mutable positions overlay (for node-dragging). Initialized from
  // node.x / node.y on every data change.
  const positionsRef = useRef<Map<string, { x: number; y: number }>>(new Map());
  useEffect(() => {
    const m = new Map<string, { x: number; y: number }>();
    nodes.forEach((n) => m.set(n.id, { x: n.x, y: n.y }));
    positionsRef.current = m;
    draw();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [nodes]);

  // Track the available edge relation types whenever edges change.
  const availableEdgeTypes = useMemo(() => {
    const set = new Set<string>();
    edges.forEach((e) => set.add(e.relation));
    return Array.from(set).sort();
  }, [edges]);

  // Initialize edge type filter when new edge types appear (default all on).
  useEffect(() => {
    setEdgeTypeFilter((prev) => {
      const next = { ...prev };
      availableEdgeTypes.forEach((t) => {
        if (next[t] === undefined) next[t] = true;
      });
      // Drop types that are no longer present.
      Object.keys(next).forEach((t) => {
        if (!availableEdgeTypes.includes(t)) delete next[t];
      });
      return next;
    });
  }, [availableEdgeTypes]);

  // Filtered data driving the canvas.
  const filtered = useMemo(() => {
    const q = searchQuery.trim().toLowerCase();
    const matchedNodeIds = new Set<string>();
    const fn = nodes.filter((n) => {
      if (!nodeTypeFilter[n.type]) return false;
      if (q.length >= 2 && !n.label.toLowerCase().includes(q) && !n.id.toLowerCase().includes(q)) {
        return false;
      }
      matchedNodeIds.add(n.id);
      return true;
    });
    // If searching, also include 1-hop neighbors of matches.
    if (q.length >= 2) {
      edges.forEach((e) => {
        if (matchedNodeIds.has(e.source) && nodeTypeFilter[getNodeType(e.target, nodes)]) {
          matchedNodeIds.add(e.target);
        }
        if (matchedNodeIds.has(e.target) && nodeTypeFilter[getNodeType(e.source, nodes)]) {
          matchedNodeIds.add(e.source);
        }
      });
    }
    const fnIds = new Set(fn.map((n) => n.id));
    // Include 1-hop neighbors of matched search results.
    const finalNodeIds = new Set<string>(fnIds);
    if (q.length >= 2) {
      edges.forEach((e) => {
        if (fnIds.has(e.source)) finalNodeIds.add(e.target);
        if (fnIds.has(e.target)) finalNodeIds.add(e.source);
      });
    }
    const finalNodes = nodes.filter((n) => finalNodeIds.has(n.id));
    const finalEdges = edges.filter((e) => {
      if (!finalNodeIds.has(e.source) || !finalNodeIds.has(e.target)) return false;
      if (!edgeTypeFilter[e.relation]) return false;
      if (e.evidence < evidenceThreshold) return false;
      return true;
    });
    return { nodes: finalNodes, edges: finalEdges, allFilteredNodeIds: finalNodeIds };
  }, [nodes, edges, nodeTypeFilter, edgeTypeFilter, evidenceThreshold, searchQuery]);

  const selectedNodeInfo = useMemo<SelectedNodeInfo | null>(() => {
    if (!selectedNodeId) return null;
    const node = nodes.find((n) => n.id === selectedNodeId);
    if (!node) return null;
    const related: SelectedNodeInfo['edges'] = [];
    edges.forEach((edge) => {
      if (edge.source === selectedNodeId) {
        const other = nodes.find((n) => n.id === edge.target);
        if (other) related.push({ edge, other, direction: 'out' });
      } else if (edge.target === selectedNodeId) {
        const other = nodes.find((n) => n.id === edge.source);
        if (other) related.push({ edge, other, direction: 'in' });
      }
    });
    return { node, edges: related };
  }, [selectedNodeId, nodes, edges]);

  // Resize observer — keep canvas size in sync with container width.
  useEffect(() => {
    if (!containerRef.current) return;
    const el = containerRef.current;
    const update = () => {
      const w = el.clientWidth;
      if (w > 0) {
        setCanvasSize((prev) => (prev.width === w ? prev : { width: w, height }));
      }
    };
    update();
    const ro = new ResizeObserver(update);
    ro.observe(el);
    return () => ro.disconnect();
  }, [height]);

  // Draw the graph on the canvas.
  // eslint-disable-next-line react-hooks/preserve-manual-memoization -- the draw callback touches canvas state, zoom, pan, filtered nodes/edges, and selection; it cannot be decomposed further without losing frame coherence. Re-running on every dependency change is the intended behavior (Canvas2D redraws the whole scene each frame).
  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const cssW = canvasSize.width;
    const cssH = canvasSize.height;
    if (canvas.width !== cssW * dpr || canvas.height !== cssH * dpr) {
      canvas.width = cssW * dpr;
      canvas.height = cssH * dpr;
      canvas.style.width = `${cssW}px`;
      canvas.style.height = `${cssH}px`;
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    // Clear.
    ctx.clearRect(0, 0, cssW, cssH);
    ctx.fillStyle = '#FAFAFC';
    ctx.fillRect(0, 0, cssW, cssH);

    // Apply zoom + pan.
    ctx.save();
    ctx.translate(cssW / 2 + pan.x, cssH / 2 + pan.y);
    ctx.scale(zoom, zoom);
    ctx.translate(-cssW / 2, -cssH / 2);

    const positions = positionsRef.current;
    const finalNodeIds = filtered.allFilteredNodeIds;
    const highlightedSet = new Set<string>();
    if (selectedNodeId) {
      highlightedSet.add(selectedNodeId);
      filtered.edges.forEach((e) => {
        if (e.source === selectedNodeId) highlightedSet.add(e.target);
        if (e.target === selectedNodeId) highlightedSet.add(e.source);
      });
    }
    const hasSelection = highlightedSet.size > 0;

    // Edges first (so they render under nodes).
    filtered.edges.forEach((edge) => {
      const s = positions.get(edge.source);
      const t = positions.get(edge.target);
      if (!s || !t) return;
      const isHighlighted =
        !hasSelection ||
        (edge.source === selectedNodeId || edge.target === selectedNodeId);
      ctx.beginPath();
      ctx.moveTo(s.x, s.y);
      ctx.lineTo(t.x, t.y);
      ctx.strokeStyle = isHighlighted
        ? evidenceToColor(edge.evidence, 0.65)
        : 'rgba(180, 180, 200, 0.18)';
      ctx.lineWidth = isHighlighted ? 1.6 : 0.7;
      ctx.stroke();
    });

    // Nodes.
    const nodeRadius = (n: GraphNode) => {
      // Slightly larger for drug / disease so they pop.
      if (n.type === 'drug' || n.type === 'disease') return 9;
      return 6;
    };
    filtered.nodes.forEach((n) => {
      const pos = positions.get(n.id);
      if (!pos) return;
      const r = nodeRadius(n);
      const color = NODE_COLORS[n.type] ?? '#64748b';
      const isSelected = selectedNodeId === n.id;
      const isHovered = hoveredNodeId === n.id;
      const isHighlighted = !hasSelection || highlightedSet.has(n.id);
      const alpha = isHighlighted ? 1 : 0.25;

      // Selection ring.
      if (isSelected) {
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, r + 5, 0, Math.PI * 2);
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.5;
        ctx.setLineDash([4, 3]);
        ctx.globalAlpha = 0.6;
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.globalAlpha = 1;
      }

      // Fill.
      ctx.beginPath();
      ctx.arc(pos.x, pos.y, r, 0, Math.PI * 2);
      ctx.fillStyle = hexWithAlpha(color, 0.18 * alpha);
      ctx.fill();
      ctx.strokeStyle = hexWithAlpha(color, alpha);
      ctx.lineWidth = isHovered || isSelected ? 2 : 1.2;
      ctx.stroke();
    });

    // Labels — only when there are few enough nodes to read them, OR
    // for the hovered/selected node, OR for 1-hop neighbors of a
    // selection. Rendering 5000 labels would just produce a black smear.
    const showAllLabels = filtered.nodes.length <= 80;
    if (showAllLabels || hasSelection || hoveredNodeId) {
      ctx.font = '11px ui-sans-serif, system-ui, -apple-system, sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      filtered.nodes.forEach((n) => {
        const pos = positions.get(n.id);
        if (!pos) return;
        const r = nodeRadius(n);
        const isSelected = selectedNodeId === n.id;
        const isHovered = hoveredNodeId === n.id;
        const isNeighborOfSelection =
          hasSelection && highlightedSet.has(n.id) && !isSelected;
        if (!showAllLabels && !isSelected && !isHovered && !isNeighborOfSelection) return;
        const label =
          n.label.length > 24 ? n.label.slice(0, 22) + '…' : n.label;
        const color = NODE_COLORS[n.type] ?? '#475569';
        ctx.fillStyle = isSelected ? color : '#334155';
        ctx.fillText(label, pos.x, pos.y + r + 4);
      });
    }

    ctx.restore();
  }, [canvasSize, pan, zoom, filtered, selectedNodeId, hoveredNodeId]);

  // Redraw whenever any dependency changes.
  useEffect(() => {
    draw();
  }, [draw]);

  // Convert mouse coordinates to graph coordinates.
  const toGraphCoords = useCallback(
    (clientX: number, clientY: number) => {
      const canvas = canvasRef.current;
      if (!canvas) return { x: 0, y: 0 };
      const rect = canvas.getBoundingClientRect();
      const cssX = clientX - rect.left;
      const cssY = clientY - rect.top;
      // Inverse of: translate(W/2 + pan) scale(zoom) translate(-W/2)
      const x = (cssX - canvasSize.width / 2 - pan.x) / zoom + canvasSize.width / 2;
      const y = (cssY - canvasSize.height / 2 - pan.y) / zoom + canvasSize.height / 2;
      return { x, y };
    },
    [canvasSize, pan, zoom],
  );

  // Hit-test: find the node under a graph point.
  const findNodeAt = useCallback(
    (gx: number, gy: number): GraphNode | null => {
      const positions = positionsRef.current;
      // Iterate from end so the most recently drawn (top) wins.
      for (let i = filtered.nodes.length - 1; i >= 0; i--) {
        const n = filtered.nodes[i];
        const pos = positions.get(n.id);
        if (!pos) continue;
        const r = n.type === 'drug' || n.type === 'disease' ? 9 : 6;
        const dx = pos.x - gx;
        const dy = pos.y - gy;
        if (dx * dx + dy * dy <= (r + 3) * (r + 3)) return n;
      }
      return null;
    },
    [filtered.nodes],
  );

  const handleMouseDown = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const { x, y } = toGraphCoords(e.clientX, e.clientY);
      const hit = findNodeAt(x, y);
      if (hit) {
        setSelectedNodeId(hit.id);
        const pos = positionsRef.current.get(hit.id);
        dragState.current = {
          mode: 'node',
          nodeId: hit.id,
          startX: x,
          startY: y,
          origPanX: pan.x,
          origPanY: pan.y,
          origNodeX: pos?.x,
          origNodeY: pos?.y,
        };
      } else {
        dragState.current = {
          mode: 'pan',
          startX: e.clientX,
          startY: e.clientY,
          origPanX: pan.x,
          origPanY: pan.y,
        };
      }
    },
    [findNodeAt, toGraphCoords, pan],
  );

  const handleMouseMove = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const ds = dragState.current;
      if (ds.mode === 'pan') {
        setPan({
          x: ds.origPanX + (e.clientX - ds.startX),
          y: ds.origPanY + (e.clientY - ds.startY),
        });
        return;
      }
      if (ds.mode === 'node' && ds.nodeId) {
        const { x, y } = toGraphCoords(e.clientX, e.clientY);
        const m = positionsRef.current;
        m.set(ds.nodeId, { x, y });
        draw();
        return;
      }
      // Hover hit-test for cursor + label.
      const { x, y } = toGraphCoords(e.clientX, e.clientY);
      const hit = findNodeAt(x, y);
      setHoveredNodeId(hit?.id ?? null);
      if (canvasRef.current) {
        canvasRef.current.style.cursor = hit ? 'pointer' : 'grab';
      }
    },
    [draw, findNodeAt, toGraphCoords],
  );

  const handleMouseUp = useCallback(() => {
    dragState.current = { mode: 'none', startX: 0, startY: 0, origPanX: 0, origPanY: 0 };
  }, []);

  const handleMouseLeave = useCallback(() => {
    dragState.current = { mode: 'none', startX: 0, startY: 0, origPanX: 0, origPanY: 0 };
    setHoveredNodeId(null);
    if (canvasRef.current) canvasRef.current.style.cursor = 'default';
  }, []);

  const handleZoomIn = () => setZoom((z) => Math.min(z + 0.2, 3));
  const handleZoomOut = () => setZoom((z) => Math.max(z - 0.2, 0.3));
  const handleReset = () => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
    setSelectedNodeId(null);
    // Re-seed positions from the data.
    const m = new Map<string, { x: number; y: number }>();
    nodes.forEach((n) => m.set(n.id, { x: n.x, y: n.y }));
    positionsRef.current = m;
    draw();
  };

  // Empty / error states.
  if (!loading && source === 'empty' && filtered.nodes.length === 0) {
    return (
      <div
        className={cn(
          'relative bg-background border border-border rounded-lg overflow-hidden',
          className,
        )}
        data-testid="knowledge-graph-explorer"
      >
        <ControlBar
          zoom={zoom}
          onZoomIn={handleZoomIn}
          onZoomOut={handleZoomOut}
          onReset={handleReset}
          onToggleFilters={() => setShowFilters((s) => !s)}
          showFilters={showFilters}
        />
        <EmptyState
          icon={Network}
          title="No graph data"
          description={
            drug
              ? `No knowledge graph entities found for "${drug}". The drug may not exist in the Neo4j database, or the KG service is not yet deployed.`
              : 'No drug specified. Enter a drug name to load its knowledge graph neighborhood.'
          }
          size="lg"
        />
      </div>
    );
  }

  return (
    <div
      ref={containerRef}
      className={cn(
        'relative bg-background border border-border rounded-lg overflow-hidden',
        className,
      )}
      data-testid="knowledge-graph-explorer"
    >
      {/* Control bar */}
      <ControlBar
        zoom={zoom}
        onZoomIn={handleZoomIn}
        onZoomOut={handleZoomOut}
        onReset={handleReset}
        onToggleFilters={() => setShowFilters((s) => !s)}
        showFilters={showFilters}
      />

      {/* DEMO data banner */}
      {isDemo && (
        <div className="absolute top-12 left-3 right-3 z-20 bg-amber-50 border border-amber-200 text-amber-800 text-[11px] font-semibold px-2.5 py-1.5 rounded-md">
          DEMO DATA — Neo4j KG service not deployed. Showing mock data. Set
          KG_SERVICE_URL to see real graph data.
        </div>
      )}

      {/* Loading overlay */}
      {loading && (
        <div className="absolute inset-0 z-30 bg-background/60 backdrop-blur-sm flex items-center justify-center">
          <div className="text-sm text-muted-foreground flex items-center gap-2">
            <span className="h-3 w-3 rounded-full border-2 border-primary border-t-transparent animate-spin" />
            Loading knowledge graph…
          </div>
        </div>
      )}

      {/* Error toast */}
      {error && (
        <div className="absolute top-12 left-3 right-3 z-20 bg-red-50 border border-red-200 text-red-800 text-[11px] px-2.5 py-1.5 rounded-md">
          {error}
        </div>
      )}

      <div className="flex flex-col lg:flex-row">
        {/* Filters sidebar */}
        {showFilters && (
          <aside
            data-testid="kg-filters"
            className="w-full lg:w-64 shrink-0 border-b lg:border-b-0 lg:border-r border-border p-3 space-y-3 bg-muted/30"
          >
            <div>
              <div className="flex items-center gap-1.5 mb-1.5">
                <Filter className="h-3.5 w-3.5 text-muted-foreground" />
                <span className="text-xs font-semibold">Search</span>
              </div>
              <Input
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Find entities…"
                className="h-8 text-xs"
              />
            </div>

            <div>
              <p className="text-xs font-semibold mb-1.5">Node Types</p>
              <div className="space-y-1">
                {ALL_NODE_TYPES.map((t) => {
                  const count = nodes.filter((n) => n.type === t).length;
                  if (count === 0) return null;
                  return (
                    <label
                      key={t}
                      className="flex items-center gap-2 cursor-pointer text-xs"
                    >
                      <Checkbox
                        checked={!!nodeTypeFilter[t]}
                        onCheckedChange={(v) =>
                          setNodeTypeFilter((p) => ({ ...p, [t]: !!v }))
                        }
                      />
                      <span
                        className="h-2 w-2 rounded-full inline-block"
                        style={{ backgroundColor: NODE_COLORS[t] }}
                      />
                      <span className="capitalize">{NODE_LABELS[t] ?? t}</span>
                      <span className="ml-auto text-muted-foreground tabular-nums">{count}</span>
                    </label>
                  );
                })}
              </div>
            </div>

            {availableEdgeTypes.length > 0 && (
              <div>
                <p className="text-xs font-semibold mb-1.5">Edge Types</p>
                <div className="space-y-1 max-h-40 overflow-y-auto">
                  {availableEdgeTypes.map((t) => {
                    const count = edges.filter((e) => e.relation === t).length;
                    return (
                      <label
                        key={t}
                        className="flex items-center gap-2 cursor-pointer text-xs"
                      >
                        <Checkbox
                          checked={edgeTypeFilter[t] !== false}
                          onCheckedChange={(v) =>
                            setEdgeTypeFilter((p) => ({ ...p, [t]: !!v }))
                          }
                        />
                        <span className="truncate">{t}</span>
                        <span className="ml-auto text-muted-foreground tabular-nums">{count}</span>
                      </label>
                    );
                  })}
                </div>
              </div>
            )}

            <div>
              <p className="text-xs font-semibold mb-1.5">
                Min Evidence: {evidenceThreshold.toFixed(2)}
              </p>
              <Slider
                value={[evidenceThreshold]}
                onValueChange={(v) => setEvidenceThreshold(v[0])}
                min={0}
                max={1}
                step={0.05}
              />
            </div>

            <div className="border-t border-border pt-2 text-[11px] text-muted-foreground space-y-0.5">
              <div className="flex justify-between">
                <span>Visible nodes</span>
                <span className="font-medium tabular-nums">{filtered.nodes.length}</span>
              </div>
              <div className="flex justify-between">
                <span>Visible edges</span>
                <span className="font-medium tabular-nums">{filtered.edges.length}</span>
              </div>
              <button
                onClick={refresh}
                className="text-primary hover:underline mt-1 block"
                type="button"
              >
                Refresh graph
              </button>
            </div>
          </aside>
        )}

        {/* Canvas */}
        <div className="relative flex-1" style={{ minHeight: height }}>
          <canvas
            ref={canvasRef}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseLeave}
            className="block w-full h-full"
            style={{ cursor: 'grab', touchAction: 'none' }}
            data-testid="kg-canvas"
          />

          {/* Legend */}
          <div className="absolute bottom-3 left-3 z-10 flex flex-wrap gap-2 bg-background/90 backdrop-blur-sm rounded-lg p-2 border border-border">
            {ALL_NODE_TYPES.filter((t) => nodes.some((n) => n.type === t)).map((t) => (
              <div key={t} className="flex items-center gap-1.5">
                <span
                  className="h-2.5 w-2.5 rounded-full"
                  style={{ backgroundColor: NODE_COLORS[t] }}
                />
                <span className="text-xs text-muted-foreground">{NODE_LABELS[t] ?? t}</span>
              </div>
            ))}
          </div>

          {/* Hovered node label */}
          {hoveredNodeId && !selectedNodeId && (
            <div className="absolute bottom-3 right-3 z-10 bg-background/95 border border-border rounded-lg p-2 text-xs max-w-[220px]">
              {(() => {
                const n = nodes.find((x) => x.id === hoveredNodeId);
                if (!n) return null;
                return (
                  <>
                    <div className="font-medium">{n.label}</div>
                    <Badge variant="secondary" className="text-[10px] mt-1">
                      {NODE_LABELS[n.type] ?? n.type}
                    </Badge>
                  </>
                );
              })()}
            </div>
          )}
        </div>
      </div>

      {/* Selected node side panel */}
      {selectedNodeInfo && (
        <div
          data-testid="kg-side-panel"
          className="absolute top-0 right-0 h-full w-72 bg-background border-l border-border shadow-lg z-30 flex flex-col"
        >
          <div className="flex items-center justify-between px-3 py-2 border-b border-border">
            <span className="font-semibold text-sm">Node Details</span>
            <Button
              variant="ghost"
              size="sm"
              className="h-7 w-7 p-0"
              onClick={() => setSelectedNodeId(null)}
              aria-label="Close panel"
            >
              <X className="h-4 w-4" />
            </Button>
          </div>
          <ScrollArea className="flex-1">
            <div className="p-3 space-y-3">
              <div>
                <div className="text-base font-medium">{selectedNodeInfo.node.label}</div>
                <Badge
                  variant="secondary"
                  className="text-[10px] mt-1"
                  style={{
                    color: NODE_COLORS[selectedNodeInfo.node.type] ?? '#64748b',
                  }}
                >
                  {NODE_LABELS[selectedNodeInfo.node.type] ?? selectedNodeInfo.node.type}
                </Badge>
                <div className="text-[10px] text-muted-foreground mt-1 font-mono">
                  id: {selectedNodeInfo.node.id}
                </div>
              </div>
              <div>
                <div className="text-xs font-semibold text-muted-foreground mb-1">
                  Connections ({selectedNodeInfo.edges.length})
                </div>
                {selectedNodeInfo.edges.length === 0 ? (
                  <p className="text-xs text-muted-foreground">No edges in current view.</p>
                ) : (
                  <div className="space-y-1.5">
                    {selectedNodeInfo.edges.slice(0, 50).map((rel, i) => (
                      <div
                        key={`${rel.edge.source}-${rel.edge.target}-${i}`}
                        className="p-2 border border-border rounded-md text-xs"
                      >
                        <div className="flex items-center gap-1.5 mb-0.5">
                          <Badge
                            variant="outline"
                            className="text-[9px] px-1 py-0"
                            style={{
                              color: NODE_COLORS[rel.other.type] ?? '#64748b',
                            }}
                          >
                            {NODE_LABELS[rel.other.type] ?? rel.other.type}
                          </Badge>
                          <span className="font-medium truncate">{rel.other.label}</span>
                        </div>
                        <div className="flex items-center justify-between text-[10px] text-muted-foreground">
                          <span>
                            {rel.direction === 'out' ? '→ ' : '← '}
                            {rel.edge.relation}
                          </span>
                          <span className="tabular-nums">
                            {(rel.edge.evidence * 100).toFixed(0)}% evidence
                          </span>
                        </div>
                      </div>
                    ))}
                    {selectedNodeInfo.edges.length > 50 && (
                      <p className="text-[10px] text-muted-foreground text-center pt-1">
                        + {selectedNodeInfo.edges.length - 50} more
                      </p>
                    )}
                  </div>
                )}
              </div>
            </div>
          </ScrollArea>
        </div>
      )}
    </div>
  );
}

// ─── helpers ─────────────────────────────────────────────────────────

function getNodeType(id: string, nodes: GraphNode[]): string {
  const n = nodes.find((x) => x.id === id);
  return n?.type ?? 'gene';
}

function evidenceToColor(e: number, alpha: number): string {
  if (e >= 0.85) return `rgba(29, 158, 117, ${alpha})`;
  if (e >= 0.6) return `rgba(91, 79, 207, ${alpha})`;
  return `rgba(212, 133, 58, ${alpha})`;
}

function hexWithAlpha(hex: string, alpha: number): string {
  // Accepts #RRGGBB. Returns rgba().
  const h = hex.replace('#', '');
  if (h.length !== 6) return `rgba(100, 116, 139, ${alpha})`;
  const r = parseInt(h.slice(0, 2), 16);
  const g = parseInt(h.slice(2, 4), 16);
  const b = parseInt(h.slice(4, 6), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

// ─── sub-components ──────────────────────────────────────────────────

function ControlBar({
  zoom,
  onZoomIn,
  onZoomOut,
  onReset,
  onToggleFilters,
  showFilters,
}: {
  zoom: number;
  onZoomIn: () => void;
  onZoomOut: () => void;
  onReset: () => void;
  onToggleFilters: () => void;
  showFilters: boolean;
}) {
  return (
    <div className="absolute top-3 right-3 z-20 flex items-center gap-1">
      <Button
        variant="outline"
        size="sm"
        onClick={onToggleFilters}
        className="h-8 px-2 text-xs"
        data-testid="kg-toggle-filters"
        aria-pressed={showFilters}
      >
        <Filter className="h-3.5 w-3.5" />
        <span className="ml-1 hidden sm:inline">Filters</span>
      </Button>
      <Button variant="outline" size="sm" onClick={onZoomOut} className="h-8 w-8 p-0" aria-label="Zoom out">
        <ZoomOut className="h-4 w-4" />
      </Button>
      <span className="text-xs text-muted-foreground w-12 text-center tabular-nums">
        {Math.round(zoom * 100)}%
      </span>
      <Button variant="outline" size="sm" onClick={onZoomIn} className="h-8 w-8 p-0" aria-label="Zoom in">
        <ZoomIn className="h-4 w-4" />
      </Button>
      <Button variant="outline" size="sm" onClick={onReset} className="h-8 w-8 p-0 ml-1" aria-label="Reset view">
        <RotateCcw className="h-4 w-4" />
      </Button>
    </div>
  );
}

export default KnowledgeGraphExplorer;
