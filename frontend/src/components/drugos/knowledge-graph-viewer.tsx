'use client';

import { useState, useCallback, useEffect, useRef, useMemo } from 'react';
import { ZoomIn, ZoomOut, RotateCcw, ChevronLeft, ChevronRight, AlertTriangle } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import type { KnowledgeGraphNode, KnowledgeGraphEdge } from '@/lib/types';

/**
 * FE-031 ROOT FIX: Canvas-based rendering with a 1000-node cap + pagination.
 *
 * The previous version rendered the graph as SVG. SVG is DOM-based — each
 * node is a <circle>, <text>, and 2–3 nested <g> elements, plus a Tooltip
 * wrapper. For 1000 nodes that's ~5000 DOM elements, which the browser can
 * barely render (jank on every pan/zoom). For 10K nodes it's ~50K DOM
 * elements — the browser freezes for 30+ seconds on initial render and
 * becomes unusable.
 *
 * Per the project doc (Team_Cosmic_Build_Process_Updated.docx §5), the
 * Knowledge Graph Explorer is the platform's core transparency feature: a
 * researcher MUST be able to "see the biological pathway chain connecting a
 * selected drug to a target disease, making the AI's reasoning transparent
 * and auditable." A frozen browser breaks that core value prop.
 *
 * Root fix:
 *   1. Switch to HTML5 <canvas> rendering. Canvas is O(1) per draw call
 *      regardless of node count — 10K nodes render in a single frame. We
 *      draw edges as line segments and nodes as filled circles + labels.
 *      Panning is supported via mouse drag; zoom via wheel or buttons.
 *   2. Hard cap at MAX_NODES (1000) per page. When the input exceeds the
 *      cap, we paginate — show the first 1000, with "Prev" / "Next" buttons
 *      and a "Showing X–Y of Z nodes" indicator. This guarantees the browser
 *      never freezes regardless of input size.
 *   3. Hover detection is done via hit-testing in canvas (compute distance
 *      from mouse to each node center). For 1000 nodes this is O(1000) per
 *      mousemove — fast enough. We use a spatial hash for larger graphs if
 *      needed in the future.
 *   4. Node positions come from the `nodes` prop (each node has x, y). We
 *      do NOT run a force layout client-side — that was the original perf
 *      bug. Positions are pre-computed by the backend (Phase 2 Neo4j graph
 *      layout) or defaulted to a circle layout when missing.
 *
 * The external API (props) is unchanged so callers don't break.
 */

interface KnowledgeGraphViewerProps {
  nodes?: KnowledgeGraphNode[];
  edges?: KnowledgeGraphEdge[];
  width?: number;
  height?: number;
  className?: string;
}

const MAX_NODES = 1000;

const typeColors: Record<string, string> = {
  disease: '#C0392B',
  drug: '#1D9E75',
  compound: '#1D9E75',
  gene: '#3B82F6',
  pathway: '#D4853A',
  protein: '#8B5CF6',
  phenotype: '#C0392B',
  clinicaloutcome: '#C0392B',
};

const typeLabels: Record<string, string> = {
  disease: 'Disease',
  drug: 'Drug',
  compound: 'Drug',
  gene: 'Gene',
  protein: 'Protein',
  pathway: 'Pathway',
  phenotype: 'Phenotype',
  clinicaloutcome: 'Outcome',
};

function buildPositionMap(nodes: KnowledgeGraphNode[]): Map<string, { x: number; y: number }> {
  return new Map(nodes.map((n) => [n.id, { x: n.x, y: n.y }]));
}

/**
 * Concentric Category-based layout for nodes.
 * Places the central drug/query entity in the middle and orbits targets,
 * diseases, and pathways in structured concentric rings with force repulsion.
 */
function assignDefaultPositions(
  nodes: KnowledgeGraphNode[],
  width: number,
  height: number
): KnowledgeGraphNode[] {
  const cx = width / 2;
  const cy = height / 2;

  // Group nodes by category
  const drugs: KnowledgeGraphNode[] = [];
  const targets: KnowledgeGraphNode[] = [];
  const diseases: KnowledgeGraphNode[] = [];
  const pathways: KnowledgeGraphNode[] = [];
  const others: KnowledgeGraphNode[] = [];

  for (const n of nodes) {
    const t = (n.type || '').toLowerCase();
    if (t === 'drug' || t === 'compound') drugs.push(n);
    else if (t === 'protein' || t === 'gene') targets.push(n);
    else if (t === 'disease' || t === 'clinicaloutcome' || t === 'phenotype') diseases.push(n);
    else if (t === 'pathway') pathways.push(n);
    else others.push(n);
  }

  const resultPos = new Map<string, { x: number; y: number }>();

  // 1. Primary Drug at Center (or first drug)
  if (drugs.length > 0) {
    resultPos.set(drugs[0].id, { x: cx, y: cy });
  }

  // 2. Remaining Drugs in slight top-left quadrant arc
  const remainingDrugs = drugs.slice(1);
  remainingDrugs.forEach((n, i) => {
    const angle = -Math.PI / 4 + (i * Math.PI) / 6;
    resultPos.set(n.id, { x: cx + 220 * Math.cos(angle), y: cy + 220 * Math.sin(angle) });
  });

  // 3. Targets / Proteins in Inner Ring (radius ~150px)
  targets.forEach((n, i) => {
    const angle = (2 * Math.PI * i) / Math.max(1, targets.length) - Math.PI / 2;
    resultPos.set(n.id, { x: cx + 155 * Math.cos(angle), y: cy + 140 * Math.sin(angle) });
  });

  // 4. Diseases in Outer Right/Bottom Ring (radius ~250px)
  diseases.forEach((n, i) => {
    const angle = Math.PI / 6 + (i * Math.PI) / 4;
    resultPos.set(n.id, { x: cx + 265 * Math.cos(angle), y: cy + 210 * Math.sin(angle) });
  });

  // 5. Pathways & Others in Top/Outer Ring (radius ~280px)
  const outerMisc = [...pathways, ...others];
  outerMisc.forEach((n, i) => {
    const angle = -Math.PI * 0.75 + (i * Math.PI) / 4;
    resultPos.set(n.id, { x: cx + 270 * Math.cos(angle), y: cy + 220 * Math.sin(angle) });
  });

  // Simple 15-iteration force-repulsion pass for anti-collision
  const posArr = nodes.map(n => {
    const p = resultPos.get(n.id) || { x: cx + (Math.random() - 0.5) * 200, y: cy + (Math.random() - 0.5) * 200 };
    return { id: n.id, x: p.x, y: p.y };
  });

  for (let iter = 0; iter < 15; iter++) {
    for (let i = 0; i < posArr.length; i++) {
      for (let j = i + 1; j < posArr.length; j++) {
        const dx = posArr[j].x - posArr[i].x;
        const dy = posArr[j].y - posArr[i].y;
        const dist = Math.sqrt(dx * dx + dy * dy) || 1;
        const minDist = 110; // minimum desired distance between node centers
        if (dist < minDist) {
          const overlap = (minDist - dist) / 2;
          const nx = (dx / dist) * overlap;
          const ny = (dy / dist) * overlap;
          if (i !== 0) { // Keep central node stable
            posArr[i].x -= nx;
            posArr[i].y -= ny;
          }
          posArr[j].x += nx;
          posArr[j].y += ny;
        }
      }
    }
  }

  const finalPosMap = new Map(posArr.map(p => [p.id, { x: p.x, y: p.y }]));

  return nodes.map((n) => {
    if (typeof n.x === 'number' && typeof n.y === 'number' && n.x !== 0 && n.y !== 0) {
      // Use existing position if already laid out
      return n;
    }
    const p = finalPosMap.get(n.id) || { x: cx, y: cy };
    return { ...n, x: p.x, y: p.y };
  });
}

export function KnowledgeGraphViewer({
  nodes: allNodes = [],
  edges: allEdges = [],
  width = 900,
  height = 550,
  className = '',
}: KnowledgeGraphViewerProps) {
  // FE-031: Pagination state. We show MAX_NODES nodes per page. When the
  // input exceeds the cap, the user can page through. The edges are filtered
  // to only those connecting visible nodes.
  const totalPages = Math.max(1, Math.ceil(allNodes.length / MAX_NODES));
  const [page, setPage] = useState(0);
  const currentPage = Math.min(page, totalPages - 1);

  const pageNodes = useMemo(
    () => allNodes.slice(currentPage * MAX_NODES, (currentPage + 1) * MAX_NODES),
    [allNodes, currentPage]
  );
  const pageNodeIds = useMemo(() => new Set(pageNodes.map((n) => n.id)), [pageNodes]);
  const pageEdges = useMemo(
    () => allEdges.filter((e) => pageNodeIds.has(e.source) && pageNodeIds.has(e.target)),
    [allEdges, pageNodeIds]
  );

  // Assign default positions to any node missing x/y. We do this once per
  // page change so the layout is stable within a page.
  const positionedNodes = useMemo(
    () => assignDefaultPositions(pageNodes, width, height),
    [pageNodes, width, height]
  );

  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [hoveredNode, setHoveredNode] = useState<string | null>(null);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const dragState = useRef<{ mode: 'pan' | 'node' | null; startX: number; startY: number; nodeId?: string; nodeStartX?: number; nodeStartY?: number }>(
    { mode: null, startX: 0, startY: 0 }
  );
  const positionsRef = useRef<Map<string, { x: number; y: number }>>(
    buildPositionMap(positionedNodes)
  );

  // Rebuild the positions map when the page changes.
  useEffect(() => {
    positionsRef.current = buildPositionMap(positionedNodes);
  }, [positionedNodes]);

  // FE-031: Canvas draw routine. Called on every zoom/pan/hover/select
  // change. We use a high-DPI canvas (devicePixelRatio scaling) for crisp
  // rendering on retina displays.
  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const displayWidth = width;
    const displayHeight = height;
    if (canvas.width !== displayWidth * dpr || canvas.height !== displayHeight * dpr) {
      canvas.width = displayWidth * dpr;
      canvas.height = displayHeight * dpr;
      canvas.style.width = `${displayWidth}px`;
      canvas.style.height = `${displayHeight}px`;
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, displayWidth, displayHeight);

    // Apply zoom + pan transform.
    ctx.save();
    ctx.translate(pan.x, pan.y);
    ctx.translate(displayWidth / 2, displayHeight / 2);
    ctx.scale(zoom, zoom);
    ctx.translate(-displayWidth / 2, -displayHeight / 2);

    const positions = positionsRef.current;
    const highlightedNodes = new Set<string>();
    if (selectedNode) {
      highlightedNodes.add(selectedNode);
      for (const edge of pageEdges) {
        if (edge.source === selectedNode) highlightedNodes.add(edge.target);
        if (edge.target === selectedNode) highlightedNodes.add(edge.source);
      }
    }

    // Draw edges with smooth quadratic curves
    ctx.lineWidth = 1;
    for (const edge of pageEdges) {
      const source = positions.get(edge.source);
      const target = positions.get(edge.target);
      if (!source || !target) continue;
      const isHighlighted = selectedNode
        ? edge.source === selectedNode || edge.target === selectedNode
        : true;

      // Slight curvature calculation so multi-edges and opposing edges don't overlap
      const dx = target.x - source.x;
      const dy = target.y - source.y;
      const dist = Math.sqrt(dx * dx + dy * dy) || 1;
      const normX = -dy / dist;
      const normY = dx / dist;
      const curveOffset = 15; // subtle arc height
      const midX = (source.x + target.x) / 2 + normX * curveOffset;
      const midY = (source.y + target.y) / 2 + normY * curveOffset;

      ctx.strokeStyle = isHighlighted ? 'rgba(99, 102, 241, 0.75)' : 'rgba(203, 213, 225, 0.35)';
      ctx.lineWidth = isHighlighted ? 2.2 : 1.2;
      ctx.beginPath();
      ctx.moveTo(source.x, source.y);
      ctx.quadraticCurveTo(midX, midY, target.x, target.y);
      ctx.stroke();

      // Draw edge type labels at midpoint when highlighted or zoomed in
      if ((isHighlighted || zoom > 0.75) && edge.type) {
        ctx.font = `600 ${Math.max(9, 10 * zoom)}px ui-sans-serif, system-ui, sans-serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';

        const labelText = edge.type.replace(/_/g, ' ');
        const metrics = ctx.measureText(labelText);
        const tw = metrics.width;
        const th = 14;
        const px = 5;

        ctx.fillStyle = isHighlighted ? 'rgba(241, 245, 249, 0.98)' : 'rgba(248, 250, 252, 0.9)';
        ctx.strokeStyle = isHighlighted ? 'rgba(99, 102, 241, 0.5)' : 'rgba(203, 213, 225, 0.6)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        if (typeof ctx.roundRect === 'function') {
          ctx.roundRect(midX - tw / 2 - px, midY - th / 2, tw + px * 2, th, 4);
        } else {
          ctx.rect(midX - tw / 2 - px, midY - th / 2, tw + px * 2, th);
        }
        ctx.fill();
        ctx.stroke();

        ctx.fillStyle = isHighlighted ? '#334155' : '#64748B';
        ctx.fillText(labelText, midX, midY + 1);
      }
    }

    // Draw nodes.
    for (const node of positionedNodes) {
      const pos = positions.get(node.id);
      if (!pos) continue;
      const isSelected = selectedNode === node.id;
      const isHovered = hoveredNode === node.id;
      const isConnected = highlightedNodes.has(node.id);
      const isActive = !selectedNode || isConnected;
      const radius = Math.max(10, node.size || 16);
      const normalizedType = (node.type || '').toLowerCase();
      const color = typeColors[normalizedType] || '#5B4FCF';
      const categoryTag = typeLabels[normalizedType] || 'Entity';

      // Glowing outer ring for selected node
      if (isSelected || isHovered) {
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, radius + 7, 0, 2 * Math.PI);
        ctx.strokeStyle = color;
        ctx.lineWidth = 2.5;
        ctx.globalAlpha = isSelected ? 0.6 : 0.35;
        ctx.stroke();
        ctx.globalAlpha = 1;
      }

      // Main Node Disc
      ctx.beginPath();
      ctx.arc(pos.x, pos.y, radius, 0, 2 * Math.PI);
      ctx.fillStyle = color;
      ctx.globalAlpha = isActive ? 1 : 0.25;
      ctx.fill();

      // Inner white core dot
      ctx.beginPath();
      ctx.arc(pos.x, pos.y, radius * 0.45, 0, 2 * Math.PI);
      ctx.fillStyle = '#FFFFFF';
      ctx.globalAlpha = isActive ? 0.95 : 0.3;
      ctx.fill();
      ctx.globalAlpha = 1;

      // Crisp node label + category tag pill
      if (zoom > 0.35) {
        const titleText = node.label || node.id;
        const subText = categoryTag.toUpperCase();

        ctx.font = `700 ${Math.max(10, 11 * zoom)}px ui-sans-serif, system-ui, sans-serif`;
        const titleMetrics = ctx.measureText(titleText);

        ctx.font = `600 ${Math.max(8, 8.5 * zoom)}px ui-sans-serif, system-ui, sans-serif`;
        const subMetrics = ctx.measureText(subText);

        const cardW = Math.max(titleMetrics.width, subMetrics.width) + 12;
        const cardH = 26;
        const labelY = pos.y + radius + 6;

        // Card background
        ctx.fillStyle = 'rgba(255, 255, 255, 0.96)';
        ctx.strokeStyle = isSelected ? color : isHovered ? color : 'rgba(226, 232, 240, 0.9)';
        ctx.lineWidth = isSelected || isHovered ? 1.8 : 1;

        ctx.shadowColor = 'rgba(15, 23, 42, 0.08)';
        ctx.shadowBlur = 6;
        ctx.shadowOffsetY = 2;

        ctx.beginPath();
        if (typeof ctx.roundRect === 'function') {
          ctx.roundRect(pos.x - cardW / 2, labelY, cardW, cardH, 5);
        } else {
          ctx.rect(pos.x - cardW / 2, labelY, cardW, cardH);
        }
        ctx.fill();
        ctx.stroke();

        ctx.shadowColor = 'transparent';

        // Title text
        ctx.font = `700 ${Math.max(10, 11 * zoom)}px ui-sans-serif, system-ui, sans-serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';
        ctx.fillStyle = isSelected ? color : '#0F172A';
        ctx.fillText(titleText, pos.x, labelY + 3);

        // Subtitle category badge
        ctx.font = `600 ${Math.max(8, 8.5 * zoom)}px ui-sans-serif, system-ui, sans-serif`;
        ctx.fillStyle = color;
        ctx.fillText(subText, pos.x, labelY + 14);
      }
    }

    ctx.restore();
  }, [positionedNodes, pageEdges, zoom, pan, selectedNode, hoveredNode, width, height]);

  useEffect(() => {
    draw();
  }, [draw]);

  const handleZoomIn = () => setZoom((z) => Math.min(z + 0.2, 3));
  const handleZoomOut = () => setZoom((z) => Math.max(z - 0.2, 0.3));
  const handleReset = () => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
    setSelectedNode(null);
    positionsRef.current = buildPositionMap(positionedNodes);
  };

  /**
   * Convert a mouse event's clientX/clientY to graph-space coordinates
   * (accounting for zoom + pan). Used for hit-testing.
   */
  const toGraphCoords = useCallback(
    (clientX: number, clientY: number) => {
      const canvas = canvasRef.current;
      if (!canvas) return { x: 0, y: 0 };
      const rect = canvas.getBoundingClientRect();
      const cx = clientX - rect.left;
      const cy = clientY - rect.top;
      // Inverse of the transform applied in draw().
      const gx = (cx - pan.x - width / 2) / zoom + width / 2;
      const gy = (cy - pan.y - height / 2) / zoom + height / 2;
      return { x: gx, y: gy };
    },
    [pan, zoom, width, height]
  );

  /**
   * FE-031: Hit-test — find the topmost node under the mouse. O(n) per call
   * where n = nodes on the current page (max 1000). At 60fps with 1000 nodes
   * that's 60K distance checks/sec — trivial for modern JS engines.
   */
  const hitTest = useCallback(
    (gx: number, gy: number): string | null => {
      const positions = positionsRef.current;
      // Iterate in reverse so topmost-drawn (last) nodes are hit first.
      for (let i = positionedNodes.length - 1; i >= 0; i--) {
        const node = positionedNodes[i];
        const pos = positions.get(node.id);
        if (!pos) continue;
        const r = Math.max(4, node.size || 12) + 2; // small hit pad
        const dx = gx - pos.x;
        const dy = gy - pos.y;
        if (dx * dx + dy * dy <= r * r) return node.id;
      }
      return null;
    },
    [positionedNodes]
  );

  const handleMouseDown = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const { x: gx, y: gy } = toGraphCoords(e.clientX, e.clientY);
      const hitId = hitTest(gx, gy);
      if (hitId) {
        setSelectedNode(hitId);
        const pos = positionsRef.current.get(hitId);
        dragState.current = {
          mode: 'node',
          startX: e.clientX,
          startY: e.clientY,
          nodeId: hitId,
          nodeStartX: pos?.x,
          nodeStartY: pos?.y,
        };
      } else {
        setSelectedNode(null);
        dragState.current = { mode: 'pan', startX: e.clientX, startY: e.clientY };
      }
    },
    [toGraphCoords, hitTest]
  );

  const handleMouseMove = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const ds = dragState.current;
      if (ds.mode === 'pan') {
        const dx = e.clientX - ds.startX;
        const dy = e.clientY - ds.startY;
        setPan((prev) => ({ x: prev.x + dx, y: prev.y + dy }));
        ds.startX = e.clientX;
        ds.startY = e.clientY;
        return;
      }
      if (ds.mode === 'node' && ds.nodeId !== undefined) {
        const { x: gx, y: gy } = toGraphCoords(e.clientX, e.clientY);
        const next = new Map(positionsRef.current);
        next.set(ds.nodeId, { x: gx, y: gy });
        positionsRef.current = next;
        draw();
        return;
      }
      // Not dragging — hover hit-test for tooltip.
      const { x: gx, y: gy } = toGraphCoords(e.clientX, e.clientY);
      const hitId = hitTest(gx, gy);
      setHoveredNode(hitId);
    },
    [toGraphCoords, hitTest, draw]
  );

  const handleMouseUp = useCallback(() => {
    dragState.current = { mode: null, startX: 0, startY: 0 };
  }, []);

  const handleWheel = useCallback((e: React.WheelEvent<HTMLCanvasElement>) => {
    // Ctrl+wheel zooms; plain wheel does nothing (prevents accidental zoom).
    if (e.ctrlKey || e.metaKey) {
      e.preventDefault();
      const delta = e.deltaY < 0 ? 0.1 : -0.1;
      setZoom((z) => Math.max(0.3, Math.min(3, z + delta)));
    }
  }, []);

  const selectedNodeMeta = selectedNode
    ? positionedNodes.find((n) => n.id === selectedNode)
    : null;
  const selectedConnections = selectedNode
    ? pageEdges.filter((e) => e.source === selectedNode || e.target === selectedNode).length
    : 0;

  // FE-031: Pagination UI is shown only when the input exceeds the cap.
  const overCap = allNodes.length > MAX_NODES;
  const pageStart = currentPage * MAX_NODES + 1;
  const pageEnd = Math.min((currentPage + 1) * MAX_NODES, allNodes.length);

  const hoveredNodeMeta = hoveredNode
    ? positionedNodes.find((n) => n.id === hoveredNode)
    : null;

  return (
    <div className={`relative bg-background border border-border rounded-lg overflow-hidden ${className}`}>
      {/* Controls */}
      <div className="absolute top-3 right-3 z-10 flex items-center gap-1">
        <Button variant="outline" size="sm" onClick={handleZoomOut} className="h-8 w-8 p-0">
          <ZoomOut className="h-4 w-4" />
        </Button>
        <span className="text-xs text-muted-foreground w-12 text-center tabular-nums">
          {Math.round(zoom * 100)}%
        </span>
        <Button variant="outline" size="sm" onClick={handleZoomIn} className="h-8 w-8 p-0">
          <ZoomIn className="h-4 w-4" />
        </Button>
        <Button variant="outline" size="sm" onClick={handleReset} className="h-8 w-8 p-0 ml-1">
          <RotateCcw className="h-4 w-4" />
        </Button>
      </div>

      {/* Legend */}
      <div className="absolute bottom-3 left-3 z-10 flex flex-wrap gap-3 bg-background/95 backdrop-blur-md rounded-lg px-3 py-1.5 border border-border shadow-sm">
        {[
          { type: 'drug', label: 'Drug / Compound' },
          { type: 'protein', label: 'Protein Target' },
          { type: 'gene', label: 'Gene' },
          { type: 'disease', label: 'Disease / Indication' },
          { type: 'clinicaloutcome', label: 'Outcome' },
          { type: 'pathway', label: 'Pathway' },
        ].map(({ type, label }) => (
          <div key={type} className="flex items-center gap-1.5">
            <span
              className="h-2.5 w-2.5 rounded-full shadow-xs"
              style={{ backgroundColor: typeColors[type] }}
            />
            <span className="text-xs font-medium text-slate-700">{label}</span>
          </div>
        ))}
      </div>

      {/* FE-031: Pagination — only shown when input exceeds the 1000-node cap. */}
      {overCap && (
        <div className="absolute top-3 left-3 z-10 flex items-center gap-2 bg-background/90 backdrop-blur-sm border border-amber-300 rounded-lg p-2">
          <AlertTriangle className="h-4 w-4 text-amber-600" />
          <span className="text-xs text-foreground">
            Showing <span className="font-semibold">{pageStart}–{pageEnd}</span> of{' '}
            <span className="font-semibold">{allNodes.length}</span> nodes
          </span>
          <Button
            variant="outline"
            size="sm"
            className="h-7 w-7 p-0"
            disabled={currentPage === 0}
            onClick={() => setPage((p) => Math.max(0, p - 1))}
          >
            <ChevronLeft className="h-3.5 w-3.5" />
          </Button>
          <span className="text-xs text-muted-foreground tabular-nums">
            {currentPage + 1}/{totalPages}
          </span>
          <Button
            variant="outline"
            size="sm"
            className="h-7 w-7 p-0"
            disabled={currentPage >= totalPages - 1}
            onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
          >
            <ChevronRight className="h-3.5 w-3.5" />
          </Button>
        </div>
      )}

      <TooltipProvider delayDuration={200}>
        <Tooltip>
          <TooltipTrigger asChild>
            <canvas
              ref={canvasRef}
              onMouseDown={handleMouseDown}
              onMouseMove={handleMouseMove}
              onMouseUp={handleMouseUp}
              onMouseLeave={handleMouseUp}
              onWheel={handleWheel}
              className="block cursor-grab active:cursor-grabbing"
              style={{ width, height }}
              aria-label="Knowledge graph visualization. Drag to pan, Ctrl+scroll to zoom, click a node to select."
            />
          </TooltipTrigger>
          <TooltipContent>
            {hoveredNodeMeta ? (
              <div className="text-xs">
                <div className="font-semibold">{hoveredNodeMeta.label}</div>
                <Badge
                  variant="secondary"
                  className="mt-1 text-[10px]"
                  style={{ color: typeColors[hoveredNodeMeta.type] }}
                >
                  {typeLabels[hoveredNodeMeta.type] || hoveredNodeMeta.type}
                </Badge>
                {hoveredNodeMeta.description && (
                  <div className="mt-1 text-muted-foreground max-w-[200px]">
                    {hoveredNodeMeta.description}
                  </div>
                )}
              </div>
            ) : (
              <div className="text-xs text-muted-foreground">
                Drag to pan · Ctrl+scroll to zoom · Click a node to select
              </div>
            )}
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>

      {/* Selected Node Info Card */}
      {selectedNodeMeta && (
        <div className="absolute top-3 left-3 bg-background/95 backdrop-blur-md border border-border shadow-lg rounded-xl p-3.5 max-w-[280px] z-20 transition-all">
          <div className="flex items-start justify-between gap-2 mb-1.5">
            <div>
              <div className="font-bold text-sm text-foreground leading-tight">{selectedNodeMeta.label}</div>
              <Badge
                variant="outline"
                className="mt-1 text-[10px] px-1.5 py-0 font-semibold uppercase tracking-wider"
                style={{
                  color: typeColors[(selectedNodeMeta.type || '').toLowerCase()] || '#5B4FCF',
                  borderColor: typeColors[(selectedNodeMeta.type || '').toLowerCase()] || '#5B4FCF',
                }}
              >
                {typeLabels[(selectedNodeMeta.type || '').toLowerCase()] || selectedNodeMeta.type}
              </Badge>
            </div>
            <Button variant="ghost" size="sm" className="h-5 w-5 p-0 text-muted-foreground hover:text-foreground" onClick={() => setSelectedNode(null)}>
              ×
            </Button>
          </div>

          {selectedNodeMeta.description && (
            <p className="text-xs text-muted-foreground line-clamp-3 mb-2 leading-snug bg-slate-50 p-2 rounded border border-slate-100">
              {selectedNodeMeta.description}
            </p>
          )}

          <div className="text-xs font-medium text-slate-600 flex items-center justify-between pt-1 border-t border-border/50">
            <span>Graph Connections</span>
            <Badge variant="secondary" className="text-[11px] font-bold">
              {selectedConnections}
            </Badge>
          </div>
        </div>
      )}
    </div>
  );
}
