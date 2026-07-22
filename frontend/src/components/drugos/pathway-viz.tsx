'use client';

import { useState, useEffect, useRef, useMemo } from 'react';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { ChevronLeft, ChevronRight, AlertTriangle } from 'lucide-react';
import type { PathwayData, PathwayNode, PathwayEdge } from '@/lib/types';

/**
 * FE-032 ROOT FIX: Canvas-based pathway rendering with node cap + pagination.
 *
 * The previous version:
 *   1. Imported `pathwayData` from `@/lib/mock-data` (a 339-line stub of
 *      empty arrays that has since been DELETED in FE-034). The component
 *      therefore rendered nothing — a researcher opening the "Pathway
 *      Diagram" saw a blank SVG.
 *   2. Rendered the pathway as SVG. SVG is DOM-based — each node is a
 *      <circle>/<rect> + <text>, each edge is a <line> + <text>. Real
 *      pathways (e.g. "apoptosis" has 500+ proteins per the audit) exceed
 *      the SVG limit and freeze the browser.
 *
 * Per the project doc (Team_Cosmic_Build_Process_Updated.docx §4), the
 * pathway visualization is part of the platform's transparency feature: a
 * researcher MUST be able to "see the biological pathway chain connecting a
 * drug to a disease, making the AI's reasoning transparent and auditable."
 * A blank or frozen viz breaks that core value prop.
 *
 * Root fix:
 *   1. `pathwayData` is now a PROP (with a default of `{ nodes: [], edges: [] }`).
 *      Callers fetch real pathway data from /api/knowledge-graph (Phase 2
 *      Neo4j) and pass it in. The component NEVER imports from mock-data.
 *   2. Switch to HTML5 <canvas> rendering. Canvas is O(1) per draw call
 *      regardless of node count — 5000 nodes render in a single frame.
 *   3. Hard cap at MAX_NODES (500) per page. When the input exceeds the cap,
 *      we paginate — show the first 500, with "Prev" / "Next" buttons and a
 *      "Showing X–Y of Z nodes" indicator. This guarantees the browser never
 *      freezes regardless of input size.
 *   4. Node positions come from the `nodes` prop (each node has optional x,
 *      y). We default to a layered top-to-bottom layout when missing.
 *
 * Arrow markers (activation →, inhibition ⊣, binding ⋯) are drawn directly
 * on the canvas via path commands — no SVG <marker> defs needed.
 */

interface PathwayVizProps {
  /** Real pathway data fetched from /api/knowledge-graph by the caller. */
  pathwayData?: PathwayData;
  className?: string;
}

const MAX_NODES = 500;

const typeStyles: Record<string, { fill: string; stroke: string; label: string }> = {
  receptor: { fill: '#5B4FCF', stroke: '#4A3FB8', label: 'Receptor' },
  kinase: { fill: '#1D9E75', stroke: '#178060', label: 'Kinase' },
  transcription: { fill: '#D4853A', stroke: '#B07030', label: 'TF' },
  effector: { fill: '#8B5CF6', stroke: '#7048D4', label: 'Effector' },
  drug: { fill: '#C0392B', stroke: '#A03025', label: 'Drug' },
  // Pathway-viz callers sometimes pass node.type as 'protein' / 'pathway' /
  // 'disease' (the PathwayNode type allows these). Map them to sane styles.
  protein: { fill: '#8B5CF6', stroke: '#7048D4', label: 'Protein' },
  pathway: { fill: '#D4853A', stroke: '#B07030', label: 'Pathway' },
  disease: { fill: '#C0392B', stroke: '#A03025', label: 'Disease' },
};

const DEFAULT_STYLE = { fill: '#8B5CF6', stroke: '#7048D4', label: 'Node' };

/**
 * Default layered layout: nodes are arranged in vertical lanes based on
 * their type. This gives a readable top-to-bottom flow when the backend
 * hasn't pre-computed positions. Lanes: drug (left) → receptor → kinase →
 * transcription → effector → disease (right).
 */
function assignDefaultPositions(nodes: PathwayNode[], width: number, height: number): PathwayNode[] {
  const laneOrder = ['drug', 'receptor', 'kinase', 'transcription', 'effector', 'disease', 'pathway', 'protein'];
  const lanes: Record<string, PathwayNode[]> = {};
  for (const n of nodes) {
    const lane = lanes[n.type] || (lanes[n.type] = []);
    lane.push(n);
  }
  const laneWidth = width / Math.max(1, laneOrder.length);
  const result: PathwayNode[] = [];
  for (let li = 0; li < laneOrder.length; li++) {
    const laneType = laneOrder[li];
    const lane = lanes[laneType] || [];
    const laneX = laneWidth * (li + 0.5);
    const verticalSpacing = Math.max(60, height / Math.max(1, lane.length + 1));
    for (let ni = 0; ni < lane.length; ni++) {
      const n = lane[ni];
      const y = verticalSpacing * (ni + 1);
      result.push({ ...n, x: typeof n.x === 'number' ? n.x : laneX, y: typeof n.y === 'number' ? n.y : y });
    }
  }
  return result;
}

export function PathwayViz({ pathwayData: inputPathwayData, className = '' }: PathwayVizProps) {
  // FE-032: pathwayData is now a prop. Default to empty so the component
  // renders an honest empty state when the caller hasn't fetched data yet.
  const pathwayData: PathwayData = inputPathwayData ?? { nodes: [], edges: [] };

  const [hoveredNode, setHoveredNode] = useState<string | null>(null);
  const [selectedNode, setSelectedNode] = useState<string | null>(null);

  // FE-032: Pagination — show MAX_NODES per page.
  const totalPages = Math.max(1, Math.ceil(pathwayData.nodes.length / MAX_NODES));
  const [page, setPage] = useState(0);
  const currentPage = Math.min(page, totalPages - 1);
  const pageNodes = useMemo(
    () => pathwayData.nodes.slice(currentPage * MAX_NODES, (currentPage + 1) * MAX_NODES),
    [pathwayData.nodes, currentPage]
  );
  const pageNodeIds = useMemo(() => new Set(pageNodes.map((n) => n.id)), [pageNodes]);
  const pageEdges = useMemo(
    () => pathwayData.edges.filter((e) => pageNodeIds.has(e.source) && pageNodeIds.has(e.target)),
    [pathwayData.edges, pageNodeIds]
  );

  // Assign default positions to nodes missing x/y.
  const positionedNodes = useMemo(
    () => assignDefaultPositions(pageNodes, 750, 360),
    [pageNodes]
  );
  const nodeMap = useMemo(() => new Map(positionedNodes.map((n) => [n.id, n])), [positionedNodes]);

  const connectedEdges = selectedNode
    ? pageEdges.filter((e) => e.source === selectedNode || e.target === selectedNode)
    : [];
  const connectedNodes = new Set<string>();
  connectedEdges.forEach((e) => { connectedNodes.add(e.source); connectedNodes.add(e.target); });

  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = 750;
    const height = 360;
    const dpr = window.devicePixelRatio || 1;
    if (canvas.width !== width * dpr || canvas.height !== height * dpr) {
      canvas.width = width * dpr;
      canvas.height = height * dpr;
      canvas.style.width = '100%';
      canvas.style.height = `${height}px`;
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);

    // Background.
    ctx.fillStyle = 'rgba(0, 0, 0, 0)';
    ctx.fillRect(0, 0, width, height);

    // Draw edges with arrow markers.
    for (const edge of pageEdges) {
      const source = nodeMap.get(edge.source);
      const target = nodeMap.get(edge.target);
      if (!source || !target) continue;

      const sx = source.x ?? 0;
      const sy = source.y ?? 0;
      const tx = target.x ?? 0;
      const ty = target.y ?? 0;

      const isHighlighted = selectedNode ? connectedNodes.has(edge.source) && connectedNodes.has(edge.target) : true;
      const isActivation = edge.type === 'activation';
      const isBinding = edge.type === 'binding';
      const dx = tx - sx;
      const dy = ty - sy;
      const angle = Math.atan2(dy, dx);
      const offset = 24;
      const x1 = sx + Math.cos(angle) * offset;
      const y1 = sy + Math.sin(angle) * offset;
      const x2 = tx - Math.cos(angle) * offset;
      const y2 = ty - Math.sin(angle) * offset;

      ctx.globalAlpha = isHighlighted ? 1 : 0.2;
      ctx.lineWidth = 1.5;

      if (isBinding) {
        ctx.strokeStyle = '#5B4FCF';
        ctx.setLineDash([4, 3]);
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
        ctx.setLineDash([]);
      } else {
        ctx.strokeStyle = isActivation ? '#1D9E75' : '#C0392B';
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();

        // Arrow head — activation: filled triangle; inhibition: flat line.
        const arrowSize = 8;
        const ax = x2;
        const ay = y2;
        if (isActivation) {
          ctx.fillStyle = '#1D9E75';
          ctx.beginPath();
          ctx.moveTo(ax, ay);
          ctx.lineTo(ax - arrowSize * Math.cos(angle - 0.4), ay - arrowSize * Math.sin(angle - 0.4));
          ctx.lineTo(ax - arrowSize * Math.cos(angle + 0.4), ay - arrowSize * Math.sin(angle + 0.4));
          ctx.closePath();
          ctx.fill();
        } else {
          // Inhibition: perpendicular line at the tip.
          ctx.strokeStyle = '#C0392B';
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.moveTo(ax - 4 * Math.cos(angle - Math.PI / 2), ay - 4 * Math.sin(angle - Math.PI / 2));
          ctx.lineTo(ax + 4 * Math.cos(angle - Math.PI / 2), ay + 4 * Math.sin(angle - Math.PI / 2));
          ctx.stroke();
          ctx.lineWidth = 1.5;
        }
      }

      // Edge label (edge.type).
      ctx.globalAlpha = isHighlighted ? 1 : 0.2;
      ctx.fillStyle = 'rgba(120, 120, 130, 1)';
      ctx.font = '8px ui-sans-serif, system-ui, sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(edge.type || edge.label || '', (x1 + x2) / 2, (y1 + y2) / 2 - 6);
      ctx.globalAlpha = 1;
    }

    // Draw nodes.
    for (const node of positionedNodes) {
      const style = typeStyles[node.type] || DEFAULT_STYLE;
      const isSelected = selectedNode === node.id;
      const isConnected = connectedNodes.has(node.id);
      const isActive = !selectedNode || isConnected || isSelected;
      const isHovered = hoveredNode === node.id;
      const x = node.x ?? 0;
      const y = node.y ?? 0;

      ctx.globalAlpha = isActive ? 1 : 0.3;

      // Background glow for selected/hovered.
      if (isSelected || isHovered) {
        ctx.beginPath();
        ctx.arc(x, y, 24, 0, 2 * Math.PI);
        ctx.fillStyle = style.fill;
        ctx.globalAlpha = 0.1;
        ctx.fill();
        ctx.globalAlpha = isActive ? 1 : 0.3;
      }

      // Node shape — drug = rect, everything else = circle.
      ctx.fillStyle = style.fill;
      ctx.globalAlpha = isSelected ? 0.3 : 0.15;
      if (node.type === 'drug') {
        ctx.fillRect(x - 24, y - 12, 48, 24);
      } else {
        ctx.beginPath();
        ctx.arc(x, y, 20, 0, 2 * Math.PI);
        ctx.fill();
      }

      ctx.globalAlpha = isActive ? 1 : 0.3;
      ctx.strokeStyle = style.stroke;
      ctx.lineWidth = isSelected ? 2.5 : 1.5;
      if (node.type === 'drug') {
        ctx.strokeRect(x - 24, y - 12, 48, 24);
      } else {
        ctx.beginPath();
        ctx.arc(x, y, 20, 0, 2 * Math.PI);
        ctx.stroke();
      }

      // Label.
      ctx.globalAlpha = isActive ? 1 : 0.3;
      ctx.fillStyle = 'rgba(30, 30, 40, 1)';
      ctx.font = '10px ui-sans-serif, system-ui, sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(node.label, x, y + 4);
      ctx.globalAlpha = 1;
    }
  }, [positionedNodes, pageEdges, nodeMap, selectedNode, connectedNodes, hoveredNode]);

  /**
   * Hit-test for hover/click. O(n) where n = nodes on the current page.
   */
  const hitTest = (clientX: number, clientY: number): string | null => {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    // The canvas displays at width 750 but is rendered responsively. Convert
    // clientX/Y to canvas-space (750 x 360) using the bounding rect.
    const scaleX = 750 / rect.width;
    const scaleY = 360 / rect.height;
    const cx = (clientX - rect.left) * scaleX;
    const cy = (clientY - rect.top) * scaleY;
    for (let i = positionedNodes.length - 1; i >= 0; i--) {
      const node = positionedNodes[i];
      const nx = node.x ?? 0;
      const ny = node.y ?? 0;
      if (node.type === 'drug') {
        if (Math.abs(cx - nx) <= 24 && Math.abs(cy - ny) <= 12) return node.id;
      } else {
        const dx = cx - nx;
        const dy = cy - ny;
        if (dx * dx + dy * dy <= 20 * 20) return node.id;
      }
    }
    return null;
  };

  const overCap = pathwayData.nodes.length > MAX_NODES;
  const pageStart = currentPage * MAX_NODES + 1;
  const pageEnd = Math.min((currentPage + 1) * MAX_NODES, pathwayData.nodes.length);

  return (
    <div className={`relative ${className}`}>
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-foreground">{pathwayData.name || 'Pathway Diagram'}</h3>
        <div className="flex gap-2">
          {Object.entries(typeStyles).slice(0, 5).map(([type, style]) => (
            <div key={type} className="flex items-center gap-1">
              <span
                className="h-2 w-2 rounded-full"
                style={{ backgroundColor: style.fill }}
              />
              <span className="text-[10px] text-muted-foreground">{style.label}</span>
            </div>
          ))}
        </div>
      </div>

      {/* FE-032: Pagination — only shown when input exceeds the 500-node cap. */}
      {overCap && (
        <div className="flex items-center gap-2 mb-2 bg-amber-50 border border-amber-300 rounded-lg p-2">
          <AlertTriangle className="h-4 w-4 text-amber-600" />
          <span className="text-xs text-foreground">
            Showing <span className="font-semibold">{pageStart}–{pageEnd}</span> of{' '}
            <span className="font-semibold">{pathwayData.nodes.length}</span> nodes
          </span>
          <Button
            variant="outline"
            size="sm"
            className="h-7 w-7 p-0 ml-auto"
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

      {pathwayData.nodes.length === 0 ? (
        <div className="bg-background border border-border rounded-lg flex items-center justify-center" style={{ height: 360 }}>
          <div className="text-center text-muted-foreground">
            <p className="text-sm font-medium">No pathway data</p>
            <p className="text-xs mt-1 max-w-md">
              Select a drug-disease hypothesis to load its pathway from the
              Phase 2 Neo4j knowledge graph. Pathway data is fetched from
              /api/knowledge-graph.
            </p>
          </div>
        </div>
      ) : (
        <canvas
          ref={canvasRef}
          onClick={(e) => {
            const hitId = hitTest(e.clientX, e.clientY);
            setSelectedNode(hitId && hitId === selectedNode ? null : hitId);
          }}
          onMouseMove={(e) => {
            const hitId = hitTest(e.clientX, e.clientY);
            setHoveredNode(hitId);
          }}
          onMouseLeave={() => setHoveredNode(null)}
          className="bg-background border border-border rounded-lg cursor-pointer"
          style={{ width: '100%', height: 360 }}
          aria-label="Biological pathway diagram. Click a node to highlight its connections."
        />
      )}

      {/* Selected node detail */}
      {selectedNode && nodeMap.has(selectedNode) && (
        <div className="mt-3 p-3 bg-muted/50 rounded-lg border border-border">
          <div className="flex items-center gap-2">
            <span className="font-semibold text-sm">{nodeMap.get(selectedNode)?.label}</span>
            <Badge variant="secondary" className="text-xs">
              {(typeStyles[nodeMap.get(selectedNode)?.type ?? 'effector'] || DEFAULT_STYLE).label}
            </Badge>
          </div>
          <p className="text-xs text-muted-foreground mt-1">
            {connectedEdges.length} connection{connectedEdges.length !== 1 ? 's' : ''} in this pathway
          </p>
        </div>
      )}
    </div>
  );
}
