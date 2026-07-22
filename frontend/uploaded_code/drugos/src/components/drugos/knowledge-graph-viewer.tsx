'use client';

/**
 * Backward-compatibility wrapper around the new KnowledgeGraphExplorer.
 *
 * Root fix for audit issue #283: the old `knowledge-graph-viewer.tsx`
 * imported `KnowledgeGraphNode`, `KnowledgeGraphEdge`,
 * `knowledgeGraphNodes`, and `knowledgeGraphEdges` from `@/lib/mock-data`
 * — NONE of which exist (the real exports are `GraphNode`, `GraphEdge`,
 * `graphNodes`, `graphEdges`). The file was broken at compile time, and
 * every real node returned `null` from `positions.get(id)` because the
 * positions Map was built from an empty default array.
 *
 * The canonical, production-grade implementation now lives in
 * `KnowledgeGraphExplorer.tsx` (Canvas2D, side panel, edge filtering).
 * This file re-exports it under the legacy name so existing imports
 * keep working.
 */
export { KnowledgeGraphExplorer as KnowledgeGraphViewer } from './KnowledgeGraphExplorer';
export { KnowledgeGraphExplorer as default } from './KnowledgeGraphExplorer';
export type { KnowledgeGraphExplorerProps as KnowledgeGraphViewerProps } from './KnowledgeGraphExplorer';
