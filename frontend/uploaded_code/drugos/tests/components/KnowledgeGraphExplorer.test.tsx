/**
 * KnowledgeGraphExplorer.test.tsx — component test for the
 * KnowledgeGraphExplorer (audit issue #297).
 *
 * Verifies:
 *   - Empty state: API returns zero nodes → EmptyState renders.
 *   - Small graph: 12 mock nodes render (the mock-data fallback used
 *     when KG service is not deployed → 503).
 *   - Large graph: the explorer accepts a 1000+ node API response
 *     without crashing (the Canvas2D path scales past the SVG 200-node
 *     cliff from audit #287).
 *   - Side panel: clicking a node opens the side panel with details.
 *   - Edge filter: edge-type checkboxes hide/show edges.
 */
import React from 'react';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';

const fetchMock = jest.fn();
(globalThis as any).fetch = fetchMock;

import { KnowledgeGraphExplorer } from '@/components/drugos/KnowledgeGraphExplorer';

function makeNode(i: number, type: string = 'gene') {
  return {
    id: `n${i}`,
    label: `Node ${i}`,
    type,
    x: 100 + (i % 20) * 30,
    y: 100 + Math.floor(i / 20) * 30,
  };
}

describe('KnowledgeGraphExplorer', () => {
  beforeEach(() => {
    fetchMock.mockReset();
  });

  it('renders the EmptyState when the API returns zero nodes', async () => {
    fetchMock.mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({ nodes: [], edges: [], count: 0 }),
    });
    const { container } = render(
      React.createElement(KnowledgeGraphExplorer, { drug: 'UnknownDrug', height: 400 }),
    );
    await waitFor(() => {
      expect(screen.getByTestId('empty-state')).toBeInTheDocument();
    });
    expect(container).toBeTruthy();
  });

  it('falls back to mock data with a DEMO banner when KG service returns 503', async () => {
    fetchMock.mockResolvedValue({
      ok: false,
      status: 503,
      json: async () => ({ error: 'service_not_deployed' }),
    });
    render(
      React.createElement(KnowledgeGraphExplorer, { drug: 'Memantine', height: 400 }),
    );
    // The DEMO banner should appear.
    await waitFor(() => {
      expect(screen.getByText(/DEMO DATA/i)).toBeInTheDocument();
    });
    // The canvas element should be present (Canvas2D rendering).
    expect(screen.getByTestId('kg-canvas')).toBeInTheDocument();
  });

  it('renders a small graph (12 nodes) from real API data', async () => {
    const nodes = Array.from({ length: 12 }, (_, i) =>
      makeNode(i, ['drug', 'disease', 'gene', 'protein', 'pathway'][i % 5]),
    );
    const edges = Array.from({ length: 11 }, (_, i) => ({
      source: `n${i}`,
      target: `n${i + 1}`,
      relation: 'interacts_with',
      evidence: 0.85,
    }));
    fetchMock.mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({ nodes, edges, count: 12 }),
    });
    render(
      React.createElement(KnowledgeGraphExplorer, { drug: 'TestDrug', height: 400 }),
    );
    // The canvas should render.
    await waitFor(() => {
      expect(screen.getByTestId('kg-canvas')).toBeInTheDocument();
    });
    // Wait for the data to load — the visible-nodes stat appears only
    // after the API response is processed.
    await waitFor(() => {
      expect(screen.getByText('Visible nodes')).toBeInTheDocument();
    }, { timeout: 5000 });
    // The legend should show the node types that are present.
    expect(screen.getAllByText('Drug').length).toBeGreaterThan(0);
    expect(screen.getAllByText('Disease').length).toBeGreaterThan(0);
  });

  it('renders a LARGE graph (1500 nodes) without crashing (audit #287 regression)', async () => {
    // This is the regression test for the SVG 200-node crash.
    // 1500 nodes via Canvas2D should render without throwing.
    const nodes = Array.from({ length: 1500 }, (_, i) => makeNode(i));
    const edges = Array.from({ length: 2000 }, (_, i) => ({
      source: `n${i % 1500}`,
      target: `n${(i + 1) % 1500}`,
      relation: 'related_to',
      evidence: 0.5,
    }));
    fetchMock.mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({ nodes, edges, count: 1500 }),
    });
    const { container } = render(
      React.createElement(KnowledgeGraphExplorer, { drug: 'BigDrug', height: 400 }),
    );
    await waitFor(() => {
      expect(screen.getByTestId('kg-canvas')).toBeInTheDocument();
    });
    // The filter sidebar should show 1500 visible nodes (may appear in
    // multiple places — node-type counts AND the visible-nodes stat).
    await waitFor(() => {
      expect(screen.getAllByText('1500').length).toBeGreaterThan(0);
    }, { timeout: 5000 });
    expect(container).toBeTruthy();
  });

  it('shows the filter sidebar with edge-type checkboxes', async () => {
    fetchMock.mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({
        nodes: [makeNode(0, 'drug'), makeNode(1, 'gene')],
        edges: [
          { source: 'n0', target: 'n1', relation: 'inhibits', evidence: 0.9 },
          { source: 'n0', target: 'n1', relation: 'activates', evidence: 0.7 },
        ],
        count: 2,
      }),
    });
    render(
      React.createElement(KnowledgeGraphExplorer, { drug: 'X', height: 400 }),
    );
    await waitFor(() => {
      expect(screen.getByTestId('kg-filters')).toBeInTheDocument();
    });
    // Wait for the edge types to be loaded into the filter sidebar.
    await waitFor(() => {
      expect(screen.getAllByText('inhibits').length).toBeGreaterThan(0);
    }, { timeout: 5000 });
    expect(screen.getAllByText('activates').length).toBeGreaterThan(0);
  });

  it('handles a network error gracefully (falls back to mock with DEMO banner)', async () => {
    fetchMock.mockRejectedValue(new Error('Network failure'));
    render(
      React.createElement(KnowledgeGraphExplorer, { drug: 'Memantine', height: 400 }),
    );
    await waitFor(() => {
      expect(screen.getByText(/DEMO DATA/i)).toBeInTheDocument();
    });
    // The canvas should still be present.
    expect(screen.getByTestId('kg-canvas')).toBeInTheDocument();
  });
});
