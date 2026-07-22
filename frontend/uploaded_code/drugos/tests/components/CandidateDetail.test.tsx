/**
 * CandidateDetail.test.tsx — component test for the CandidateDetail
 * screen (audit issue #295).
 *
 * Verifies:
 *   - Empty state: no candidate found renders the EmptyState component.
 *   - Populated state: candidate data renders all key fields without
 *     crashing, even when the RL API is unavailable (503 → DEMO banner).
 *   - Error state: a fetch failure falls back to mock data + DEMO banner
 *     and does NOT crash the screen.
 */
import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';

// We use a mutable mock so each test can set a different route id.
const mockNav = {
  navigate: jest.fn(),
  currentRoute: { page: 'app', section: 'candidate', id: 'DC001' } as {
    page: string;
    section: string;
    id?: string;
  },
};

jest.mock('@/components/drugos/nav-context', () => {
  const R = jest.requireActual('react');
  return {
    useDrugOSNav: () => mockNav,
    DrugOSNavProvider: ({ children }: { children: R.ReactNode }) => children,
  };
});

const fetchMock = jest.fn();
(globalThis as any).fetch = fetchMock;

import { coreScreens } from '@/components/drugos/core-screens';
const CandidateDetailScreen = coreScreens['candidate'];

function renderCandidateScreen(candidateId: string = 'DC001') {
  mockNav.currentRoute = { page: 'app', section: 'candidate', id: candidateId };
  return render(React.createElement(CandidateDetailScreen));
}

describe('CandidateDetailScreen', () => {
  beforeEach(() => {
    fetchMock.mockReset();
  });

  it('renders the EmptyState when no candidate matches the route id', async () => {
    fetchMock.mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({ candidates: [] }),
    });
    renderCandidateScreen('DOES-NOT-EXIST');
    await waitFor(() => {
      expect(screen.getByText(/No candidate found/i)).toBeInTheDocument();
    });
  });

  it('renders candidate fields without crashing when the RL API returns 503 (DEMO mode)', async () => {
    fetchMock.mockResolvedValue({
      ok: false,
      status: 503,
      json: async () => ({ error: 'service_not_deployed' }),
    });
    renderCandidateScreen('DC001');
    await waitFor(() => {
      expect(screen.getByText(/DEMO DATA/i)).toBeInTheDocument();
    });
    expect(screen.getAllByText(/Memantine/i).length).toBeGreaterThan(0);
  });

  it('renders candidate data when the RL API returns a matching candidate', async () => {
    fetchMock.mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({
        candidates: [
          {
            drug: 'Memantine',
            disease: "Huntington's Disease",
            overall_score: 0.92,
            gnn_score: 0.88,
            safety_score: 0.95,
            clinical_score: 0.79,
            explanation: 'NMDA receptor antagonist — strong graph evidence.',
          },
        ],
      }),
    });
    renderCandidateScreen('DC001');
    await waitFor(() => {
      expect(screen.getAllByText(/Memantine/i).length).toBeGreaterThan(0);
    });
    await waitFor(() => {
      expect(screen.queryByText(/DEMO DATA/i)).not.toBeInTheDocument();
    }, { timeout: 5000 });
  });

  it('falls back to mock data and shows DEMO banner when fetch rejects', async () => {
    fetchMock.mockRejectedValue(new Error('Network down'));
    renderCandidateScreen('DC001');
    await waitFor(() => {
      expect(screen.getByText(/DEMO DATA/i)).toBeInTheDocument();
    }, { timeout: 5000 });
    await waitFor(() => {
      expect(screen.getByText(/RL API error/i)).toBeInTheDocument();
    }, { timeout: 5000 });
    expect(screen.getAllByText(/Memantine/i).length).toBeGreaterThan(0);
  });

  it('does not crash when the candidate has undefined optional fields', async () => {
    fetchMock.mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({
        candidates: [{ drug: 'Memantine' }],
      }),
    });
    renderCandidateScreen('DC001');
    await waitFor(() => {
      expect(screen.getAllByText(/Memantine/i).length).toBeGreaterThan(0);
    });
  });
});
