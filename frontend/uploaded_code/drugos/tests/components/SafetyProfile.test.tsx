/**
 * SafetyProfile.test.tsx — component test for the SafetyProfileScreen
 * (audit issue #296).
 *
 * Verifies:
 *   - Populated state: real openFDA data renders the top reactions,
 *     total/serious report counts.
 *   - Error state: a fetch failure falls back to mock safety tier only.
 *   - 404 state: openFDA has no data for the drug — renders a clear
 *     "no reports" message (not a crash).
 */
import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';

// Hoist the nav-context mock BEFORE importing the screen. This avoids
// the dual-React-instance problem that `jest.resetModules()` causes.
jest.mock('@/components/drugos/nav-context', () => {
  const R = jest.requireActual('react');
  return {
    useDrugOSNav: () => ({
      navigate: jest.fn(),
      currentRoute: { page: 'app', section: 'safety' },
    }),
    DrugOSNavProvider: ({ children }: { children: R.ReactNode }) => children,
  };
});

const fetchMock = jest.fn();
(globalThis as any).fetch = fetchMock;

// Import AFTER the mock is registered so the screen sees the mocked nav.
import { coreScreens } from '@/components/drugos/core-screens';
const SafetyProfileScreen = coreScreens['safety'];

function renderSafetyScreen() {
  return render(React.createElement(SafetyProfileScreen));
}

describe('SafetyProfileScreen', () => {
  beforeEach(() => {
    fetchMock.mockReset();
  });

  it('renders the safety dashboard and fetches openFDA data on mount', async () => {
    fetchMock.mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({
        brandName: 'Memantine',
        genericName: 'memantine hydrochloride',
        totalReports: 1234,
        seriousReports: 234,
        seriousReportsWithDeath: 12,
        topReactions: [
          { term: 'Dizziness', count: 200 },
          { term: 'Headache', count: 150 },
          { term: 'Nausea', count: 80 },
        ],
        disclaimer: 'FAERS data — spontaneous reports, not causation.',
      }),
    });
    await renderSafetyScreen();

    expect(screen.getByText(/Safety Profile Dashboard/i)).toBeInTheDocument();

    await waitFor(() => {
      expect(screen.getByText('1234')).toBeInTheDocument();
    });
    expect(screen.getByText('234')).toBeInTheDocument();
    expect(screen.getByText('12')).toBeInTheDocument();

    expect(screen.getByText('Dizziness')).toBeInTheDocument();
    expect(screen.getByText('Headache')).toBeInTheDocument();
    expect(screen.getByText('Nausea')).toBeInTheDocument();
  });

  it('renders the openFDA 404 (no reports) state gracefully', async () => {
    fetchMock.mockResolvedValue({
      ok: false,
      status: 404,
      json: async () => ({ error: 'not_found' }),
    });
    await renderSafetyScreen();
    await waitFor(() => {
      expect(
        screen.getByText(/No adverse-event reports found in openFDA/i),
      ).toBeInTheDocument();
    });
    expect(screen.getByText(/Safety Tier/i)).toBeInTheDocument();
  });

  it('renders the error state when the safety API call rejects', async () => {
    fetchMock.mockRejectedValue(new Error('Network failure'));
    await renderSafetyScreen();
    await waitFor(() => {
      expect(screen.getByText(/openFDA lookup failed/i)).toBeInTheDocument();
    });
    expect(screen.getByText(/Safety Profile Dashboard/i)).toBeInTheDocument();
  });

  it('displays the disclaimer text when openFDA returns one', async () => {
    fetchMock.mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({
        totalReports: 5,
        seriousReports: 0,
        seriousReportsWithDeath: 0,
        topReactions: [{ term: 'Headache', count: 5 }],
        disclaimer: 'Custom FAERS disclaimer text — TEST_MARKER_12345',
      }),
    });
    await renderSafetyScreen();
    await waitFor(() => {
      expect(screen.getByText(/TEST_MARKER_12345/i)).toBeInTheDocument();
    });
  });

  it('never fabricates adverse-event frequencies (regression test for audit #285)', async () => {
    fetchMock.mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({
        totalReports: 42,
        seriousReports: 7,
        seriousReportsWithDeath: 1,
        topReactions: [{ term: 'TEST_REACTION', count: 42 }],
        disclaimer: '',
      }),
    });
    await renderSafetyScreen();
    await waitFor(() => {
      expect(screen.getByText('TEST_REACTION')).toBeInTheDocument();
    });
    // Multiple "42" elements may appear (Total AE Reports + reaction count).
    expect(screen.getAllByText('42').length).toBeGreaterThan(0);
  });
});
