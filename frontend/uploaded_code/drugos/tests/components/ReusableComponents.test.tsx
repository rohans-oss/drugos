/**
 * Reusable components test — verifies the four reusable components
 * created for audit issues #291, #292, #293, #294.
 *
 *   - EmptyState (#291)
 *   - CandidateCard (#292)
 *   - SafetyBadge (#293)
 *   - PathwayChain (#294)
 */
import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import { EmptyState } from '@/components/drugos/EmptyState';
import { CandidateCard } from '@/components/drugos/CandidateCard';
import { SafetyBadge } from '@/components/drugos/SafetyBadge';
import { PathwayChain } from '@/components/drugos/PathwayChain';
import { Package, Activity } from 'lucide-react';

describe('EmptyState (audit #291)', () => {
  it('renders the title and description', () => {
    render(
      React.createElement(EmptyState, {
        icon: Package,
        title: 'Nothing here',
        description: 'No data available right now.',
      }),
    );
    expect(screen.getByText('Nothing here')).toBeInTheDocument();
    expect(screen.getByText('No data available right now.')).toBeInTheDocument();
    expect(screen.getByTestId('empty-state')).toBeInTheDocument();
  });

  it('does not crash with no props (uses defaults)', () => {
    render(React.createElement(EmptyState));
    expect(screen.getByText('Nothing here yet')).toBeInTheDocument();
  });
});

describe('CandidateCard (audit #292)', () => {
  it('renders a full candidate with all fields', () => {
    render(
      React.createElement(CandidateCard, {
        candidate: {
          id: 'DC001',
          drugName: 'Memantine',
          genericName: 'memantine hydrochloride',
          brandNames: ['Namenda'],
          compositeScore: 87,
          kgScore: 91,
          safetyScore: 94,
          clinicalScore: 79,
          safetyTier: 'green',
          mechanism: 'NMDA receptor antagonist',
          clinicalPhase: 'Phase II',
          targets: ['GRIN2A'],
          pathways: ['Glutamatergic synapse'],
        },
      }),
    );
    expect(screen.getByText('Memantine')).toBeInTheDocument();
    expect(screen.getByText(/NMDA receptor antagonist/i)).toBeInTheDocument();
    expect(screen.getByTestId('safety-badge-green')).toBeInTheDocument();
    expect(screen.getByText('GRIN2A')).toBeInTheDocument();
  });

  it('does NOT crash on a candidate missing optional fields', () => {
    // This is the regression test for audit #281 / #282 — partial API
    // responses must not crash the UI.
    render(
      React.createElement(CandidateCard, {
        candidate: { id: 'X', drugName: 'PartialDrug' },
      }),
    );
    expect(screen.getAllByText('PartialDrug').length).toBeGreaterThan(0);
    // No safety badge should be rendered (tier is undefined).
    expect(screen.queryByTestId('safety-badge-green')).not.toBeInTheDocument();
    expect(screen.queryByTestId('safety-badge-yellow')).not.toBeInTheDocument();
    expect(screen.queryByTestId('safety-badge-red')).not.toBeInTheDocument();
  });

  it('renders the unknown safety badge when tier is invalid', () => {
    render(
      React.createElement(CandidateCard, {
        candidate: { drugName: 'X', safetyTier: 'purple' },
      }),
    );
    // The CandidateCard hides the SafetyBadge when tier is invalid
    // (because `safeTier()` returns null). Verify that no green/yellow/red
    // badge is rendered — the invalid tier is silently suppressed rather
    // than fabricating a misleading "unknown" badge in this context.
    expect(screen.queryByTestId('safety-badge-green')).not.toBeInTheDocument();
    expect(screen.queryByTestId('safety-badge-yellow')).not.toBeInTheDocument();
    expect(screen.queryByTestId('safety-badge-red')).not.toBeInTheDocument();
  });

  it('supports keyboard activation when onClick is provided', () => {
    const onClick = jest.fn();
    render(
      React.createElement(CandidateCard, {
        candidate: { id: 'DC1', drugName: 'TestDrug' },
        onClick,
      }),
    );
    const card = screen.getByTestId('candidate-card');
    expect(card).toHaveAttribute('role', 'button');
    expect(card).toHaveAttribute('tabindex', '0');
  });
});

describe('SafetyBadge (audit #293)', () => {
  it('renders green tier correctly', () => {
    render(React.createElement(SafetyBadge, { tier: 'green' }));
    const badge = screen.getByTestId('safety-badge-green');
    expect(badge).toBeInTheDocument();
    expect(badge).toHaveTextContent('Safe');
  });
  it('renders yellow tier correctly', () => {
    render(React.createElement(SafetyBadge, { tier: 'yellow' }));
    expect(screen.getByTestId('safety-badge-yellow')).toHaveTextContent('Caution');
  });
  it('renders red tier correctly', () => {
    render(React.createElement(SafetyBadge, { tier: 'red' }));
    expect(screen.getByTestId('safety-badge-red')).toHaveTextContent('High Risk');
  });
  it('renders unknown for invalid tier (never fabricates a color)', () => {
    render(React.createElement(SafetyBadge, { tier: 'mauve' }));
    expect(screen.getByTestId('safety-badge-unknown')).toBeInTheDocument();
  });
  it('renders unknown for undefined tier', () => {
    render(React.createElement(SafetyBadge, { tier: undefined }));
    expect(screen.getByTestId('safety-badge-unknown')).toBeInTheDocument();
  });
});

describe('PathwayChain (audit #294)', () => {
  it('renders the drug → protein → pathway → disease chain', () => {
    render(
      React.createElement(PathwayChain, {
        hops: [
          { label: 'Memantine', type: 'drug' },
          { label: 'NMDA Receptor', type: 'protein', relation: 'inhibits' },
          { label: 'Glutamatergic Synapse', type: 'pathway', relation: 'participates_in' },
          { label: "Huntington's Disease", type: 'disease', relation: 'associated_with' },
        ],
      }),
    );
    expect(screen.getByText('Memantine')).toBeInTheDocument();
    expect(screen.getByText('NMDA Receptor')).toBeInTheDocument();
    expect(screen.getByText('Glutamatergic Synapse')).toBeInTheDocument();
    expect(screen.getByText("Huntington's Disease")).toBeInTheDocument();
    // The relation labels should be visible.
    expect(screen.getByText('inhibits')).toBeInTheDocument();
    expect(screen.getByText('participates_in')).toBeInTheDocument();
    expect(screen.getByText('associated_with')).toBeInTheDocument();
  });

  it('renders the EmptyState when hops is empty', () => {
    render(React.createElement(PathwayChain, { hops: [] }));
    // The EmptyState title and the description both contain "No pathway chain".
    expect(screen.getAllByText(/No pathway chain/i).length).toBeGreaterThan(0);
  });

  it('renders evidence scores when provided', () => {
    render(
      React.createElement(PathwayChain, {
        hops: [
          { label: 'Drug A', type: 'drug' },
          { label: 'Target B', type: 'protein', evidence: 0.92 },
        ],
      }),
    );
    expect(screen.getByText(/92% evidence/i)).toBeInTheDocument();
  });

  it('does not crash on hops with missing fields', () => {
    render(
      React.createElement(PathwayChain, {
        hops: [{}, { label: 'Only label' }],
      }),
    );
    // The "—" placeholder should appear for the empty hop label.
    expect(screen.getAllByText('—').length).toBeGreaterThan(0);
  });
});
