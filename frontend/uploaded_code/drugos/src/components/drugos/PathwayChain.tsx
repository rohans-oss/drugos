'use client';

/**
 * PathwayChain — visualization for the drug → protein → pathway → disease
 * reasoning chain that explains a repurposing prediction.
 *
 * Root fix for audit issue #294: the existing `pathway-viz.tsx` rendered
 * an arbitrary node-link diagram, but the Phase 2 build plan mandates
 * that EVERY repurposing hypothesis must be backed by a transparent,
 * auditable 4-hop chain: drug → target protein → biological pathway →
 * disease. This component renders EXACTLY that chain — not a generic
 * graph — so a pharma researcher can see at a glance WHY the model
 * believes a drug treats a disease.
 *
 * Design:
 *   - Horizontal flow on desktop, vertical on mobile.
 *   - Each hop shows: entity label, entity type chip, evidence score
 *     (when provided), and a relation arrow with the relation label.
 *   - Empty hops render as "—" placeholders so partial API data
 *     (e.g. a drug with no known target protein) never crashes the UI.
 *   - Optional `onHopClick` lets the parent open a detail panel.
 *
 * Acceptance criteria: renders for any chain with 0–6 hops, never throws
 * on undefined fields, looks identical to the Figma mock.
 */
import { ChevronRight, Circle, ArrowRight } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { cn } from '@/lib/utils';
import { EmptyState } from '@/components/drugos/EmptyState';
import { Network } from 'lucide-react';

export type PathwayEntityType =
  | 'drug'
  | 'protein'
  | 'gene'
  | 'pathway'
  | 'disease'
  | 'outcome'
  | string;

export interface PathwayHop {
  /** Entity id (optional). */
  id?: string;
  /** Display label (e.g. "Memantine"). */
  label?: string;
  /** Entity type — drives the chip color. */
  type?: PathwayEntityType;
  /** Optional sub-label (e.g. gene symbol,DrugBank ID). */
  subLabel?: string;
  /** Optional evidence score 0..1 for THIS hop's incoming edge. */
  evidence?: number;
  /** Relation label on the edge leading INTO this hop (e.g. "inhibits"). */
  relation?: string;
}

export interface PathwayChainProps {
  hops: PathwayHop[];
  /** Click handler for a hop. */
  onHopClick?: (hop: PathwayHop, index: number) => void;
  /** Optional title shown above the chain. */
  title?: string;
  /** Optional description shown below the title. */
  description?: string;
  /** Show evidence scores. Default true. */
  showEvidence?: boolean;
  /** Empty-state message when hops is []. */
  emptyMessage?: string;
  className?: string;
}

const TYPE_COLORS: Record<string, string> = {
  drug: '#1D9E75',
  protein: '#8B5CF6',
  gene: '#5B4FCF',
  pathway: '#D4853A',
  disease: '#C0392B',
  outcome: '#0EA5E9',
};

const TYPE_LABELS: Record<string, string> = {
  drug: 'Drug',
  protein: 'Protein',
  gene: 'Gene',
  pathway: 'Pathway',
  disease: 'Disease',
  outcome: 'Outcome',
};

function evidenceColor(e: number): string {
  if (e >= 0.85) return '#1D9E75';
  if (e >= 0.6) return '#D4853A';
  return '#C0392B';
}

function HopNode({
  hop,
  index,
  onClick,
  showEvidence,
}: {
  hop: PathwayHop;
  index: number;
  onClick?: (h: PathwayHop, i: number) => void;
  showEvidence: boolean;
}) {
  const color = (hop.type && TYPE_COLORS[hop.type]) || '#64748b';
  const typeLabel = (hop.type && TYPE_LABELS[hop.type]) || (hop.type ?? 'Entity');
  const interactive = typeof onClick === 'function';

  const handleKey = (e: React.KeyboardEvent) => {
    if (!interactive) return;
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      onClick!(hop, index);
    }
  };

  return (
    <div
      role={interactive ? 'button' : undefined}
      tabIndex={interactive ? 0 : undefined}
      onClick={interactive ? () => onClick!(hop, index) : undefined}
      onKeyDown={handleKey}
      data-testid={`pathway-hop-${index}`}
      data-hop-type={hop.type ?? 'unknown'}
      className={cn(
        'flex flex-col items-center text-center px-3 py-3 rounded-lg border bg-background min-w-[120px] max-w-[180px]',
        interactive && 'cursor-pointer hover:shadow-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40',
      )}
      style={{ borderColor: `${color}40` }}
    >
      <Badge
        variant="outline"
        className="text-[9px] mb-1.5 px-1.5 py-0 font-medium"
        style={{ color, borderColor: `${color}50`, backgroundColor: `${color}10` }}
      >
        {typeLabel}
      </Badge>
      <span className="text-sm font-medium text-foreground line-clamp-2 leading-tight">
        {hop.label ?? '—'}
      </span>
      {hop.subLabel && (
        <span className="text-[10px] text-muted-foreground mt-0.5 font-mono truncate max-w-full">
          {hop.subLabel}
        </span>
      )}
      {showEvidence && typeof hop.evidence === 'number' && (
        <span
          className="text-[10px] font-semibold mt-1.5 tabular-nums"
          style={{ color: evidenceColor(hop.evidence) }}
        >
          {(hop.evidence * 100).toFixed(0)}% evidence
        </span>
      )}
    </div>
  );
}

function HopArrow({ hop }: { hop: PathwayHop }) {
  return (
    <div
      className="flex flex-col items-center justify-center text-muted-foreground shrink-0 min-w-[60px] py-1"
      data-testid="pathway-arrow"
    >
      <ArrowRight className="h-4 w-4 hidden sm:block" aria-hidden="true" />
      <ChevronRight className="h-4 w-4 sm:hidden" aria-hidden="true" />
      {hop.relation && (
        <span className="text-[9px] uppercase tracking-wide mt-0.5 text-center max-w-[80px] truncate">
          {hop.relation}
        </span>
      )}
    </div>
  );
}

export function PathwayChain({
  hops,
  onHopClick,
  title,
  description,
  showEvidence = true,
  emptyMessage = 'No pathway chain available for this hypothesis.',
  className,
}: PathwayChainProps) {
  const safeHops = Array.isArray(hops) ? hops.filter(Boolean) : [];

  return (
    <div className={cn('w-full', className)} data-testid="pathway-chain">
      {title && <h3 className="text-base font-semibold mb-1">{title}</h3>}
      {description && <p className="text-xs text-muted-foreground mb-3">{description}</p>}

      {safeHops.length === 0 ? (
        <EmptyState
          icon={Network}
          title="No pathway chain"
          description={emptyMessage}
          size="sm"
        />
      ) : (
        <div className="flex flex-wrap items-stretch gap-y-3 sm:flex-nowrap sm:overflow-x-auto sm:pb-2">
          {safeHops.map((hop, i) => (
            <div key={hop.id ?? i} className="flex items-center shrink-0">
              <HopNode
                hop={hop}
                index={i}
                onClick={onHopClick}
                showEvidence={showEvidence}
              />
              {i < safeHops.length - 1 && <HopArrow hop={safeHops[i + 1]} />}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default PathwayChain;
