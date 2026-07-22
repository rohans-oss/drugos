'use client';

/**
 * CandidateCard — reusable card rendering a single drug repurposing
 * candidate. Used by CandidateDetail (issue #292) and may be reused by
 * any list/grid that needs a compact candidate preview.
 *
 * Root fix for audit issue #292: previously the candidate detail screen
 * inlined its own bespoke card markup; that markup dereferenced
 * `candidate.drugName`, `candidate.compositeScore`, etc. without any
 * null/undefined guards, so any candidate coming from the real RL API
 * that was missing a field would crash the screen. This component:
 *
 *   1. Accepts a宽松 typed `candidate` prop (anything with optional
 *      fields) so partial API responses don't crash the renderer.
 *   2. Uses optional chaining + `?? 'N/A'` for every display field.
 *   3. Renders a SafetyBadge only when `safetyTier` is one of the
 *      known enum values; otherwise hides it.
 *   4. Exposes an optional `onClick` for navigation.
 */
import { type LucideIcon, Activity } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { SafetyBadge } from '@/components/drugos/safety-badge';
import { cn } from '@/lib/utils';

export type CandidateTier = 'green' | 'yellow' | 'red' | string | undefined;

export interface CandidateLike {
  id?: string;
  drugName?: string;
  genericName?: string;
  brandNames?: string[];
  compositeScore?: number;
  kgScore?: number;
  molSimScore?: number;
  safetyScore?: number;
  clinicalScore?: number;
  safetyTier?: CandidateTier;
  mechanism?: string;
  clinicalPhase?: string;
  ipStatus?: string;
  diseaseId?: string;
  diseaseName?: string;
  targets?: string[];
  pathways?: string[];
  [k: string]: unknown;
}

export interface CandidateCardProps {
  candidate: CandidateLike;
  /** Optional icon for the card header. */
  icon?: LucideIcon;
  /** Click handler — when set, card is keyboard-focusable and role=button. */
  onClick?: (c: CandidateLike) => void;
  /** Compact mode: hides mechanism and brand names. */
  compact?: boolean;
  /** Show the safety badge (default true). */
  showSafety?: boolean;
  /** Show the disease row (default true). */
  showDisease?: boolean;
  /** Extra class names. */
  className?: string;
}

function safeTier(t: CandidateTier): 'green' | 'yellow' | 'red' | null {
  if (t === 'green' || t === 'yellow' || t === 'red') return t;
  return null;
}

function scoreColor(s: number | undefined): string {
  if (typeof s !== 'number' || Number.isNaN(s)) return '#94a3b8';
  if (s >= 80) return '#1D9E75';
  if (s >= 60) return '#D4853A';
  return '#C0392B';
}

function ScoreRow({ label, value }: { label: string; value: number | undefined }) {
  const v = typeof value === 'number' && !Number.isNaN(value) ? value : 0;
  const color = scoreColor(value);
  return (
    <div>
      <div className="flex justify-between text-xs mb-0.5">
        <span className="text-muted-foreground">{label}</span>
        <span className="font-medium tabular-nums">{value ?? '—'}</span>
      </div>
      <div className="w-full bg-slate-100 rounded-full h-1.5 overflow-hidden" aria-hidden="true">
        <div
          className="h-full rounded-full"
          style={{ width: `${Math.min(100, Math.max(0, v))}%`, backgroundColor: color }}
        />
      </div>
    </div>
  );
}

export function CandidateCard({
  candidate,
  icon: Icon = Activity,
  onClick,
  compact = false,
  showSafety = true,
  showDisease = true,
  className,
}: CandidateCardProps) {
  if (!candidate || typeof candidate !== 'object') {
    return null;
  }
  const tier = safeTier(candidate.safetyTier);
  const brandNames = Array.isArray(candidate.brandNames) ? candidate.brandNames : [];
  const targets = Array.isArray(candidate.targets) ? candidate.targets : [];
  const pathways = Array.isArray(candidate.pathways) ? candidate.pathways : [];
  const diseaseName =
    (candidate.diseaseName as string | undefined) ??
    (typeof candidate.diseaseId === 'string' ? candidate.diseaseId : undefined);

  const interactive = typeof onClick === 'function';

  const handleKey = (e: React.KeyboardEvent) => {
    if (!interactive) return;
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      onClick!(candidate);
    }
  };

  return (
    <Card
      role={interactive ? 'button' : undefined}
      tabIndex={interactive ? 0 : undefined}
      onClick={interactive ? () => onClick!(candidate) : undefined}
      onKeyDown={handleKey}
      data-testid="candidate-card"
      data-candidate-id={candidate.id ?? 'unknown'}
      className={cn(
        'transition-shadow',
        interactive && 'cursor-pointer hover:shadow-md focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40',
        className,
      )}
    >
      <CardHeader className="pb-2">
        <div className="flex items-start justify-between gap-2">
          <div className="flex items-center gap-2 min-w-0">
            <Icon className="h-4 w-4 text-muted-foreground shrink-0" aria-hidden="true" />
            <CardTitle className="text-base truncate">
              {candidate.drugName ?? 'Unknown drug'}
            </CardTitle>
          </div>
          <div className="flex items-center gap-2 shrink-0">
            {tier && showSafety && <SafetyBadge tier={tier} />}
            {candidate.clinicalPhase && (
              <Badge variant="outline" className="text-xs">
                {candidate.clinicalPhase}
              </Badge>
            )}
          </div>
        </div>
        <p className="text-xs text-muted-foreground mt-1 truncate">
          {candidate.genericName ?? candidate.drugName ?? 'Unknown'}
          {brandNames.length > 0 && !compact && (
            <> · {brandNames.slice(0, 3).join(', ')}</>
          )}
        </p>
      </CardHeader>
      <CardContent className="space-y-3">
        {showDisease && diseaseName && (
          <div className="text-xs text-muted-foreground">
            <span className="font-medium">Disease:</span>{' '}
            <span className="text-foreground">{diseaseName}</span>
          </div>
        )}

        <div className="grid grid-cols-2 gap-x-4 gap-y-2">
          <ScoreRow label="Composite" value={candidate.compositeScore} />
          <ScoreRow label="KG Score" value={candidate.kgScore} />
          <ScoreRow label="Safety" value={candidate.safetyScore} />
          <ScoreRow label="Clinical" value={candidate.clinicalScore} />
        </div>

        {!compact && candidate.mechanism && (
          <p className="text-xs text-muted-foreground line-clamp-3">
            {candidate.mechanism}
          </p>
        )}

        {(targets.length > 0 || pathways.length > 0) && !compact && (
          <div className="flex flex-wrap gap-1.5 pt-1">
            {targets.slice(0, 4).map((t) => (
              <Badge key={`t-${t}`} variant="secondary" className="text-[10px] font-mono">
                {t}
              </Badge>
            ))}
            {pathways.slice(0, 2).map((p) => (
              <Badge key={`p-${p}`} variant="outline" className="text-[10px]">
                {p}
              </Badge>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export default CandidateCard;
