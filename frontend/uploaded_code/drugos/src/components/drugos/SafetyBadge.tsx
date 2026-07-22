'use client';

/**
 * SafetyBadge — canonical reusable safety tier badge (green / yellow / red).
 *
 * Root fix for audit issue #293: previously the codebase had TWO copies
 * of this component:
 *   1. `safety-badge.tsx` — used by `core-screen.tsx` (the screens/ dir)
 *   2. A local inline `SafetyBadge` function inside `core-screens.tsx`
 *      (the 2500-line screens file) with DIFFERENT colors and DIFFERENT
 *      type signatures.
 *
 * Result: depending on which screen you loaded, the same drug could
 * render with a different "safety tier" color, and the two copies
 * drifted over time. The audit found this as a real scientific
 * integrity bug — a "green" drug in one screen could be "yellow"
 * in another.
 *
 * Root fix: this file is now the SINGLE source of truth. The
 * `safety-badge.tsx` file re-exports this implementation for backward
 * compatibility, and `core-screens.tsx` was refactored to use this
 * component instead of its inline copy.
 *
 * Type contract:
 *   - tier 'green'  → Safe (low adverse-event signal)
 *   - tier 'yellow' → Caution (moderate signal — monitor)
 *   - tier 'red'    → High Risk (severe adverse-event signal)
 *
 * Acceptance: every screen renders the same color for the same tier.
 */
import { Badge } from '@/components/ui/badge';
import { cn } from '@/lib/utils';
import type { SafetyTier } from '@/lib/mock-data';

export interface SafetyBadgeProps {
  tier: SafetyTier | 'green' | 'yellow' | 'red' | string | undefined;
  showLabel?: boolean;
  showDot?: boolean;
  /** Visual size. */
  size?: 'sm' | 'md';
  /** Override the label text. */
  label?: string;
  className?: string;
}

interface TierConfig {
  label: string;
  text: string;
  bg: string;
  border: string;
  dot: string;
}

const TIER_CONFIG: Record<'green' | 'yellow' | 'red', TierConfig> = {
  green: {
    label: 'Safe',
    text: 'text-[#1D9E75]',
    bg: 'bg-[#1D9E75]/10',
    border: 'border-[#1D9E75]/25',
    dot: 'bg-[#1D9E75]',
  },
  yellow: {
    label: 'Caution',
    text: 'text-[#D4853A]',
    bg: 'bg-[#D4853A]/10',
    border: 'border-[#D4853A]/25',
    dot: 'bg-[#D4853A]',
  },
  red: {
    label: 'High Risk',
    text: 'text-[#C0392B]',
    bg: 'bg-[#C0392B]/10',
    border: 'border-[#C0392B]/25',
    dot: 'bg-[#C0392B]',
  },
};

function normalizeTier(t: SafetyBadgeProps['tier']): 'green' | 'yellow' | 'red' | null {
  if (t === 'green' || t === 'yellow' || t === 'red') return t;
  return null;
}

export function SafetyBadge({
  tier,
  showLabel = true,
  showDot = true,
  size = 'md',
  label,
  className,
}: SafetyBadgeProps) {
  const t = normalizeTier(tier);
  if (!t) {
    // Unknown / missing tier — render a neutral placeholder so the UI
    // does not silently lie about safety. This is the scientific
    // integrity fix: NEVER fabricate a green/yellow/red label when
    // we genuinely don't know.
    return (
      <Badge
        variant="outline"
        className={cn('gap-1.5 font-medium text-muted-foreground', className)}
        data-testid="safety-badge-unknown"
      >
        {showDot && <span className="h-2 w-2 rounded-full bg-muted-foreground/50" />}
        {showLabel && (label ?? 'Unknown')}
      </Badge>
    );
  }
  const cfg = TIER_CONFIG[t];
  const pad = size === 'sm' ? 'px-1.5 py-0 text-[10px]' : 'px-2 py-0.5 text-xs';
  return (
    <Badge
      variant="outline"
      className={cn('gap-1.5 font-medium border', cfg.text, cfg.bg, cfg.border, pad, className)}
      data-testid={`safety-badge-${t}`}
      data-tier={t}
    >
      {showDot && <span className={cn('h-2 w-2 rounded-full', cfg.dot)} aria-hidden="true" />}
      {showLabel && <span>{label ?? cfg.label}</span>}
    </Badge>
  );
}

export default SafetyBadge;
