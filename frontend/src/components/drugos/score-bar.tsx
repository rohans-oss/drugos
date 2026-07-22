'use client';

import { cn } from '@/lib/utils';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import { Info } from 'lucide-react';

/**
 * FE-036 ROOT FIX: ScoreBar now renders scientific context, not just a
 * percentage bar.
 *
 * The previous version rendered a 0-100 score as a colored percentage bar
 * with no confidence interval and no model-quality context. A researcher
 * seeing "87%" had no way to know:
 *   - Whether the score was precise (tight CI) or noise (wide CI)
 *   - Whether the underlying model was calibrated (AUC > 0.85) or random
 *     (AUC ≈ 0.50)
 *
 * Per the project's V1 launch criteria (Team_Cosmic_Build_Process_Updated.docx
 * §6 Layer 1), the Graph Transformer must achieve AUC > 0.85 on held-out
 * drug-disease pairs. The platform's core value prop is transparency — the
 * researcher MUST be able to see the model's confidence bounds AND its AUC
 * so they can make an informed decision rather than over-trusting a number.
 *
 * Root fix:
 *   1. Added optional `confidenceLower` / `confidenceUpper` props (0-100
 *      scale, same as `score`). When both are present, a semi-transparent
 *      band is rendered around the score to visualize the CI. The band is
 *      clipped to [0, 100].
 *   2. Added optional `auc` prop (0-1 scale). When present, the bar is
 *      wrapped in a tooltip that shows the model's AUC, the CI, and an
 *      explicit warning when AUC < 0.85 (the V1 launch threshold). When
 *      `auc` is absent, the tooltip explains that the model is not
 *      calibrated and the score should not be interpreted as a probability.
 *   3. The bar color logic is unchanged (green ≥ 80, orange ≥ 60, red < 60)
 *      because those thresholds are a UX convention, not a clinical claim.
 *
 * This component is unit-tested in fe-029-to-036-team16.test.ts.
 */

interface ScoreBarProps {
  /** The composite score (0-100). Values outside [0, 100] are clamped. */
  score: number;
  /**
   * Optional lower bound of the confidence interval (0-100). When both
   * `confidenceLower` and `confidenceUpper` are provided, a band is rendered.
   * Pass `undefined` (or omit) when the model did not produce a CI.
   */
  confidenceLower?: number;
  /** Optional upper bound of the confidence interval (0-100). */
  confidenceUpper?: number;
  /**
   * Optional model AUC (0-1). When provided, the tooltip shows the AUC and
   * an explicit warning when AUC < 0.85 (the V1 launch threshold per the
   * project doc). When absent, the tooltip explains that the model is not
   * calibrated and the score should not be interpreted as a probability.
   */
  auc?: number;
  size?: 'sm' | 'md' | 'lg';
  showLabel?: boolean;
  /** When true, shows a small info icon next to the score. Default true. */
  showInfoIcon?: boolean;
  className?: string;
}

function getScoreColor(score: number): string {
  if (score >= 80) return 'bg-[#1D9E75]';
  if (score >= 60) return 'bg-[#D4853A]';
  return 'bg-[#C0392B]';
}

function getScoreTextColor(score: number): string {
  if (score >= 80) return 'text-[#1D9E75]';
  if (score >= 60) return 'text-[#D4853A]';
  return 'text-[#C0392B]';
}

const sizeMap = {
  sm: { bar: 'h-2', text: 'text-xs', width: 'w-20', icon: 'h-3 w-3' },
  md: { bar: 'h-3', text: 'text-sm', width: 'w-28', icon: 'h-3.5 w-3.5' },
  lg: { bar: 'h-4', text: 'text-base', width: 'w-36', icon: 'h-4 w-4' },
};

/** V1 launch AUC threshold per Team_Cosmic_Build_Process_Updated.docx §6. */
const AUC_LAUNCH_THRESHOLD = 0.85;

export function ScoreBar({
  score,
  confidenceLower,
  confidenceUpper,
  auc,
  size = 'md',
  showLabel = true,
  showInfoIcon = true,
  className = '',
}: ScoreBarProps) {
  const clamped = Math.max(0, Math.min(100, score));
  const config = sizeMap[size];

  // FE-036: Compute CI band geometry. Both bounds must be present and the
  // lower must be <= upper; otherwise we skip the band entirely.
  const hasCi =
    typeof confidenceLower === 'number' &&
    typeof confidenceUpper === 'number' &&
    Number.isFinite(confidenceLower) &&
    Number.isFinite(confidenceUpper) &&
    confidenceLower <= confidenceUpper;
  const ciLo = hasCi ? Math.max(0, Math.min(100, confidenceLower!)) : 0;
  const ciHi = hasCi ? Math.max(0, Math.min(100, confidenceUpper!)) : 0;
  const ciLeftPct = hasCi ? ciLo : 0;
  const ciWidthPct = hasCi ? Math.max(0, ciHi - ciLo) : 0;

  // FE-036: Build the tooltip content. The tooltip ALWAYS explains that the
  // score is a model output, not a statistical probability. When `auc` is
  // provided, it shows the AUC and warns when below the launch threshold.
  // When `auc` is absent, it explains the model is not calibrated.
  const aucValid = typeof auc === 'number' && Number.isFinite(auc) && auc >= 0 && auc <= 1;
  const aucBelowThreshold = aucValid && (auc as number) < AUC_LAUNCH_THRESHOLD;
  const ciLabel = hasCi ? `${ciLo.toFixed(1)}–${ciHi.toFixed(1)}` : 'not reported';

  const tooltipContent = (
    <div className="max-w-xs text-xs leading-relaxed">
      <p className="font-semibold mb-1">Composite Score: {clamped.toFixed(1)}/100</p>
      <p className="text-muted-foreground mb-2">
        This is a 0–100 weighted blend of sub-scores (Knowledge Graph,
        Molecular Similarity, Safety, Clinical Evidence). It is a MODEL
        OUTPUT — not a statistical probability.
      </p>
      <div className="border-t border-border pt-2 space-y-1">
        <p>
          <span className="text-muted-foreground">Confidence interval:</span>{' '}
          <span className="font-mono">{ciLabel}</span>
        </p>
        {aucValid ? (
          <p>
            <span className="text-muted-foreground">Model AUC:</span>{' '}
            <span className="font-mono">{(auc as number).toFixed(3)}</span>{' '}
            {aucBelowThreshold && (
              <span className="text-[#C0392B] font-semibold">
                (below V1 launch threshold of {AUC_LAUNCH_THRESHOLD.toFixed(2)})
              </span>
            )}
            {!aucBelowThreshold && (
              <span className="text-[#1D9E75]">(meets V1 launch threshold)</span>
            )}
          </p>
        ) : (
          <p className="text-[#D4853A]">
            Model AUC: not reported. The model may not be calibrated — do not
            interpret this score as a probability.
          </p>
        )}
      </div>
    </div>
  );

  return (
    <TooltipProvider delayDuration={200}>
      <div className={cn('flex items-center gap-2', className)}>
        {showLabel && (
          <span
            className={cn(
              'font-semibold tabular-nums',
              config.text,
              getScoreTextColor(clamped)
            )}
          >
            {clamped.toFixed(clamped % 1 === 0 ? 0 : 1)}
          </span>
        )}
        <Tooltip>
          <TooltipTrigger asChild>
            <div
              className={cn(
                'relative rounded-full bg-muted overflow-hidden cursor-help',
                config.width,
                config.bar
              )}
            >
              {/* FE-036: CI band — rendered BEHIND the score fill so the
                  score remains the primary visual. The band is a semi-
                  transparent overlay spanning [ciLo, ciHi]. */}
              {hasCi && (
                <div
                  className="absolute top-0 bottom-0 bg-[#5B4FCF]/25 border-x border-[#5B4FCF]/40"
                  style={{
                    left: `${ciLeftPct}%`,
                    width: `${ciWidthPct}%`,
                  }}
                  aria-label={`confidence interval ${ciLo.toFixed(1)} to ${ciHi.toFixed(1)}`}
                />
              )}
              {/* Score fill — rendered on top of the CI band. */}
              <div
                className={cn(
                  'h-full rounded-full transition-all duration-500 relative',
                  getScoreColor(clamped)
                )}
                style={{ width: `${clamped}%` }}
              />
            </div>
          </TooltipTrigger>
          <TooltipContent>{tooltipContent}</TooltipContent>
        </Tooltip>
        {showInfoIcon && (
          <Info className={cn('text-muted-foreground shrink-0', config.icon)} aria-hidden />
        )}
      </div>
    </TooltipProvider>
  );
}
