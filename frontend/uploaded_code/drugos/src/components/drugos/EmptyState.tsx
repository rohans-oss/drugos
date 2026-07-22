'use client';

/**
 * EmptyState — reusable empty-state component for the three core
 * value-prop screens (CandidateDetail, SafetyProfile, KnowledgeGraph).
 *
 * Root fix for audit issues #291 / #300: the previous screens crashed
 * when the underlying API returned no candidates / no safety data /
 * an empty graph, because they dereferenced `drugCandidates[0]` /
 * `safetyData[0]` / `positions.get(id)` directly. This component gives
 * every screen a single, consistent, crash-proof fallback UI.
 *
 * Acceptance criteria satisfied:
 *   - empty case renders gracefully (no exception)
 *   - shows a Lucide icon, a headline, a hint, and an optional CTA
 *   - never throws on undefined props
 */
import type { LucideIcon } from 'lucide-react';
import { Inbox } from 'lucide-react';
import { cn } from '@/lib/utils';

export interface EmptyStateProps {
  /** Lucide icon to show. Defaults to Inbox. */
  icon?: LucideIcon;
  /** Bold headline. Defaults to "Nothing here yet". */
  title?: string;
  /** Secondary explanation. */
  description?: string;
  /** Optional call-to-action element (Button, link, etc.). */
  action?: React.ReactNode;
  /** Hide the icon (text-only state). */
  hideIcon?: boolean;
  /** Extra class names. */
  className?: string;
  /** Size variant. */
  size?: 'sm' | 'md' | 'lg';
}

const sizeMap: Record<NonNullable<EmptyStateProps['size']>, {
  container: string;
  iconWrap: string;
  iconSize: string;
  title: string;
  description: string;
}> = {
  sm: {
    container: 'py-8',
    iconWrap: 'h-10 w-10',
    iconSize: 'h-5 w-5',
    title: 'text-sm font-semibold',
    description: 'text-xs',
  },
  md: {
    container: 'py-14',
    iconWrap: 'h-14 w-14',
    iconSize: 'h-7 w-7',
    title: 'text-base font-semibold',
    description: 'text-sm',
  },
  lg: {
    container: 'py-24',
    iconWrap: 'h-20 w-20',
    iconSize: 'h-10 w-10',
    title: 'text-lg font-semibold',
    description: 'text-sm',
  },
};

export function EmptyState({
  icon: Icon = Inbox,
  title = 'Nothing here yet',
  description,
  action,
  hideIcon = false,
  className,
  size = 'md',
}: EmptyStateProps) {
  const s = sizeMap[size];
  return (
    <div
      role="status"
      aria-live="polite"
      className={cn(
        'flex flex-col items-center justify-center text-center px-6',
        s.container,
        className,
      )}
      data-testid="empty-state"
    >
      {!hideIcon && (
        <div
          className={cn(
            'mb-4 flex items-center justify-center rounded-full bg-muted/60 text-muted-foreground/70',
            s.iconWrap,
          )}
        >
          <Icon className={s.iconSize} aria-hidden="true" />
        </div>
      )}
      <h3 className={cn('text-foreground', s.title)}>{title}</h3>
      {description && (
        <p className={cn('mt-1 text-muted-foreground max-w-md', s.description)}>
          {description}
        </p>
      )}
      {action && <div className="mt-4">{action}</div>}
    </div>
  );
}

export default EmptyState;
