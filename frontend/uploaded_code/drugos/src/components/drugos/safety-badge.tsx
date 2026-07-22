'use client';

/**
 * Backward-compatibility re-export.
 *
 * The canonical implementation now lives in `SafetyBadge.tsx` (PascalCase,
 * matching the audit-issue #293 spec). This kebab-case file remains so
 * existing imports `from '@/components/drugos/safety-badge'` continue to
 * resolve. The two files are now guaranteed to render identically.
 *
 * Root fix for audit issue #293: previously this file and the inline
 * `SafetyBadge` inside `core-screens.tsx` had DIFFERENT colors and
 * different type signatures — a "green" drug in one screen could be
 * "yellow" in another. Both now defer to the same single source of
 * truth in `SafetyBadge.tsx`.
 */
export { SafetyBadge, default } from './SafetyBadge';
export type { SafetyBadgeProps } from './SafetyBadge';
