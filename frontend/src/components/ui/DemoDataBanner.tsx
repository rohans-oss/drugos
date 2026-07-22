'use client';

/**
 * DemoDataBanner — visible "DEMO DATA — DO NOT USE FOR DECISIONS" notice.
 *
 * Issue 317 (audit 301-320): Screens that still render mock/hardcoded data
 * in production MUST surface a clearly visible banner so a researcher,
 * admin, investor, or compliance officer cannot mistake fabricated numbers
 * for real telemetry. The banner is intentionally loud (amber, bold,
 * full-width, sticky to the top of the screen) because silent mock data
 * has already caused real harm in this codebase (investors were shown
 * fabricated ARR; compliance officers were shown fabricated HIPAA status).
 *
 * USAGE CONTRACT (enforced by frontend/tests/e2e/no-mock-data-in-production.e2e.ts):
 *
 *   1. ANY screen that still renders hardcoded/fabricated data MUST render
 *      <DemoDataBanner reason="..." /> at the very top of its body.
 *   2. The `reason` prop MUST explain WHAT is fabricated and WHY the real
 *      backend has not been wired (e.g. "SSO provider config requires
 *      SAML/OIDC integration not yet deployed").
 *   3. Screens that wire to a real API and render real (possibly empty)
 *      data MUST NOT render this banner — an EmptyState is the correct
 *      component for "real call returned zero rows".
 *   4. The banner must be rendered BEFORE any mock data so a user who
 *      scrolls sees the warning first.
 *
 * This component is intentionally stateless and presentational. It does
 * not call any API, does not depend on context, and renders identically
 * on server and client so Playwright e2e tests can assert its presence
 * deterministically.
 */

import { AlertTriangle } from 'lucide-react';
import { cn } from '@/lib/utils';

export interface DemoDataBannerProps {
  /**
   * Short, specific explanation of what is fabricated and why the real
   * backend is not wired. Shown as the banner's secondary line.
   *
   * Example: "SSO provider config is fabricated. Real SAML/OIDC
   * integration has not been deployed."
   */
  reason?: string;
  /**
   * Optional override of the banner title. Defaults to
   * "DEMO DATA — DO NOT USE FOR DECISIONS".
   */
  title?: string;
  className?: string;
}

/**
 * The canonical, machine-greppable banner string. The e2e test in
 * `frontend/tests/e2e/no-mock-data-in-production.e2e.ts` greps the
 * production bundle for this exact phrase to ensure every screen that
 * still shows mock data also shows the banner.
 */
export const DEMO_DATA_BANNER_TEXT = 'DEMO DATA — DO NOT USE FOR DECISIONS';

export function DemoDataBanner({
  reason,
  title = DEMO_DATA_BANNER_TEXT,
  className,
}: DemoDataBannerProps) {
  return (
    <div
      role="alert"
      aria-live="polite"
      data-testid="demo-data-banner"
      className={cn(
        'w-full rounded-md border border-amber-300 bg-amber-100 text-amber-950',
        'dark:border-amber-700 dark:bg-amber-950/60 dark:text-amber-100',
        'px-4 py-3 mb-4 flex items-start gap-3 shadow-sm',
        className,
      )}
    >
      <AlertTriangle className="h-5 w-5 shrink-0 mt-0.5 text-amber-600 dark:text-amber-400" />
      <div className="flex-1 min-w-0">
        <p className="text-sm font-bold uppercase tracking-wide">{title}</p>
        {reason && (
          <p className="text-xs mt-1 text-amber-900 dark:text-amber-200 leading-relaxed">
            {reason}
          </p>
        )}
      </div>
    </div>
  );
}

export default DemoDataBanner;
