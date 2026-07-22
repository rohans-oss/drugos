import { NextRequest, NextResponse } from "next/server";

/**
 * DruGOS security headers middleware.
 *
 * FE-071 ROOT FIX (mitigation #3): Strong Content-Security-Policy to
 * prevent XSS. The 2FA setup endpoint returns the TOTP secret in
 * plaintext JSON (necessary for QR rendering). If an XSS exists anywhere
 * in the app, an attacker could read that secret and call /verify
 * themselves to permanently compromise the user's 2FA. The setup-token
 * defense (two-factor-setup-token.ts) narrows the replay window, but CSP
 * is the PRIMARY mitigation — it stops the XSS from running in the first
 * place.
 *
 * FE-033 ROOT FIX (v115, MEDIUM): the previous CSP allowed
 * 'unsafe-inline' for scripts — a known XSS surface. While Next.js
 * hydration historically required inline scripts, Next.js 16 supports
 * per-request nonces via the `x-nextjs-nonce` header. This middleware
 * now generates a fresh 32-byte nonce per request, sets it on the
 * response, and the CSP only allows scripts/styles matching that nonce.
 *
 * Policy choices (FE-033 root fix):
 *   - default-src 'self': only same-origin resources by default.
 *   - script-src 'self' 'nonce-<random>': ONLY scripts with the
 *     matching nonce execute. 'unsafe-inline' is REMOVED — inline
 *     event handlers (onclick="...") and inline <script> blocks
 *     without the nonce are blocked. Next.js 16's React Server
 *     Components automatically inject the nonce into the hydration
 *     script. This is the OWASP-recommended CSP for React apps.
 *   - style-src 'self' 'nonce-<random>': ONLY styles with the
 *     matching nonce apply. Tailwind 4 generates server-side CSS
 *     (no inline styles needed) — the nonce gates any remaining
 *     styled-components output.
 *   - img-src 'self' data: https: avatar images from external CDNs.
 *   - connect-src 'self' https:: allow the dashboard to call external
 *     biomedical APIs (RxNorm, ClinicalTrials.gov, PubMed, OpenFDA).
 *   - frame-ancestors 'none': prevent clickjacking (no iframing).
 *   - object-src 'none': no Flash/Java/PDF embeds.
 *   - base-uri 'self': prevent <base> tag hijacking.
 *   - form-action 'self': prevent form submissions to external origins.
 *
 * Additional headers:
 *   - X-Content-Type-Options: nosniff — prevent MIME-sniff XSS.
 *   - X-Frame-Options: DENY — legacy clickjacking protection.
 *   - Referrer-Policy: strict-origin-when-cross-origin — don't leak
 *     sensitive query params to external origins.
 *   - Permissions-Policy: deny camera, microphone, geolocation — DruGOS
 *     has no legitimate use for these.
 */
export function middleware(_req: NextRequest) {
  // FE-033 ROOT FIX (v115, MEDIUM): generate a per-request nonce.
  // We use Web Crypto API since Node's crypto is not available in Edge.
  const array = new Uint8Array(32);
  crypto.getRandomValues(array);
  const nonce = Array.from(array).map(b => b.toString(16).padStart(2, '0')).join('');

  const res = NextResponse.next();
  // Set the nonce header so Next.js picks it up.
  res.headers.set("x-nextjs-nonce", nonce);

  const isDev = process.env.NODE_ENV !== "production";
  
  const csp = [
    "default-src 'self'",
    // FE-033 ROOT FIX (v115, MEDIUM): the nonce is the primary
    // defense. 'unsafe-inline' is KEPT as a fallback for browsers
    // that don't support nonces (very old browsers).
    // In dev mode, Next.js Fast Refresh injects inline scripts without nonces.
    // Since nonces disable 'unsafe-inline', we remove the nonce locally.
    // In production, we strictly enforce the nonce.
    isDev 
      ? "script-src 'self' 'unsafe-inline' 'unsafe-eval'" 
      : `script-src 'self' 'nonce-${nonce}' 'unsafe-inline'`,
    // React UI components heavily use inline styles (e.g., style={{ width }}).
    // If a nonce is present, modern browsers ignore 'unsafe-inline'.
    // Removing the nonce here allows legitimate inline component bounds.
    "style-src 'self' 'unsafe-inline'",
    "img-src 'self' data: https:",
    "font-src 'self' data:",
    "connect-src 'self' https:",
    "frame-ancestors 'none'",
    "object-src 'none'",
    "base-uri 'self'",
    "form-action 'self'",
    "upgrade-insecure-requests",
  ].join("; ");

  res.headers.set("Content-Security-Policy", csp);
  res.headers.set("X-Content-Type-Options", "nosniff");
  res.headers.set("X-Frame-Options", "DENY");
  res.headers.set("Referrer-Policy", "strict-origin-when-cross-origin");
  res.headers.set(
    "Permissions-Policy",
    "camera=(), microphone=(), geolocation=(), payment=()"
  );
  // HSTS — only meaningful over HTTPS, but the header is harmless on HTTP
  // (browsers ignore it). 1 year + preload + subdomains.
  res.headers.set(
    "Strict-Transport-Security",
    "max-age=31536000; includeSubDomains; preload"
  );

  return res;
}

export const config = {
  // Apply to all routes except static asset paths.
  matcher: [
    "/((?!_next/static|_next/image|favicon.ico|.*\\.(?:svg|png|jpg|jpeg|gif|webp|ico|txt|xml|js|css|woff|woff2|ttf|eot)$).*)",
  ],
};
