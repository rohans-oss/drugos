import type { NextConfig } from "next";

/**
 * BE-012 ROOT FIX (v115, HIGH): production security headers.
 *
 * ROOT CAUSE: the previous next.config.ts had ZERO security headers.
 * The middleware.ts file set CSP / X-Frame-Options / etc., but only on
 * routes that matched the middleware matcher (which EXCLUDES static
 * assets, _next/static, etc.). Worse, middleware runs AFTER Next.js
 * serves a response for some paths — so the headers were missing on
 * critical responses.
 *
 * The OWASP ASVS V5.1 requires a hardened HTTP response header set on
 * EVERY response from a production web app. The Next.js `headers()`
 * function in next.config.ts is the canonical place for these — it
 * applies to every route (including static assets, API routes, and
 * the Next.js internal routes) and is evaluated at build time so there
 * is no runtime overhead.
 *
 * This is defense-in-depth alongside the middleware CSP:
 *   - middleware.ts: applies the strict CSP with 'unsafe-inline' (needed
 *     for Next.js hydration) on dynamic routes.
 *   - next.config.ts headers(): applies the broader security header
 *     baseline (X-Frame-Options, X-Content-Type-Options, Referrer-Policy,
 *     HSTS, Permissions-Policy) to EVERY response, including static
 *     assets that bypass middleware.
 *
 * The CSP here is a SECONDARY defense — it's looser than middleware's
 * (allows 'unsafe-inline' for both scripts and styles, which is what
 * Next.js requires without per-request nonces). The middleware's CSP
 * overrides this for dynamic routes. For static assets (JS/CSS files
 * served from /_next/static/), this CSP still applies — and since
 * those assets are same-origin, 'unsafe-inline' is irrelevant for them.
 */
const securityHeaders = [
  {
    key: "X-Frame-Options",
    value: "DENY",
  },
  {
    key: "X-Content-Type-Options",
    value: "nosniff",
  },
  {
    key: "Referrer-Policy",
    value: "strict-origin-when-cross-origin",
  },
  {
    key: "Permissions-Policy",
    value: "camera=(), microphone=(), geolocation=(), payment=(), usb=(), magnetometer=(), gyroscope=(), accelerometer=()",
  },
  {
    key: "Strict-Transport-Security",
    value: "max-age=63072000; includeSubDomains; preload",
  },
  // Primary CSP — also enforced by middleware.ts for dynamic routes.
  // 'unsafe-inline' is required for Next.js 16 hydration scripts and
  // styled-components / Tailwind inline styles. A future hardening
  // pass should replace 'unsafe-inline' with per-request nonces once
  // Next.js 16's nonce support is wired up (see middleware.ts comment).
  {
    key: "Content-Security-Policy",
    value: [
      "default-src 'self'",
      "script-src 'self' 'unsafe-inline'",
      "style-src 'self' 'unsafe-inline'",
      "img-src 'self' data: https:",
      "font-src 'self' data:",
      "connect-src 'self' https:",
      "frame-ancestors 'none'",
      "object-src 'none'",
      "base-uri 'self'",
      "form-action 'self'",
      "upgrade-insecure-requests",
    ].join("; "),
  },
];

const nextConfig: NextConfig = {
  // Note: output: "standalone" is enabled for production Docker/Node deployments.
  // Disable it locally if you just want `next dev` / `next start` to work without
  // copying the .next/standalone folder around.
  output: "standalone",
  // FE-011/FE-012/FE-013 ROOT FIX: typescript.ignoreBuildErrors was previously
  // `true`, which let broken imports silently pass the build. Production-grade
  // code MUST fail the build on type errors.
  typescript: {
    ignoreBuildErrors: false,
  },
  // FE-028 ROOT FIX: reactStrictMode was disabled — React 19's built-in
  // bug detection was off. Strict mode catches stale closures and missing
  // effect cleanups BEFORE they reach production.
  reactStrictMode: true,
  // BE-012 ROOT FIX (v115): apply security headers to EVERY response,
  // including static assets that bypass the middleware matcher.
  async headers() {
    return [
      {
        source: "/:path*",
        headers: securityHeaders,
      },
    ];
  },
};

export default nextConfig;
