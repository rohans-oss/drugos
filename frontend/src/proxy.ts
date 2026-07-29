import { NextRequest, NextResponse } from "next/server";

/**
 * DruGOS security headers proxy.
 */
export function proxy(_req: NextRequest) {
  const array = new Uint8Array(32);
  crypto.getRandomValues(array);
  const nonce = Array.from(array).map(b => b.toString(16).padStart(2, '0')).join('');

  const res = NextResponse.next();
  res.headers.set("x-nextjs-nonce", nonce);

  const csp = [
    "default-src 'self'",
    "script-src 'self' 'unsafe-inline' 'unsafe-eval'",
    "style-src 'self' 'unsafe-inline'",
    "img-src 'self' data: https:",
    "font-src 'self' data:",
    "connect-src 'self' https:",
    "frame-ancestors 'none'",
    "object-src 'none'",
    "base-uri 'self'",
    "form-action 'self'",
  ].join("; ");

  res.headers.set("Content-Security-Policy", csp);
  res.headers.set("X-Content-Type-Options", "nosniff");
  res.headers.set("X-Frame-Options", "DENY");
  res.headers.set("Referrer-Policy", "strict-origin-when-cross-origin");
  res.headers.set(
    "Permissions-Policy",
    "camera=(), microphone=(), geolocation=(), payment=()"
  );

  return res;
}

export const config = {
  matcher: [
    "/((?!_next/static|_next/image|favicon.ico|.*\\.(?:svg|png|jpg|jpeg|gif|webp|ico|txt|xml|js|css|woff|woff2|ttf|eot)$).*)",
  ],
};
