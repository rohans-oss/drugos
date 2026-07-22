/**
 * Ambient module declarations for OPTIONAL runtime dependencies.
 *
 * These packages are loaded via dynamic `import()` at runtime ONLY when
 * the corresponding env var is set (e.g. REDIS_URL → ioredis). They are
 * NOT listed in package.json `dependencies` because the platform needs to
 * run without them in single-instance dev/test mode.
 *
 * In production multi-instance deployments, the operator installs them
 * explicitly: `npm install ioredis`.
 *
 * This .d.ts file gives TypeScript enough type info to compile. The actual
 * runtime behavior is unchanged — if the package is missing, the dynamic
 * import throws and the caller falls back to the in-memory implementation
 * (which is the correct single-instance dev behavior).
 */

declare module "ioredis" {
  // Minimal surface area we use. The real `ioredis` package has many more
  // methods, but we only call `multi`, `scan`, and `del` directly. The
  // MULTI/EXEC builder methods (zremrangebyscore, zadd, zcard, pexpire)
  // are chained off `multi()` and are typed as `any` here.
  export interface RedisCommand {
    zremrangebyscore(key: string, min: string | number, max: string | number): RedisCommand;
    zadd(key: string, score: number, member: string): RedisCommand;
    zcard(key: string): RedisCommand;
    pexpire(key: string, ms: number): RedisCommand;
    exec(): Promise<Array<[Error | null, any]>>;
  }
  export interface RedisClient {
    multi(): RedisCommand;
    scan(cursor: string, ...args: any[]): Promise<[string, string[]]>;
    del(...keys: string[]): Promise<number>;
  }
  const Redis: {
    new (url: string, opts?: Record<string, unknown>): RedisClient;
  };
  export default Redis;
}

// BE-058 / IN-043 ROOT FIX (v115): ambient declaration for the optional
// @sentry/nextjs dependency. When operators install @sentry/nextjs and
// set SENTRY_DSN, lib/sentry.ts loads the package via dynamic import
// and uses it. When the package is NOT installed, the dynamic import
// throws and lib/sentry.ts catches and falls back to no-op. This
// declaration gives TypeScript enough type info to compile without
// the package installed.
declare module "@sentry/nextjs" {
  // Minimal surface area used by lib/sentry.ts. The real package has
  // many more methods; we only call init, captureException, setTag,
  // setUser, and addBreadcrumb.
  export interface SentryEvent {
    request?: { headers?: Record<string, string> };
  }
  export interface CaptureOptions {
    tags?: Record<string, string>;
    extra?: Record<string, unknown>;
  }
  export interface Breadcrumb {
    message: string;
    level?: "info" | "warning" | "error";
    data?: Record<string, unknown>;
  }
  export interface User {
    id?: string;
    email?: string;
  }
  export function init(opts: Record<string, unknown>): void;
  export function captureException(err: unknown, opts?: CaptureOptions | Record<string, unknown>): string;
  export function setTag(key: string, value: string): void;
  export function setUser(user: User | null): void;
  export function addBreadcrumb(crumb: Breadcrumb | Record<string, unknown>): void;
}
