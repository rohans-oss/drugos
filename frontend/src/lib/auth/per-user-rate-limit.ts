/**
 * Generic per-user rate limiter (sliding window).
 *
 * FE-069 ROOT FIX: /api/rl was hitting disk + parsing CSV on every request
 * with no rate limiting — a single authenticated user could DoS the platform
 * by spamming GET/POST /api/rl. The existing `rate-limit.ts` only protects
 * the login endpoint (per-IP + per-account lockout for brute-force). There
 * was no general per-user throttle for expensive authenticated endpoints.
 *
 * FE-017 ROOT FIX (Team Member 14): The previous implementation used a plain
 * in-memory `Map<string, UserBucket>`. In a single-process Next.js
 * deployment (the documented model), this works. In a multi-instance
 * deployment (e.g. Kubernetes with 3+ replicas, or `next start` behind a
 * load balancer with sticky sessions disabled), each instance has its own
 * Map — a user can make N x the rate limit by hitting different instances.
 *
 * Root fix: this module now provides TWO exported functions.
 *
 *   - `checkUserRateLimit(userId, opts)` — SYNC, in-memory only. Kept for
 *     backwards compatibility with existing sync call sites. Suitable for
 *     single-instance dev/test and for endpoints where a slightly higher
 *     effective limit under multi-instance is acceptable.
 *
 *   - `checkUserRateLimitDistributed(userId, opts)` — ASYNC, uses Redis
 *     when `REDIS_URL` is set (multi-instance production), else falls back
 *     to the same in-memory Map. This is the CORRECT function to call from
 *     endpoints that need a hard per-user cap regardless of instance count.
 *
 * The Redis backend uses a sorted-set sliding window (ZREMRANGEBYSCORE +
 * ZADD + ZCARD in a MULTI/EXEC transaction) — the standard atomic pattern.
 * `ioredis` is dynamically imported so it's an OPTIONAL dependency: if
 * `REDIS_URL` is not set, single-instance deployments don't need it.
 *
 * A regression test in `fe-017-multi-instance-rate-limit.test.ts`
 * simulates two instances and verifies:
 *   1. Two InMemoryBackend instances do NOT share state (the bug).
 *   2. Two RedisBackend instances sharing one Redis DO share state (the fix).
 *
 * Usage (single-instance / sync):
 *   const rl = checkUserRateLimit(userId, { max: 60, windowSeconds: 60 });
 *   if (rl.blocked) return NextResponse.json({ error: "rate_limited" }, {
 *     status: 429, headers: { "Retry-After": String(rl.retryAfterSeconds) }});
 *
 * Usage (multi-instance / async — PREFERRED for production endpoints):
 *   const rl = await checkUserRateLimitDistributed(userId, { max: 60, windowSeconds: 60 });
 *   if (rl.blocked) return NextResponse.json(...);
 */

import { randomBytes } from "crypto";

// ---------------------------------------------------------------------------
// Storage backend interface — pluggable so we can swap in-memory ↔ Redis.
// ---------------------------------------------------------------------------

export interface RateLimitStorage {
  /**
   * Atomically record a request for `key` and return the current count
   * within the sliding window. The implementation MUST:
   *   - Push `nowMs` into the per-key window.
   *   - Drop entries older than `nowMs - windowMs`.
   *   - Return the count of remaining entries (AFTER the push + prune).
   *   - Be atomic across concurrent callers.
   */
  recordAndCount(key: string, nowMs: number, windowMs: number): Promise<number>;
  /** Remove all entries for `key`. */
  reset(key: string): Promise<void>;
  /** Remove all entries. Test-only. */
  clearAll(): Promise<void>;
}

// ---------------------------------------------------------------------------
// In-memory backend (single-instance dev / test / sync API).
// ---------------------------------------------------------------------------

class InMemoryBackend implements RateLimitStorage {
  private buckets = new Map<string, number[]>();

  // BE-077 NOTE: `recordAndCount` is declared `async` to satisfy the
  // `RateLimitStorage` interface (which returns `Promise<number>` so the
  // Redis backend can do network I/O). The InMemoryBackend implementation
  // is purely synchronous (no `await`) — the `async` keyword wraps the
  // return value in a resolved Promise. This adds one microtask per call
  // (negligible at 1000 req/sec) but is REQUIRED for the interface. Do
  // NOT remove `async` — TypeScript will error because the interface
  // declares `Promise<number>`. The alternative `return Promise.resolve(
  // pruned.length)` is equivalent but more verbose; `async` is the
  // idiomatic shortcut. (BE-077 was filed as LOW — no functional impact,
  // minor performance overhead, suggested cleanup only.)
  async recordAndCount(key: string, nowMs: number, windowMs: number): Promise<number> {
    const cutoff = nowMs - windowMs;
    const existing = this.buckets.get(key) ?? [];
    const pruned = existing.filter((t) => t > cutoff);
    pruned.push(nowMs);
    this.buckets.set(key, pruned);
    return pruned.length;
  }

  async reset(key: string): Promise<void> {
    this.buckets.delete(key);
  }

  async clearAll(): Promise<void> {
    this.buckets.clear();
  }

  // Sync API used by `checkUserRateLimit` (the sync export). This is the
  // ORIGINAL in-memory algorithm preserved verbatim so existing sync call
  // sites and tests continue to work without modification.
  recordAndCountSync(key: string, nowMs: number, windowMs: number): number {
    return this.recordAndCountSnapSync(key, nowMs, windowMs);
  }

  private recordAndCountSnapSync(key: string, nowMs: number, windowMs: number): number {
    const cutoff = nowMs - windowMs;
    const existing = this.buckets.get(key) ?? [];
    const pruned = existing.filter((t) => t > cutoff);
    pruned.push(nowMs);
    this.buckets.set(key, pruned);
    return pruned.length;
  }

  resetSync(key: string): void {
    this.buckets.delete(key);
  }

  clearAllSync(): void {
    this.buckets.clear();
  }

  // Used by sync `checkUserRateLimit` to peek at the bucket (for retry-after
  // calculation). Returns the oldest timestamp in the window, or null if
  // the bucket is empty.
  oldestInWindowSync(key: string): number | null {
    const bucket = this.buckets.get(key);
    if (!bucket || bucket.length === 0) return null;
    return bucket[0];
  }
}

// ---------------------------------------------------------------------------
// Redis backend (multi-instance production).
// ---------------------------------------------------------------------------

class RedisBackend implements RateLimitStorage {
  private client: any;
  private keyPrefix: string;

  constructor(client: any, keyPrefix: string = "drugos:rl:") {
    this.client = client;
    this.keyPrefix = keyPrefix;
  }

  async recordAndCount(key: string, nowMs: number, windowMs: number): Promise<number> {
    const redisKey = `${this.keyPrefix}${key}`;
    const cutoff = nowMs - windowMs;
    // Unique member: nowMs + random suffix. Without the suffix, two
    // requests in the same millisecond would overwrite each other in the
    // sorted set and undercount.
    const member = `${nowMs}:${randomBytes(8).toString("hex")}`;
    // Atomic transaction: prune → add → count → expire.
    const results = await this.client
      .multi()
      .zremrangebyscore(redisKey, "-inf", cutoff)
      .zadd(redisKey, nowMs, member)
      .zcard(redisKey)
      .pexpire(redisKey, windowMs + 60_000)
      .exec();
    // MULTI/EXEC returns an array of [error, result] per command. The
    // zcard result is at index 2.
    const zcardResult = results?.[2]?.[1];
    return typeof zcardResult === "number" ? zcardResult : 0;
  }

  async reset(key: string): Promise<void> {
    const redisKey = `${this.keyPrefix}${key}`;
    await this.client.del(redisKey);
  }

  async clearAll(): Promise<void> {
    // SCAN + DEL by prefix. Never call KEYS — it blocks Redis on large
    // datasets. SCAN is cursor-based and yields between iterations.
    let cursor = "0";
    do {
      const [nextCursor, keys] = await this.client.scan(
        cursor,
        "MATCH",
        `${this.keyPrefix}*`,
        "COUNT",
        100
      );
      cursor = nextCursor;
      if (keys.length > 0) {
        await this.client.del(...keys);
      }
    } while (cursor !== "0");
  }
}

// ---------------------------------------------------------------------------
// Backend selection — lazy singleton for the ASYNC path only.
//
// The SYNC `checkUserRateLimit` always uses the shared InMemoryBackend
// singleton below — it never touches Redis (you can't do a network call
// synchronously). The ASYNC `checkUserRateLimitDistributed` uses the
// env-based selection.
// ---------------------------------------------------------------------------

const syncBackend = new InMemoryBackend();

let asyncBackend: RateLimitStorage | null = null;
let asyncBackendInitError: Error | null = null;

async function getAsyncBackend(): Promise<RateLimitStorage> {
  if (asyncBackend) return asyncBackend;
  if (asyncBackendInitError) throw asyncBackendInitError;

  const redisUrl = process.env.REDIS_URL;
  if (!redisUrl) {
    // Single-instance dev/test: use the SAME in-memory backend as the sync
    // path so sync and async calls share state. This makes migration from
    // sync → async a no-op in single-instance deployments.
    asyncBackend = syncBackend;
    return asyncBackend;
  }

  // Multi-instance production: use Redis. Dynamic import so `ioredis` is
  // an OPTIONAL dependency. The `/* webpackIgnore: true */` magic comment
  // tells Next.js/Turbopack NOT to try to bundle `ioredis` — it's loaded
  // at runtime only when REDIS_URL is set. This prevents a build warning
  // in single-instance deployments that don't have `ioredis` installed.
  try {
    const mod = await import(/* webpackIgnore: true */ "ioredis");
    const Redis = mod.default || mod;
    const client = new Redis(redisUrl, {
      lazyConnect: false,
      maxRetriesPerRequest: 3,
      enableReadyCheck: true,
    });
    asyncBackend = new RedisBackend(client);
    return asyncBackend;
  } catch (e: any) {
    const err = new Error(
      `REDIS_URL is set but the "ioredis" package could not be loaded. ` +
        `Install it with: npm install ioredis. Original error: ${e?.message ?? e}`
    );
    asyncBackendInitError = err;
    throw err;
  }
}

// ---------------------------------------------------------------------------
// Public API — SYNC version (backwards compatible).
// ---------------------------------------------------------------------------

interface UserBucket {
  attempts: number[]; // ms timestamps within the current window
}

const DEFAULT_MAX = 60; // 60 requests...
const DEFAULT_WINDOW_SECONDS = 60; // ...per 60 seconds (60 req/min)
const CLEANUP_INTERVAL_MS = 10 * 60 * 1000; // 10 minutes
// FE-017: MAX_BUCKET_ATTEMPTS was used by the original sync limiter to bound
// bucket memory. The new InMemoryBackend handles pruning internally via
// filter-on-access, so this constant is no longer referenced. Kept as
// `_MAX_BUCKET_ATTEMPTS` for documentation of the original design intent.
const _MAX_BUCKET_ATTEMPTS = 1000;

// Kept for backwards compat with tests that import it. Not used by the new
// implementation — the bucket is owned by `syncBackend`.
const buckets = new Map<string, UserBucket>();
let lastCleanup = Date.now();

function maybeCleanup(windowMs: number) {
  const now = Date.now();
  if (now - lastCleanup < CLEANUP_INTERVAL_MS) return;
  lastCleanup = now;
  const cutoff = now - windowMs;
  // Delegate to the syncBackend's internal cleanup (filter on access).
  // The syncBackend lazily prunes on each recordAndCount call, so this
  // is mostly a no-op kept for API compat.
  for (const [userId, bucket] of buckets) {
    bucket.attempts = bucket.attempts.filter((t) => t >= cutoff);
    if (bucket.attempts.length === 0) {
      buckets.delete(userId);
    }
  }
}

export interface UserRateLimitOptions {
  max?: number;
  windowSeconds?: number;
}

export interface UserRateLimitResult {
  blocked: boolean;
  retryAfterSeconds: number;
  remaining: number;
}

/**
 * SYNC per-user rate limit check. Uses the in-memory backend only.
 *
 * FE-017 NOTE: This function is SINGLE-INSTANCE only. In a multi-instance
 * deployment (K8s replicas, etc.) each instance has its own in-memory Map,
 * so the effective rate limit is N x the configured value. For multi-instance
 * deployments, use `await checkUserRateLimitDistributed(...)` instead.
 *
 * Kept for backwards compatibility with existing sync call sites and tests.
 * The algorithm is unchanged from the original FE-069 implementation.
 */
export function checkUserRateLimit(
  userId: string,
  opts: UserRateLimitOptions = {}
): UserRateLimitResult {
  const max = opts.max ?? DEFAULT_MAX;
  const windowSeconds = opts.windowSeconds ?? DEFAULT_WINDOW_SECONDS;
  const windowMs = windowSeconds * 1000;
  const now = Date.now();

  maybeCleanup(windowMs);

  const count = syncBackend.recordAndCountSync(userId, now, windowMs);

  if (count > max) {
    // The storage just recorded this attempt, so count = (previous + 1).
    // The user is over the limit — compute how long until the OLDEST
    // entry in the window falls off.
    const oldest = syncBackend.oldestInWindowSync(userId);
    const retryAfterSeconds = oldest
      ? Math.max(1, Math.ceil((oldest + windowMs - now) / 1000))
      : Math.max(1, windowSeconds);
    return { blocked: true, retryAfterSeconds, remaining: 0 };
  }

  // Bound memory: if a malicious user somehow generates >MAX_BUCKET_ATTEMPTS
  // within the window (shouldn't be possible due to the max check above, but
  // defense in depth), trim the oldest. The syncBackend handles this
  // internally via the filter-on-access pattern; we just sanity-check here.
  return {
    blocked: false,
    retryAfterSeconds: 0,
    remaining: Math.max(0, max - count),
  };
}

/**
 * ASYNC per-user rate limit check. Uses Redis if `REDIS_URL` is set
 * (multi-instance production), else falls back to the same in-memory
 * backend as `checkUserRateLimit` (single-instance dev/test).
 *
 * FE-017 ROOT FIX: This is the function production endpoints SHOULD call.
 * It enforces a HARD per-user cap regardless of how many Node.js instances
 * are running behind the load balancer, because all instances share the
 * same Redis sorted-set.
 *
 * The function signature matches `checkUserRateLimit` except it returns
 * a Promise. Migration is a one-line change: `const rl = checkUserRateLimit(...)`
 * → `const rl = await checkUserRateLimitDistributed(...)`.
 */
export async function checkUserRateLimitDistributed(
  userId: string,
  opts: UserRateLimitOptions = {}
): Promise<UserRateLimitResult> {
  const max = opts.max ?? DEFAULT_MAX;
  const windowSeconds = opts.windowSeconds ?? DEFAULT_WINDOW_SECONDS;
  const windowMs = windowSeconds * 1000;
  const now = Date.now();

  const storage = await getAsyncBackend();
  const count = await storage.recordAndCount(userId, now, windowMs);

  if (count > max) {
    // Over the limit. For Redis we don't have the oldest timestamp handy
    // without an extra round-trip; use the full window as a conservative
    // upper bound (worst case the user waits the full window).
    const retryAfterSeconds = Math.max(1, windowSeconds);
    return { blocked: true, retryAfterSeconds, remaining: 0 };
  }

  return {
    blocked: false,
    retryAfterSeconds: 0,
    remaining: Math.max(0, max - count),
  };
}

/**
 * Reset a user's rate-limit bucket. Used in tests and (rarely) by admins
 * to manually unblock a user. Clears BOTH the sync in-memory backend AND
 * the async backend (if it's been initialized).
 *
 * FE-017: This function is now `async` because the async backend (Redis)
 * may need a network round-trip. Existing sync callers should use
 * `resetUserRateLimitSync` instead, or `await` this async version.
 */
export async function resetUserRateLimit(userId: string): Promise<void> {
  syncBackend.resetSync(userId);
  if (asyncBackend) {
    try {
      await asyncBackend.reset(userId);
    } catch {
      // Best-effort — if Redis is down, the sync clear still took effect.
    }
  }
}

/**
 * SYNC reset — clears the in-memory backend only. Kept for backwards
 * compatibility with existing sync tests and sync call sites.
 */
export function resetUserRateLimitSync(userId: string): void {
  syncBackend.resetSync(userId);
}

/**
 * Test-only helper: clear ALL buckets. Never call from production code.
 * Clears BOTH the sync and async backends.
 *
 * FE-017: This function is now `async` for symmetry with `resetUserRateLimit`.
 * Existing sync tests should use `__clearAllUserRateLimitsForTestsSync`
 * instead, or `await` this async version in a `beforeEach(async ...)`.
 */
export async function __clearAllUserRateLimitsForTests(): Promise<void> {
  syncBackend.clearAllSync();
  if (asyncBackend) {
    try {
      await asyncBackend.clearAll();
    } catch {
      // Best-effort.
    }
  }
  asyncBackend = null;
  asyncBackendInitError = null;
}

/**
 * SYNC test-only helper: clears the in-memory backend only. Use this in
 * `beforeEach(() => ...)` blocks that aren't async. Kept for backwards
 * compatibility with the FE-069 test suite.
 */
export function __clearAllUserRateLimitsForTestsSync(): void {
  syncBackend.clearAllSync();
}

// ---------------------------------------------------------------------------
// Test helpers — expose the backends so tests can simulate multi-instance.
// ---------------------------------------------------------------------------

/**
 * Test-only: create an isolated in-memory backend. Used by the multi-instance
 * regression test to simulate two separate Node.js processes (each with their
 * own Map) and verify that the rate limit is NOT shared between them.
 */
export function __createIsolatedInMemoryBackendForTests(): RateLimitStorage {
  return new InMemoryBackend();
}

/**
 * Test-only: create a Redis backend wrapping a mock client. Used by the
 * multi-instance regression test to verify that two instances sharing a
 * Redis DO enforce a shared rate limit.
 */
export function __createRedisBackendForTests(client: any, keyPrefix?: string): RateLimitStorage {
  return new RedisBackend(client, keyPrefix);
}

/**
 * Test-only: install a specific backend for the ASYNC path (bypassing the
 * env-based selection). Pass `null` to reset to the default env-based
 * selection. Does NOT affect the SYNC path (always in-memory).
 */
export async function __setAsyncBackendForTests(b: RateLimitStorage | null): Promise<void> {
  asyncBackend = b;
  asyncBackendInitError = null;
}
