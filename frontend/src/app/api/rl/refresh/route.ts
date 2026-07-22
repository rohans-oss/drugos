import { NextRequest, NextResponse } from "next/server";
import { requireAuth, internalError, writeAuditLog } from "@/lib/api-helpers";
import { requireCsrfOrSend } from "@/lib/api-helpers";
// Issue 224 ROOT FIX: in HTTP-only mode (Issue 231), there is no local
// CSV cache to clear. The cache-clearing functions
// (clearRlRankerCsvCache, getRlRankerCsvCacheState) are now no-ops.
//
// This route now serves TWO purposes:
//   1. If RL_SERVICE_URL is set, it calls the service's /health endpoint
//      to verify the service is reachable. This is the "refresh" action
//      in HTTP mode — the next /api/rl request will re-fetch from the
//      service (which owns its own cache).
//   2. If RL_SERVICE_URL is NOT set, it returns a clear message telling
//      the operator to configure the service.
//
// The previous version cleared a cache module that no production route
// read from (rl-csv-cache.ts). The fix in this version: there is NO
// local cache at all — the Python service is the single source of truth.
import { checkRlHealth } from "@/lib/services/rl-ranker";

/**
 * POST /api/rl/refresh
 *
 * Issue 224 ROOT FIX: in HTTP-only mode, this route verifies the RL
 * service is reachable. There is no local cache to clear — the Python
 * service owns the cache.
 *
 * Auth: any authenticated user may refresh (the cache is per-process,
 * not per-user). We log the action to the audit trail.
 *
 * CSRF: required (this is a state-changing POST).
 */
export async function POST(req: NextRequest) {
  const csrf = await requireCsrfOrSend(req);
  if (csrf.response) return csrf.response;

  const auth = await requireAuth();
  if (auth.user === null) return auth.response;

  try {
    // Issue 224: check the RL service health instead of clearing a
    // local cache. The Python service is the single source of truth.
    const health = await checkRlHealth();

    await writeAuditLog({
      user: auth.user,
      action: "rl_cache_refresh",
      resource: "rl:cache",
      metadata: {
        configured: health.configured,
        reachable: health.reachable,
        checkpointConfigured: health.checkpointConfigured,
        csvOutputAvailable: health.csvOutputAvailable,
        // Issue 224: document that there is no local cache to clear.
        cachesCleared: [],
        mode: "http_only",
      },
    });

    if (!health.configured) {
      return NextResponse.json({
        ok: false,
        mode: "http_only",
        message:
          "RL_SERVICE_URL is not set. There is no local cache to clear " +
          "(HTTP-only mode). Set RL_SERVICE_URL to enable the Phase 4 " +
          "RL service proxy. Issue 224 ROOT FIX: the previous version " +
          "cleared a cache module (rl-csv-cache.ts) that no production " +
          "route read from — the refresh was a no-op.",
      });
    }

    if (!health.reachable) {
      return NextResponse.json({
        ok: false,
        mode: "http_only",
        configured: true,
        reachable: false,
        message:
          "RL_SERVICE_URL is set but the service is not reachable. " +
          "Check that `python rl/service.py` is running and the URL " +
          "is correct. There is no local cache to clear (HTTP-only mode).",
      });
    }

    return NextResponse.json({
      ok: true,
      mode: "http_only",
      configured: true,
      reachable: true,
      checkpointConfigured: health.checkpointConfigured,
      csvOutputAvailable: health.csvOutputAvailable,
      message:
        "RL service is reachable. There is no local cache to clear " +
        "(HTTP-only mode) — the Python service owns the cache. The " +
        "next /api/rl request will fetch fresh data from the service.",
    });
  } catch (e: unknown) {
    const msg = e instanceof Error ? e.message : String(e);
    return internalError(`RL health check failed: ${msg}`);
  }
}
