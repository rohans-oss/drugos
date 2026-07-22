import { NextRequest, NextResponse } from "next/server";
import {
  requireAuth,
  internalError,
  writeAuditLog,
  requireCsrfOrSend,
} from "@/lib/api-helpers";
import {
  clearDrugMechanismCache,
  getDrugMechanismCacheState,
} from "@/lib/services/drug-mechanism";

/**
 * POST /api/drugs/mechanism/refresh
 * Body (optional): { drugName?: string }
 *
 * FE-028 ROOT FIX (Team Member 15):
 *
 * ROOT CAUSE: drug-mechanism.ts cached ChEMBL mechanism lookups with
 * no TTL — entries lived forever (until LRU-evicted). If ChEMBL
 * published a new mechanism, the dashboard continued showing the old
 * one indefinitely. The 5-minute TTL (added in drug-mechanism.ts)
 * bounds staleness, but operators may still want to force a refresh
 * on demand (e.g. right after a known ChEMBL update, or before a
 * pharma partner demo).
 *
 * ROOT FIX: expose this manual refresh endpoint. The dashboard's
 * "Refresh mechanism" button POSTs here. We clear the cache (either
 * a single drug or all entries), log the action, and return the
 * number of entries cleared.
 *
 * Auth: any authenticated user may refresh. CSRF required (state-changing).
 */
export async function POST(req: NextRequest) {
  const csrf = await requireCsrfOrSend(req);
  if (csrf.response) return csrf.response;

  const auth = await requireAuth();
  if (auth.user === null) return auth.response;

  let body: { drugName?: unknown } = {};
  try {
    body = await req.json();
  } catch {
    // Empty body is fine — clear all.
  }

  const drugName =
    typeof body.drugName === "string" ? body.drugName.trim() : undefined;

  try {
    const stateBefore = getDrugMechanismCacheState();
    if (drugName) {
      clearDrugMechanismCache(drugName);
    } else {
      clearDrugMechanismCache();
    }
    const stateAfter = getDrugMechanismCacheState();

    const clearedCount = stateBefore.length - stateAfter.length;

    await writeAuditLog({
      user: auth.user,
      action: "drug_mechanism_cache_refresh",
      resource: drugName ? `drug-mechanism:${drugName}` : "drug-mechanism:all",
      metadata: {
        drugName: drugName || null,
        clearedEntries: clearedCount,
        remainingEntries: stateAfter.length,
      },
    });

    return NextResponse.json({
      ok: true,
      drugName: drugName || null,
      clearedEntries: clearedCount,
      message:
        clearedCount === 0
          ? "Drug-mechanism cache was already empty — nothing to refresh."
          : drugName
            ? `Cleared cache entry for "${drugName}".`
            : `Cleared ${clearedCount} drug-mechanism cache entr${clearedCount === 1 ? "y" : "ies"}.`,
    });
  } catch (e: unknown) {
    const msg = e instanceof Error ? e.message : String(e);
    return internalError(`Drug-mechanism cache refresh failed: ${msg}`);
  }
}
