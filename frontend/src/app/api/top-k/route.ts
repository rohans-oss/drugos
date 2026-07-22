import { NextRequest, NextResponse } from "next/server";
import { requireAuth, internalError, writeAuditLog } from "@/lib/api-helpers";
import { topKNovel } from "@/lib/services/gt-inference";

/**
 * GET /api/top-k?top_k=<n>
 *
 * RT-006 ROOT FIX (Team Member 17): this route exposes the Phase 3
 * Graph Transformer's `top_k_novel_predictions()` inference function
 * to the frontend. Returns the highest-scoring NOVEL (drug, disease)
 * pairs from the trained model — "novel" means pairs not in the
 * known_pairs training set.
 *
 * This is the endpoint the Phase 6 literature cross-check calls to get
 * the top 50 novel predictions for PubMed verification (project docx
 * Section 8: "We take the model's top 50 novel predictions and run an
 * automated PubMed literature search").
 *
 * Issue 222 ROOT FIX: same as Issue 221 — the previous version had a
 * subprocess fallback that spawned a non-existent script. The new
 * gt-inference.ts (Issue 230) is HTTP-ONLY: it proxies to
 * `GT_SERVICE_URL/top-k?k=<n>` via the shared mlFetch HTTP client.
 * We NEVER fabricate predictions — if no checkpoint exists, we return
 * an empty list with `source: "none"` and a clear note.
 */
export async function GET(req: NextRequest) {
  const auth = await requireAuth();
  if (auth.user === null) return auth.response;

  const topK = Math.min(parseInt(req.nextUrl.searchParams.get("top_k") || "50", 10), 500);

  try {
    const result = await topKNovel(topK);
    await writeAuditLog({
      user: auth.user,
      action: "gt_top_k",
      resource: "gt:top_k",
      metadata: { count: result.count, source: result.source, topK },
    });
    return NextResponse.json(result);
  } catch (e: unknown) {
    const msg = e instanceof Error ? e.message : String(e);
    return internalError(`GT top-k failed: ${msg}`);
  }
}
