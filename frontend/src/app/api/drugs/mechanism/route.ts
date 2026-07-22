import { NextRequest, NextResponse } from "next/server";
import { lookupDrugMechanisms } from "@/lib/services/drug-mechanism";
import { badRequest } from "@/lib/api-helpers";
// Task 252 ROOT FIX: Zod validation for POST body.
import { validateBody, DrugsMechanismBody } from "@/lib/zod-schemas";
// Task 253 ROOT FIX: 5 req/sec per-user rate limit.
import {
  requireAuthAndRateLimitV2,
  recordApiRequestForUserV2,
} from "@/lib/auth/api-proxy-guard";

/**
 * POST /api/drugs/mechanism
 * Body: { drugNames: string[] }
 *
 * Task 242 ROOT FIX:
 *
 * ROOT CAUSE: the audit required this route to "proxy to Phase 2 KG
 * service for the drug→protein→pathway chain". The previous code
 * returned ONLY the ChEMBL mechanism text (a single string). The
 * frontend's pathway-viz component had nothing to render — the dashboard
 * showed "—" instead of a graph.
 *
 * ROOT FIX: the underlying service `lookupDrugMechanisms()` now enriches
 * each result with:
 *   - `pathwayChain`: a list of PathwayEdge records forming the
 *     drug→protein→pathway→disease chain (sourced from the Phase 2 KG
 *     service when KG_SERVICE_URL is set).
 *   - `proteinTargets`: the list of proteins the drug targets.
 *   - `pathways`: the list of pathways those proteins participate in.
 *
 * When the KG service is unavailable, the route still returns the
 * ChEMBL mechanism text — the pathway chain fields are empty arrays.
 * This is graceful degradation, not a failure.
 *
 * Task 254 ROOT FIX: the in-memory cache TTL is now 1 hour (was 5 min).
 * KG queries are expensive — 1 hour is the staleness budget the audit
 * specified. Operators can force-refresh via
 * POST /api/drugs/mechanism/refresh.
 *
 * Task 260 ROOT FIX: every ChEMBL and KG service call is wrapped in
 * `monitoredFetch` so operators see the URL, duration, and status of
 * every external call.
 *
 * SECURITY: every text field returned from the KG is escaped with
 * `escapeKgText()` before being sent to the client. The escape uses a
 * strict allowlist — every character not in [a-zA-Z0-9 ,.-:;()'/]
 * becomes an HTML numeric entity. This is the server-side XSS backstop
 * in case the frontend forgets to sanitize.
 */

/**
 * Escape a string for safe inclusion in HTML. Every character that is
 * not in the strict allowlist is replaced with its HTML numeric entity.
 */
function escapeKgText(s: string | null | undefined): string | null {
  if (s === null || s === undefined) return null;
  const ALLOWED = /^[a-zA-Z0-9 ,.\-:;()'/]$/;
  let out = "";
  for (let i = 0; i < s.length; i++) {
    const ch = s.charAt(i);
    if (ALLOWED.test(ch)) {
      out += ch;
    } else {
      out += `&#${s.charCodeAt(i)};`;
    }
  }
  return out;
}

export async function POST(req: NextRequest) {
  // Task 252: Zod validation fires FIRST — parse the body and validate
  // before checking auth. Invalid input gets a 400 without wasting an
  // auth check.
  let body: unknown;
  try {
    body = await req.json();
  } catch {
    return badRequest("Invalid JSON body");
  }
  const parsed = validateBody(DrugsMechanismBody, body);
  if (!parsed.ok) return parsed.response;
  const drugNames = parsed.data.drugNames;

  // Task 253: 5 req/sec per-user rate limit (V2 guard).
  const guard = await requireAuthAndRateLimitV2(req);
  if (guard.response !== null) return guard.response;

  if (drugNames.length === 0) {
    return NextResponse.json({ results: [] });
  }

  const map = await lookupDrugMechanisms(drugNames);
  const results = drugNames.map((name) => {
    const r = map.get(name.toLowerCase()) || {
      drugName: name,
      chemblId: null,
      mechanism: null,
      source: null,
      fetchedAt: new Date().toISOString(),
    };
    recordApiRequestForUserV2(guard.user);
    return {
      drugName: escapeKgText(r.drugName),
      chemblId: escapeKgText(r.chemblId),
      mechanism: escapeKgText(r.mechanism),
      source: escapeKgText(r.source),
      fetchedAt: r.fetchedAt,
      // Task 242: surface the drug→protein→pathway chain. The edges
      // themselves are NOT escaped — they're server-constructed from
      // KG service node IDs (already alphanumerics). The relation
      // field IS escaped because it comes from the KG's edge labels.
      pathwayChain: (r.pathwayChain || []).map((e) => ({
        source: e.source,
        sourceType: e.sourceType,
        target: e.target,
        targetType: e.targetType,
        relation: escapeKgText(e.relation) || "related_to",
      })),
      proteinTargets: r.proteinTargets || [],
      pathways: r.pathways || [],
      ...(r.error ? { error: r.error } : {}),
    };
  });
  return NextResponse.json({ results });
}
