import { NextRequest, NextResponse } from "next/server";
import { requireAuth, requireRole, badRequest, writeAuditLog } from "@/lib/api-helpers";
// Issue 227 ROOT FIX: replace the subprocess shell-out to
// scripts/hypothesis_writeback.py with an HTTP proxy to
// RL_SERVICE_URL/validate. The new /validate endpoint on rl/service.py
// calls phase4.writeback.write_validated_hypothesis() directly.
//
// The previous version spawned a Python subprocess per request, which:
//   1. Failed in deployments where the frontend container doesn't have
//      Python installed (e.g., Vercel, Netlify, Cloudflare Pages).
//   2. Required the scripts/hypothesis_writeback.py file to exist at
//      a path resolved via process.cwd() — which is frontend/ in dev,
//      so the script was never found.
//   3. Could not be load-balanced — each frontend instance spawned its
//      own subprocess, with no shared state.
//
// The HTTP path goes through the shared mlFetch client (Issue 234) with
// timeout, retry, and structured error normalization. The Python
// service owns the writeback logic (Phase 1 CSV append, Phase 2 Neo4j
// edge, Phase 3 retrain trigger).
import { mlFetch, resolveServiceUrl, buildServiceUrl, MlServiceError } from "@/lib/http-client";
import {
  RlValidateRequestSchema,
  RlValidateResponseSchema,
  type RlValidateResponse,
  validateMlResponse,
} from "@/lib/ml-contracts";
// TASK-268: notification trigger for hypothesis validation completion.
// Fires AFTER the writeback succeeds — notifies the submitter + the
// org's PIs so they can review the validation result.
import { notifyHypothesisValidationComplete } from "@/lib/services/notifications";

const SERVICE_NAME = "phase4_rl_validate";

/**
 * POST /api/hypothesis/validate
 * Body: {
 *   drug: string,
 *   disease: string,
 *   outcome: "validated_positive" | "validated_negative" | "validated_toxic" | "invalidated",
 *   validationStudyId?: string,    // e.g., NCT number
 *   notes?: string,
 *   originalGtScore?: number,
 *   originalRlRank?: number,
 * }
 *
 * RT-010 ROOT FIX (Team Member 17): this route implements the data
 * flywheel writeback (project docx Section 10, step 3):
 *
 *   "Pharma partner validates a hypothesis (in wet lab or clinical
 *    study). This validation result is fed back into the platform as
 *    a new labeled data point."
 *
 * Issue 227 ROOT FIX: the route now proxies to RL_SERVICE_URL/validate
 * via HTTP instead of spawning a subprocess. The Python service's new
 * /validate endpoint (added to rl/service.py) calls
 * phase4.writeback.write_validated_hypothesis() which writes to:
 *   - Phase 1: appends to phase1/processed_data/validated_hypotheses.csv
 *   - Phase 2: adds a VALIDATED_TREATS edge to Neo4j (when available)
 *   - Phase 3: appends to graph_transformer/retrain_triggered.json
 *
 * Security: only data-scientist / pi / developer / admin / owner roles
 * can call this route. Viewers cannot. The validated_by field is set
 * AUTOMATICALLY from the authenticated user's identity — the caller
 * cannot impersonate another partner.
 */
export async function POST(req: NextRequest) {
  const auth = await requireAuth();
  if (auth.user === null) return auth.response;

  // RT-010: only analytics roles can submit validation results.
  const roleCheck = await requireRole(auth.user, "data-scientist", "pi", "developer");
  if (roleCheck.user === null) return roleCheck.response;

  let body: {
    drug?: string;
    disease?: string;
    outcome?: string;
    validationStudyId?: string;
    notes?: string;
    originalGtScore?: number;
    originalRlRank?: number;
  };
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "bad_request", message: "Invalid JSON" }, { status: 400 });
  }

  const drug = (body.drug || "").trim();
  const disease = (body.disease || "").trim();
  const outcome = body.outcome;

  if (!drug || !disease) {
    return badRequest("drug and disease are required");
  }
  const validOutcomes = ["validated_positive", "validated_negative", "validated_toxic", "invalidated"];
  if (!outcome || !validOutcomes.includes(outcome)) {
    return badRequest(`outcome must be one of: ${validOutcomes.join(", ")}`);
  }

  // RT-010: validated_by is set from the authenticated user's identity
  // (cannot be spoofed by the caller). The user's orgId or userId is
  // the partner identifier — this gives us an audit trail.
  const validatedBy = auth.user.orgId || auth.user.userId;

  // Issue 227: resolve RL_SERVICE_URL. If not set, return 503 with a
  // clear message — we cannot fall back to a subprocess (the script
  // doesn't exist at the resolved path).
  const baseUrl = resolveServiceUrl("RL_SERVICE_URL");
  if (!baseUrl) {
    return NextResponse.json(
      {
        error: "service_not_deployed",
        service: "Phase 4 RL Hypothesis Ranker",
        reason:
          "RL_SERVICE_URL is not set. The hypothesis writeback requires " +
          "the Phase 4 RL service (rl/service.py) to be running with the " +
          "new /validate endpoint. Start it with `python rl/service.py` " +
          "and set RL_SERVICE_URL=http://localhost:8004 in frontend/.env.local. " +
          "Issue 227 ROOT FIX: this route NO LONGER shells out to " +
          "scripts/hypothesis_writeback.py (which does not exist at the " +
          "resolved path).",
        documentation:
          "See Phase 4 of the build plan (RL-Driven Hypothesis Ranking).",
      },
      { status: 503 }
    );
  }

  // Build the request payload — validate against the contract schema.
  const payload = {
    drug,
    disease,
    outcome,
    validated_by: validatedBy,
    validation_study_id: body.validationStudyId,
    notes: body.notes,
    original_gt_score: body.originalGtScore,
    original_rl_rank: body.originalRlRank,
  };

  // Validate the request payload against the contract. This catches
  // type errors (e.g., originalGtScore as string) before sending.
  const payloadValidation = RlValidateRequestSchema.safeParse(payload);
  if (!payloadValidation.success) {
    return badRequest(
      `Invalid request: ${payloadValidation.error.issues[0]?.message ?? "validation failed"}`
    );
  }

  // Issue 227: proxy to RL_SERVICE_URL/validate via the shared HTTP client.
  const url = buildServiceUrl(baseUrl, "/validate");
  const result = await mlFetch<unknown>(url, {
    service: SERVICE_NAME,
    method: "POST",
    body: payloadValidation.data,
    timeoutMs: 30_000, // writeback touches 3 phases — may take time
    maxRetries: 0, // writeback is NOT idempotent (appends to CSV) — don't retry
  });

  if (!result.ok) {
    const err = result.error as MlServiceError;
    // 4xx = bad request from the service. Surface the message.
    if (err.httpStatus >= 400 && err.httpStatus < 500) {
      return NextResponse.json(
        {
          error: "validate_failed",
          message: `RL service rejected validation request (${err.httpStatus}): ${err.message}`,
          service_error: err.toJSON(),
        },
        { status: err.httpStatus === 400 ? 400 : 502 }
      );
    }
    // 5xx / network — surface as 502.
    return NextResponse.json(
      {
        error: "validate_failed",
        message: `RL service validate endpoint failed: ${err.message}`,
        service_error: err.toJSON(),
      },
      { status: 502 }
    );
  }

  // Validate the response against the contract.
  let validated: RlValidateResponse;
  try {
    validated = validateMlResponse(
      SERVICE_NAME,
      "/validate",
      RlValidateResponseSchema,
      result.body,
    );
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    return NextResponse.json(
      {
        error: "validate_contract_violation",
        message: `RL service /validate response did not match expected schema: ${msg}`,
      },
      { status: 502 }
    );
  }

  await writeAuditLog({
    user: auth.user,
    action: "hypothesis_validate",
    resource: `hypothesis:${drug}:${disease}`,
    metadata: {
      outcome,
      validatedBy,
      validationStudyId: body.validationStudyId,
      phase2Neo4jWritten: validated.writeback?.phase2_neo4j_written,
    },
  });

  // TASK-268: fire the notification trigger. NON-BLOCKING — the
  // writeback already succeeded; a notification failure must not roll
  // it back. The helper notifies (a) the submitter and (b) the org's
  // PIs (principal investigators) who need to review the validation.
  if (auth.user.orgId) {
    await notifyHypothesisValidationComplete(
      auth.user.userId,
      auth.user.orgId,
      drug,
      disease,
      outcome,
    ).catch((e) => {
      // Non-critical — the validation already wrote back to all 3 phases.
      console.error("[HYPOTHESIS-VALIDATE] notification failed:", e);
    });
  }

  return NextResponse.json({
    ok: true,
    writeback: validated.writeback,
    message:
      "Hypothesis validation written back to Phase 1 (CSV), Phase 2 (Neo4j edge), and Phase 3 (retrain trigger).",
  });
}
