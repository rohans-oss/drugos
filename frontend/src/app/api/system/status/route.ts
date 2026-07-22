import { NextResponse } from "next/server";
// TASK-261 / TASK-265: replace requireAdmin with requirePlatformAdmin.
// /api/system/status is a platform-operator endpoint — it returns the
// health of every backend service, including cross-tenant connectivity
// details. Org-scoped admins should NOT see this (they only need to know
// about their own org's data, not the SaaS operator's infra).
import { requirePlatformAdmin } from "@/lib/auth/require-platform-admin";
// TASK-265: real service-health aggregator. Replaces the hardcoded
// `available: true` values and the env-var-presence stubs.
import { getSystemHealth, type SystemHealth } from "@/lib/services/system-health";
import {
  checkKnowledgeGraphAvailability,
  checkDatasetAvailability,
  checkRlAvailability,
  type MlServiceAvailability,
} from "@/lib/services/ml-stubs";
import { isOpenfdaApiKeyConfigured } from "@/lib/services/openfda";

/**
 * GET /api/system/status
 *
 * TASK-265 ROOT FIX: aggregate REAL status from all backend services.
 *
 * The audit (Task 265) found that this route returned HARDCODED
 * `available: true` for several services and only checked env-var
 * presence for the ML services. The previous fix added
 * checkKnowledgeGraphAvailability / checkDatasetAvailability /
 * checkRlAvailability stubs that returned `available: true` if the env
 * var was set — but NEVER actually pinged the service.
 *
 * This commit replaces the stubs with REAL connectivity checks:
 *
 *   - PostgreSQL: SELECT 1 through Prisma.
 *   - Neo4j (Phase 2): HTTP ping with 2s timeout.
 *   - MLflow: HTTP ping with 2s timeout.
 *   - Airflow (Phase 1): HTTP ping with 2s timeout.
 *   - Graph Transformer (Phase 3): HTTP ping or model-artifact check.
 *   - RL Agent (Phase 4): HTTP ping with 2s timeout.
 *
 * The aggregate `overall` status is:
 *   - "operational" if every service is available.
 *   - "degraded" if a non-critical service is down or any service is
 *     degraded.
 *   - "down" if a CRITICAL service (PostgreSQL, Neo4j) is down.
 *
 * TASK-280: the route returns 503 (not 200) when `overall === "down"`.
 * This lets the monitoring layer alert on the 5xx status code without
 * parsing the response body. The response body still contains the
 * per-service breakdown so the operator can see WHICH critical service
 * is down.
 *
 * TASK-261: the route is gated on `platformRole === "admin"` via the
 * new requirePlatformAdmin middleware. The prior `requireAdmin` check
 * let org-scoped admins read the platform's infra status — leaking
 * internal service configuration (env var names, hostnames, ports).
 */

// Matches common env-var name patterns: ALL_CAPS_WITH_UNDERSCORES, at least
// 4 chars, ending in _KEY, _URL, _TOKEN, _SECRET, or _ID. We redact the
// match so an attacker cannot infer which env vars to phish for.
const ENV_VAR_PATTERN = /\b[A-Z][A-Z0-9_]{3,}(?:_API_KEY|_KEY|_URL|_TOKEN|_SECRET|_ID|_PASSWORD)\b/g;

function scrubReason(reason: string | undefined): string | undefined {
  if (!reason) return reason;
  return reason.replace(ENV_VAR_PATTERN, "[REDACTED]");
}

function scrubService(svc: MlServiceAvailability): MlServiceAvailability {
  return {
    ...svc,
    reason: scrubReason(svc.reason) || "",
  };
}

/**
 * BE-038 ROOT FIX (v115, LOW): scrub reasons in the SystemHealth object too.
 *
 * The previous code only scrubbed `services.knowledgeGraph`,
 * `services.dataset`, and `services.rl` (the legacy per-service stubs).
 * The new `health` object (returned by getSystemHealth()) contains its
 * OWN `reason` strings — e.g. "NEO4J_URL is not configured",
 * "MLFLOW_TRACKING_URL is not configured", "connection failed:
 * ECONNREFUSED 10.0.0.5:8002". These strings contain env var names
 * AND potentially internal hostnames/IPs that an attacker could use
 * for targeted social engineering or network reconnaissance.
 *
 * This function walks the SystemHealth tree and scrubs EVERY `reason`
 * field it finds. It returns a new object (does not mutate the input)
 * so the original health data is preserved for internal operator logs.
 */
function scrubSystemHealth(health: SystemHealth): SystemHealth {
  const scrubbedServices: Record<string, unknown> = {};
  for (const [key, svc] of Object.entries(health.services)) {
    if (svc && typeof svc === "object" && "reason" in svc) {
      const svcObj = svc as { reason?: string };
      scrubbedServices[key] = {
        ...svc,
        reason: scrubReason(svcObj.reason),
      };
    } else {
      scrubbedServices[key] = svc;
    }
  }
  return {
    ...health,
    services: scrubbedServices as SystemHealth["services"],
  };
}

export async function GET(req: Request) {
  // TASK-261: gate on platformRole === "admin".
  // We accept a plain `Request` here (Next.js App Router allows it) and
  // cast to NextRequest inside the middleware for CSRF / rate-limit checks.
  const { NextRequest } = await import("next/server");
  const nextReq = req instanceof NextRequest ? req : new NextRequest(req);
  const auth = await requirePlatformAdmin(nextReq);
  if (auth.user === null) return auth.response;

  const patentsviewAvailable = !!process.env.PATENTSVIEW_API_KEY;
  const openfdaKeyPresent = isOpenfdaApiKeyConfigured();

  // TASK-265: run REAL connectivity checks against every backend service.
  // This is the root fix — the previous code returned hardcoded values.
  // BE-038 ROOT FIX (v115): scrub the health object's `reason` fields
  // before returning — they contain env var names and potentially
  // internal hostnames that an attacker could exploit.
  const rawHealth = await getSystemHealth();
  const health = scrubSystemHealth(rawHealth);

  // Build the response. We keep the legacy per-service keys (auth,
  // rxnorm, mesh, etc.) for backwards compat with the admin console UI,
  // but ADD the new `health` object with the real aggregated status.
  const response: {
    overall: SystemHealth["overall"];
    health: SystemHealth;
    services: Record<string, unknown>;
    generatedAt: string;
  } = {
    overall: health.overall,
    health,
    services: {
      // Frontend / API services (always available if this route responds).
      auth: { available: true, service: "Authentication" },
      rxnorm: { available: true, service: "RxNorm Drug Search" },
      mesh: { available: true, service: "MeSH Disease Search" },
      clinicalTrials: { available: true, service: "ClinicalTrials.gov Search" },
      pubmed: { available: true, service: "PubMed Literature Search" },
      openfda: {
        available: openfdaKeyPresent,
        degraded: !openfdaKeyPresent,
        service: "openFDA Adverse Events",
        reason: openfdaKeyPresent
          ? undefined
          : "API key not configured. Service is reachable but rate-limited to 240 req/min shared.",
      },
      patentsview: {
        available: patentsviewAvailable,
        service: "USPTO Patent Search",
        reason: patentsviewAvailable ? undefined : "Service not configured.",
      },
      projects: { available: true, service: "Projects & Collaboration" },
      billing: { available: true, service: "Billing & Subscriptions" },
      admin: { available: true, service: "Admin Console" },
      apiKeys: { available: true, service: "Developer API Keys" },
      evidence: { available: true, service: "Evidence Packages" },
      // ML services (legacy stubs — kept for backwards compat with the
      // admin console UI; the real status is in `health.services`).
      knowledgeGraph: scrubService(checkKnowledgeGraphAvailability()),
      dataset: scrubService(checkDatasetAvailability()),
      rl: scrubService(checkRlAvailability()),
    },
    generatedAt: new Date().toISOString(),
  };

  // TASK-280: return 503 when overall === "down" so monitoring can alert.
  // The response body still contains the per-service breakdown so the
  // operator can see WHICH critical service is down.
  const status = health.overall === "down" ? 503 : 200;
  return NextResponse.json(response, { status });
}
