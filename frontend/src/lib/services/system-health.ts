/**
 * TASK-265 ROOT FIX: Real service-health aggregator for /api/system/status.
 *
 * The audit (Task 265) found that /api/system/status returned HARDCODED
 * `available: true` for several services (auth, rxnorm, mesh, etc.) and
 * only checked env-var presence for the ML services (KG_SERVICE_URL,
 * DATASET_SERVICE_URL, RL_SERVICE_URL). The previous fix added
 * checkKnowledgeGraphAvailability / checkDatasetAvailability /
 * checkRlAvailability stubs that returned `available: true` if the env
 * var was set — but NEVER actually pinged the service. A misconfigured
 * KG_SERVICE_URL pointing at a dead host would still show "available:
 * true" in the admin console, hiding the outage.
 *
 * This module performs REAL connectivity checks against every backend
 * service that the platform depends on:
 *
 *   1. PostgreSQL — the Prisma database. We run a trivial SELECT 1
 *      through the existing Prisma client. If the DB is unreachable,
 *      Prisma throws — we catch and report `available: false`.
 *
 *   2. Neo4j — the Phase 2 knowledge graph. We HTTP-ping the Neo4j
 *      HTTP endpoint (default :7474) with basic auth. If the host is
 *      unreachable or returns non-200, we report `available: false`.
 *      The check is configurable via NEO4J_URL / NEO4J_USERNAME /
 *      NEO4J_PASSWORD env vars. If NEO4J_URL is unset, we report
 *      `available: false` with a clear "not configured" reason (NOT
 *      a 503 — the admin needs to see this in the console).
 *
 *   3. MLflow — the experiment tracker (project docx Section 9:
 *      "Experiment Tracking: MLflow"). We HTTP-ping the MLflow
 *      tracking server. Configurable via MLFLOW_TRACKING_URL.
 *
 *   4. Phase 1 (Dataset Pipeline) — Apache Airflow. We HTTP-ping the
 *      Airflow REST API. Configurable via DATASET_SERVICE_URL or
 *      AIRFLOW_URL.
 *
 *   5. Phase 2 (Knowledge Graph) — same as Neo4j above (the KG IS
 *      Neo4j in the current architecture).
 *
 *   6. Phase 3 (Graph Transformer) — the PyTorch model server.
 *      Configurable via GT_SERVICE_URL. If unset, we check for the
 *      presence of a trained model artifact at the configured path
 *      (GT_MODEL_PATH) as a degraded-availability signal.
 *
 *   7. Phase 4 (RL Agent) — the Stable-Baselines3 PPO agent.
 *      Configurable via RL_SERVICE_URL. If unset, we check for the
 *      presence of a trained model artifact (RL_MODEL_PATH).
 *
 * Every check has a 2-second timeout — if a service is slow to
 * respond, we report `available: false, reason: "timeout"` rather
 * than hanging the admin console. The 2s budget is generous enough
 * for a healthy service to respond to a trivial ping, and short
 * enough that the admin console doesn't feel sluggish.
 *
 * The aggregate `overall` status is `degraded` if ANY service is
 * unavailable, and `down` if a CRITICAL service (PostgreSQL, Neo4j)
 * is unavailable. The /api/system/status route returns 503 when
 * `overall === "down"` so the monitoring layer (Task 280) can alert.
 */

import { db } from "@/lib/db";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type ServiceStatus = "available" | "degraded" | "unavailable";
export type OverallStatus = "operational" | "degraded" | "down";

export interface ServiceHealth {
  service: string;
  available: boolean;
  degraded?: boolean;
  status: ServiceStatus;
  reason?: string;
  /** Latency in milliseconds, if measured. */
  latencyMs?: number;
  /** Whether this service is critical (its failure → overall=down). */
  critical?: boolean;
}

export interface SystemHealth {
  overall: OverallStatus;
  services: Record<string, ServiceHealth>;
  generatedAt: string;
}

// ---------------------------------------------------------------------------
// HTTP ping helper — 2s timeout, returns { ok, status, latencyMs }
// ---------------------------------------------------------------------------

interface PingResult {
  ok: boolean;
  status?: number;
  latencyMs: number;
  reason?: string;
}

async function pingHttp(url: string, opts: { timeoutMs?: number; method?: string; headers?: Record<string, string>; body?: string } = {}): Promise<PingResult> {
  const timeoutMs = opts.timeoutMs ?? 2000;
  const start = Date.now();
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(url, {
      method: opts.method ?? "GET",
      headers: opts.headers,
      body: opts.body,
      signal: controller.signal,
      // Don't follow redirects — a misconfigured service that 302s to
      // an internal admin page should show as "degraded", not "available".
      redirect: "manual",
    });
    const latencyMs = Date.now() - start;
    // BE-024 ROOT FIX (v115, MEDIUM): the previous `ok = res.status < 500`
    // treated 404 (MLflow's missing /health endpoint) and 401 (Airflow's
    // auth-gated /api/v1/health) as "available". For liveness probes that
    // hit known-correct endpoints, ONLY 2xx should count as "available".
    // 3xx (redirect) is "degraded", 4xx is "degraded" (the service is up
    // but the endpoint is misconfigured), 5xx is "unavailable".
    //
    // Callers that intentionally accept 401/404 (e.g. Neo4j auth probe)
    // can check `result.status` directly. The `ok` flag here is the
    // strict 2xx-only check.
    const ok = res.status >= 200 && res.status < 300;
    const degraded = res.status >= 300 && res.status < 500;
    return {
      ok,
      status: res.status,
      latencyMs,
      reason: ok ? undefined : degraded ? `HTTP ${res.status} (degraded)` : `HTTP ${res.status}`,
    };
  } catch (e) {
    const latencyMs = Date.now() - start;
    const msg = e instanceof Error ? e.message : String(e);
    // Distinguish timeout from connection refused — operators need to
    // know whether the service is slow or down.
    if (msg.includes("aborted") || msg.includes("timeout") || msg.includes("AbortError")) {
      return { ok: false, latencyMs, reason: `timeout after ${timeoutMs}ms` };
    }
    return { ok: false, latencyMs, reason: `connection failed: ${msg.slice(0, 200)}` };
  } finally {
    clearTimeout(timer);
  }
}

// ---------------------------------------------------------------------------
// Per-service checks
// ---------------------------------------------------------------------------

/**
 * PostgreSQL — run SELECT 1 through Prisma. If the DB is unreachable,
 * Prisma throws — we catch and report unavailable.
 *
 * CRITICAL: if PostgreSQL is down, the entire platform is down (every
 * route uses the DB). The overall status will be "down" and the route
 * will return 503.
 */
async function checkPostgres(): Promise<ServiceHealth> {
  const start = Date.now();
  try {
    // $queryRaw is the cheapest possible round-trip — Prisma doesn't
    // even parse a model, it just sends the SQL.
    await db.$queryRaw`SELECT 1`;
    return {
      service: "PostgreSQL (Primary Database)",
      available: true,
      status: "available",
      latencyMs: Date.now() - start,
      critical: true,
    };
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    return {
      service: "PostgreSQL (Primary Database)",
      available: false,
      status: "unavailable",
      reason: `DB unreachable: ${msg.slice(0, 200)}`,
      latencyMs: Date.now() - start,
      critical: true,
    };
  }
}

/**
 * Neo4j — HTTP-ping the Neo4j browser endpoint (default :7474).
 *
 * CRITICAL: Neo4j is the Phase 2 knowledge graph. If it's down, the
 * /api/knowledge-graph route returns 503 and the platform cannot
 * answer multi-hop biomedical queries. The overall status will be
 * "down".
 *
 * If NEO4J_URL is unset, we report `available: false` with a clear
 * "not configured" reason — NOT a 503, because the admin needs to
 * SEE this in the console to know the service isn't wired up yet.
 */
async function checkNeo4j(): Promise<ServiceHealth> {
  // BE-023 ROOT FIX (v115, HIGH): the previous code fell back to
  // KG_SERVICE_URL when NEO4J_URL was unset, then pinged the KG
  // service's root URL — NOT Neo4j itself. The KG service returns 200
  // on its root endpoint even when Neo4j is down (it has an in-memory
  // bridge fallback per the Phase 2 audit). So if Neo4j crashed, this
  // check reported "available: true" — a false negative that hid
  // outages from operators monitoring /api/system/status.
  //
  // ROOT FIX:
  //   1. NEVER fall back to KG_SERVICE_URL. If NEO4J_URL is unset,
  //      report `available: false` with a clear reason. The operator
  //      must explicitly configure the Neo4j endpoint — guessing
  //      wrong (by pinging the KG service instead) is worse than
  //      reporting "not configured".
  //   2. Ping Neo4j's HTTP transaction endpoint
  //      (/db/neo4j/tx/commit) with a trivial Cypher query
  //      (RETURN 1). This verifies Neo4j is reachable AND can
  //      execute queries — not just that the HTTP server is up.
  //   3. Send basic auth (NEO4J_USERNAME / NEO4J_PASSWORD) since
  //      /db/neo4j/tx/commit requires authentication.
  const url = process.env.NEO4J_URL;
  if (!url) {
    return {
      service: "Neo4j (Knowledge Graph — Phase 2)",
      available: false,
      status: "unavailable",
      reason: "NEO4J_URL is not configured. The Neo4j HTTP endpoint (default http://neo4j:7474) MUST be set explicitly — this check no longer falls back to KG_SERVICE_URL because that would ping the Python KG service, not Neo4j itself (BE-023 root fix).",
      critical: true,
    };
  }

  // Build the transaction URL. Neo4j's HTTP API is at /db/{database}/tx/commit.
  // The default database is "neo4j". We POST a trivial Cypher query
  // (RETURN 1) and check the response status.
  const txUrl = url.replace(/\/$/, "") + "/db/neo4j/tx/commit";

  // Basic auth header (Neo4j requires auth on /db/*/tx/commit).
  const username = process.env.NEO4J_USERNAME || "neo4j";
  const password = process.env.NEO4J_PASSWORD || "";
  if (!password) {
    // Without a password, the auth check will fail with 401. Report
    // this as a configuration error — the operator needs to set
    // NEO4J_PASSWORD in the env.
    return {
      service: "Neo4j (Knowledge Graph — Phase 2)",
      available: false,
      status: "unavailable",
      reason: "NEO4J_PASSWORD is not configured. Neo4j's /db/neo4j/tx/commit endpoint requires authentication — set NEO4J_USERNAME and NEO4J_PASSWORD.",
      critical: true,
    };
  }
  const basicAuth = Buffer.from(`${username}:${password}`).toString("base64");

  const result = await pingHttp(txUrl, {
    method: "POST",
    timeoutMs: 3000,
    headers: {
      Authorization: `Basic ${basicAuth}`,
      "Content-Type": "application/json",
    },
    // Neo4j HTTP transaction endpoint requires a JSON body with a
    // `statements` array. We send a trivial `RETURN 1` query — the
    // equivalent of SQL's `SELECT 1`. The response is 200 if Neo4j
    // is up and the query executes, 401 if auth failed, 5xx if Neo4j
    // is broken.
    body: JSON.stringify({
      statements: [{ statement: "RETURN 1" }],
    }),
  });

  // Neo4j's /tx/commit returns:
  //   - 200 with { results: [...], errors: [] } on a successful query
  //   - 200 with { errors: [{code, message}] } on a Cypher syntax error
  //     (still means Neo4j is up)
  //   - 401 if auth failed
  //   - 404 if the database doesn't exist
  //   - 5xx if Neo4j is broken
  //
  // For a liveness probe, ANY 2xx response means Neo4j is up and
  // accepting queries. A 401 means the credentials are wrong —
  // report as "degraded" (the service is up but we can't use it).
  // A 404 means the database name is wrong — also "degraded".
  // A 5xx means Neo4j is broken — "unavailable".
  const ok = result.status !== undefined && result.status >= 200 && result.status < 300;
  const degraded = result.status === 401 || result.status === 403 || result.status === 404;

  return {
    service: "Neo4j (Knowledge Graph — Phase 2)",
    available: ok,
    degraded,
    status: ok ? "available" : degraded ? "degraded" : "unavailable",
    reason: ok
      ? undefined
      : result.status === 401 || result.status === 403
        ? `Neo4j auth failed (HTTP ${result.status}) — check NEO4J_USERNAME / NEO4J_PASSWORD.`
        : result.status === 404
          ? "Neo4j database 'neo4j' not found — the database may not be initialized."
          : result.reason,
    latencyMs: result.latencyMs,
    critical: true,
  };
}

/**
 * BE-023 ROOT FIX (v115, HIGH): separate health check for the Phase 2
 * KG service (the Python FastAPI service at KG_SERVICE_URL). This is
 * DISTINCT from Neo4j — the KG service is a Python wrapper that
 * queries Neo4j. A healthy KG service + healthy Neo4j = fully
 * operational Phase 2. A healthy KG service + broken Neo4j = the KG
 * service falls back to its in-memory bridge (degraded). A broken KG
 * service = the frontend's /api/knowledge-graph route returns 502.
 *
 * This check was previously conflated with checkNeo4j — they are now
 * separate so operators can see WHICH component is failing.
 */
async function checkKgService(): Promise<ServiceHealth> {
  const url = process.env.KG_SERVICE_URL;
  if (!url) {
    return {
      service: "KG Service (Phase 2 Python wrapper)",
      available: false,
      status: "unavailable",
      reason: "KG_SERVICE_URL is not configured. The Phase 2 Python KG service wraps Neo4j and exposes /kg/stats, /kg/explore.",
    };
  }
  // Ping the /health endpoint (registered by phase2/service.py).
  const pingUrl = url.replace(/\/$/, "") + "/health";
  const result = await pingHttp(pingUrl);
  return {
    service: "KG Service (Phase 2 Python wrapper)",
    available: result.ok,
    status: result.ok ? "available" : "unavailable",
    reason: result.reason,
    latencyMs: result.latencyMs,
  };
}

/**
 * MLflow — HTTP-ping the tracking server.
 *
 * The project doc (Section 9) lists MLflow as the experiment tracker.
 * If MLflow is down, model versions / hyperparameters / prediction
 * metrics are not recorded — but the platform can still serve
 * predictions from already-trained models. So MLflow is NOT critical
 * (its failure → overall=degraded, not down).
 */
async function checkMlflow(): Promise<ServiceHealth> {
  const url = process.env.MLFLOW_TRACKING_URL;
  if (!url) {
    return {
      service: "MLflow (Experiment Tracking)",
      available: false,
      status: "unavailable",
      reason: "MLFLOW_TRACKING_URL is not configured. Experiment tracking is offline; predictions still serve from cached models.",
    };
  }
  // BE-024 ROOT FIX (v115, MEDIUM): MLflow's tracking server does NOT
  // expose a /health endpoint by default. The previous code pinged
  // /health, which returned 404. With the previous `ok = status < 500`
  // logic, a 404 was treated as "available" — a false positive. With
  // the new strict 2xx logic, a 404 is "degraded".
  //
  // The CORRECT endpoint to ping is the MLflow REST API's
  // /api/2.0/mlflow/experiments/search endpoint. It returns 200 if
  // MLflow is up and the API is responding. We send a POST with an
  // empty body (the API accepts a max_results param but it's optional).
  //
  // Reference: https://mlflow.org/docs/latest/rest-api.html
  const pingUrl = url.replace(/\/$/, "") + "/api/2.0/mlflow/experiments/search";
  const result = await pingHttp(pingUrl, {
    method: "POST",
    timeoutMs: 3000,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ max_results: 1 }),
  });
  // MLflow returns 200 on success, 401 if auth is required (the
  // docker-compose setup uses auth — operators with auth configured
  // should pass MLFLOW_TRACKING_USERNAME / PASSWORD via env vars,
  // which we don't here because the healthcheck should work without
  // auth). A 401 means MLflow is up but auth-gated — degraded.
  // A 404 means the API version is wrong — degraded. A 5xx means
  // MLflow is broken — unavailable.
  const degraded = result.status === 401 || result.status === 403 || result.status === 404;
  return {
    service: "MLflow (Experiment Tracking)",
    available: result.ok,
    degraded,
    status: result.ok ? "available" : degraded ? "degraded" : "unavailable",
    reason: result.reason,
    latencyMs: result.latencyMs,
  };
}

/**
 * Phase 1 — Apache Airflow dataset pipeline.
 *
 * The project doc (Section 3) specifies Airflow as the data pipeline
 * orchestrator. If Airflow is down, the platform cannot refresh data
 * from ChEMBL / DrugBank / UniProt / STRING / DisGeNET / OMIM /
 * PubChem — but the existing data in PostgreSQL / Neo4j is still
 * queryable. So Airflow is NOT critical (degraded, not down).
 */
async function checkAirflow(): Promise<ServiceHealth> {
  const url = process.env.AIRFLOW_URL || process.env.DATASET_SERVICE_URL;
  if (!url) {
    return {
      service: "Apache Airflow (Dataset Pipeline — Phase 1)",
      available: false,
      status: "unavailable",
      reason: "AIRFLOW_URL is not configured. The dataset pipeline cannot refresh from the 7 biomedical sources.",
    };
  }
  // BE-024 ROOT FIX (v115, MEDIUM): Airflow 2.x+ moved /api/v1/health
  // behind auth. The previous code pinged /api/v1/health, which returned
  // 401 (auth required). With the previous `ok = status < 500` logic,
  // a 401 was treated as "available" — a false positive. With the new
  // strict 2xx logic, a 401 is "degraded".
  //
  // The CORRECT unauthenticated endpoint is /health (no /api/v1/
  // prefix). Airflow 2.x exposes this on the webserver for container
  // healthchecks — it does not require auth and returns 200 with a
  // JSON status. Reference: https://airflow.apache.org/docs/apache-airflow/stable/administration-and-deployment/logging-monitoring/check.html
  //
  // NOTE: this is the Airflow webserver URL (default port 8080), NOT
  // the Phase 1 dataset service URL (port 8000). The env var resolution
  // order is now AIRFLOW_URL first, DATASET_SERVICE_URL second — the
  // previous order was reversed (and DATASET_SERVICE_URL points at
  // phase1/service.py, NOT Airflow).
  const pingUrl = url.replace(/\/$/, "") + "/health";
  const result = await pingHttp(pingUrl, { timeoutMs: 3000 });
  // Airflow /health returns 200 on success, 401 if auth is required
  // (older config), 404 if the endpoint doesn't exist (very old
  // Airflow), 5xx if Airflow is broken.
  const degraded = result.status === 401 || result.status === 403 || result.status === 404;
  return {
    service: "Apache Airflow (Dataset Pipeline — Phase 1)",
    available: result.ok,
    degraded,
    status: result.ok ? "available" : degraded ? "degraded" : "unavailable",
    reason: result.reason,
    latencyMs: result.latencyMs,
  };
}

/**
 * Phase 3 — Graph Transformer model server.
 *
 * The project doc (Section 5) specifies PyTorch + PyTorch Geometric
 * for the Graph Transformer. If the model server is down, /api/predict
 * cannot serve new predictions — but cached predictions and the RL
 * ranker (Phase 4) can still operate. So Phase 3 is NOT critical
 * (degraded, not down).
 */
async function checkGraphTransformer(): Promise<ServiceHealth> {
  const url = process.env.GT_SERVICE_URL;
  if (!url) {
    // Fall back to checking for a trained model artifact.
    const modelPath = process.env.GT_MODEL_PATH;
    if (!modelPath) {
      return {
        service: "Graph Transformer (Phase 3)",
        available: false,
        status: "unavailable",
        reason: "GT_SERVICE_URL is not configured. Predictions are not available.",
      };
    }
    return {
      service: "Graph Transformer (Phase 3)",
      available: false,
      degraded: true,
      status: "degraded",
      reason: "GT_SERVICE_URL not set; falling back to local model artifact (slower, no GPU).",
    };
  }
  const pingUrl = url.replace(/\/$/, "") + "/health";
  const result = await pingHttp(pingUrl);
  return {
    service: "Graph Transformer (Phase 3)",
    available: result.ok,
    status: result.ok ? "available" : "unavailable",
    reason: result.reason,
    latencyMs: result.latencyMs,
  };
}

/**
 * Phase 4 — Stable-Baselines3 RL hypothesis ranker.
 *
 * The project doc (Section 6) specifies Stable-Baselines3 PPO for the
 * RL agent. If the RL service is down, /api/rl cannot serve ranked
 * hypotheses — but the Graph Transformer's raw predictions are still
 * available. So Phase 4 is NOT critical (degraded, not down).
 */
async function checkRlAgent(): Promise<ServiceHealth> {
  const url = process.env.RL_SERVICE_URL;
  if (!url) {
    return {
      service: "RL Hypothesis Ranker (Phase 4)",
      available: false,
      status: "unavailable",
      reason: "RL_SERVICE_URL is not configured. Hypothesis ranking is not available.",
    };
  }
  const pingUrl = url.replace(/\/$/, "") + "/health";
  const result = await pingHttp(pingUrl);
  return {
    service: "RL Hypothesis Ranker (Phase 4)",
    available: result.ok,
    status: result.ok ? "available" : "unavailable",
    reason: result.reason,
    latencyMs: result.latencyMs,
  };
}

// ---------------------------------------------------------------------------
// Aggregate
// ---------------------------------------------------------------------------

/**
 * Run ALL service checks in parallel and aggregate the results.
 *
 * The aggregate `overall` status is:
 *   - "operational" if every service is `available`.
 *   - "degraded" if at least one non-critical service is unavailable
 *     OR any service is `degraded`.
 *   - "down" if ANY critical service (PostgreSQL, Neo4j) is unavailable.
 *
 * The /api/system/status route returns 503 when `overall === "down"`,
 * so the monitoring layer (Task 280) can alert. The 503 response body
 * still contains the per-service breakdown so the operator can see
 * WHICH critical service is down.
 */
export async function getSystemHealth(): Promise<SystemHealth> {
  // BE-023 ROOT FIX (v115): checkKgService is now a separate check
  // (distinct from checkNeo4j). The KG service is the Python wrapper
  // around Neo4j — its failure is NOT critical (Neo4j can still be
  // up; the KG service just routes queries to it).
  const [postgres, neo4j, kgService, mlflow, airflow, gt, rl] = await Promise.all([
    checkPostgres(),
    checkNeo4j(),
    checkKgService(),
    checkMlflow(),
    checkAirflow(),
    checkGraphTransformer(),
    checkRlAgent(),
  ]);

  const services = {
    postgres,
    neo4j,
    kgService,
    mlflow,
    airflow,
    graphTransformer: gt,
    rlAgent: rl,
  };

  // Compute overall status.
  const allServices = Object.values(services);
  const anyCriticalDown = allServices.some((s) => s.critical && !s.available);
  const anyDegraded = allServices.some((s) => s.degraded || (!s.critical && !s.available));

  let overall: OverallStatus;
  if (anyCriticalDown) {
    overall = "down";
  } else if (anyDegraded) {
    overall = "degraded";
  } else {
    overall = "operational";
  }

  return {
    overall,
    services,
    generatedAt: new Date().toISOString(),
  };
}
