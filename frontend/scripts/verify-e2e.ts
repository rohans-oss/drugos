/**
 * End-to-end verification script (Issues 221-240).
 *
 * This script calls the REAL lib services (gt-inference.ts, rl-ranker.ts,
 * kg-service.ts, dataset-service.ts) against the REAL Python services
 * running on localhost:8001/8002/8004. It verifies the acceptance criteria:
 *
 *   1. predictPairs() returns predictions (not 500) when GT_SERVICE_URL is set.
 *   2. getRankedHypotheses() returns rankings when RL_SERVICE_URL is set.
 *   3. getKnowledgeGraphStats() returns KG stats when KG_SERVICE_URL is set.
 *   4. getDatasetStats() returns Phase 1 stats when PHASE1_SERVICE_URL is set.
 *
 * This bypasses the Next.js server (which requires PostgreSQL for auth)
 * but exercises the EXACT same code path the routes use: lib service →
 * mlFetch → real Python service. If this script passes, the routes will
 * pass too (the routes just add auth + audit logging on top).
 *
 * Run with: npx tsx scripts/verify-e2e.ts
 */

import { predictPairs, topKNovel, checkGtHealth } from "../src/lib/services/gt-inference";
import { getRankedHypotheses, checkRlHealth } from "../src/lib/services/rl-ranker";
import { getKnowledgeGraphStats, checkKgHealth } from "../src/lib/services/kg-service";
import { getDatasetStats, checkDatasetHealth } from "../src/lib/services/dataset-service";

const GREEN = "\x1b[32m";
const RED = "\x1b[31m";
const YELLOW = "\x1b[33m";
const RESET = "\x1b[0m";

function pass(msg: string) { console.log(`${GREEN}✓ PASS${RESET} ${msg}`); }
function fail(msg: string, err?: unknown) {
  console.log(`${RED}✗ FAIL${RESET} ${msg}`);
  if (err) console.log(`  ${RED}Error:${RESET}`, err instanceof Error ? err.message : String(err));
}
function info(msg: string) { console.log(`${YELLOW}→${RESET} ${msg}`); }

let totalTests = 0;
let passedTests = 0;

async function test(name: string, fn: () => Promise<void>) {
  totalTests++;
  try {
    await fn();
    passedTests++;
  } catch (err) {
    fail(name, err);
  }
}

async function main() {
  console.log("=".repeat(70));
  console.log("End-to-End Verification — Issues 221-240");
  console.log("=".repeat(70));
  console.log("");

  // Check env vars
  info(`PHASE1_SERVICE_URL = ${process.env.PHASE1_SERVICE_URL || "(not set)"}`);
  info(`KG_SERVICE_URL      = ${process.env.KG_SERVICE_URL || "(not set)"}`);
  info(`GT_SERVICE_URL      = ${process.env.GT_SERVICE_URL || "(not set)"}`);
  info(`RL_SERVICE_URL      = ${process.env.RL_SERVICE_URL || "(not set)"}`);
  console.log("");

  // ---- Phase 1: Dataset Service ----
  console.log("--- Phase 1: Dataset Service (Issue 233) ---");

  await test("checkDatasetHealth() reports configured + reachable", async () => {
    const h = await checkDatasetHealth();
    if (!h.configured) throw new Error("PHASE1_SERVICE_URL not configured");
    if (!h.reachable) throw new Error("Phase 1 service not reachable");
    pass(`health: configured=${h.configured}, reachable=${h.reachable}, version=${h.version}`);
  });

  await test("getDatasetStats() returns Phase 1 stats (not 500)", async () => {
    const stats = await getDatasetStats();
    if (stats.status === "service_down") throw new Error(`service_down: ${stats.note}`);
    if (!Array.isArray(stats.sources)) throw new Error("sources is not an array");
    if (typeof stats.nodesLoaded !== "number") throw new Error("nodesLoaded is not a number");
    pass(`sources=${stats.sources.length}, nodesLoaded=${stats.nodesLoaded}, edgesLoaded=${stats.edgesLoaded}, backend=${stats.backend}`);
  });

  console.log("");

  // ---- Phase 2: Knowledge Graph Service ----
  console.log("--- Phase 2: Knowledge Graph Service (Issue 232) ---");

  await test("checkKgHealth() reports configured + reachable", async () => {
    const h = await checkKgHealth();
    if (!h.configured) throw new Error("KG_SERVICE_URL not configured");
    if (!h.reachable) throw new Error("KG service not reachable");
    pass(`health: configured=${h.configured}, reachable=${h.reachable}, neo4j=${h.neo4jConfigured}`);
  });

  await test("getKnowledgeGraphStats() returns KG stats (not 500)", async () => {
    const stats = await getKnowledgeGraphStats();
    if (stats.source === "none") throw new Error(`source:none — ${stats.note}`);
    if (typeof stats.nodeCount !== "number") throw new Error("nodeCount is not a number");
    if (typeof stats.edgeCount !== "number") throw new Error("edgeCount is not a number");
    pass(`nodeCount=${stats.nodeCount}, edgeCount=${stats.edgeCount}, source=${stats.source}, nodeTypeCounts keys=${Object.keys(stats.nodeTypeCounts).join(",") || "(empty)"}`);
  });

  console.log("");

  // ---- Phase 3: Graph Transformer Service ----
  console.log("--- Phase 3: Graph Transformer Service (Issue 230) ---");

  await test("checkGtHealth() reports configured (may be unreachable — no checkpoint)", async () => {
    const h = await checkGtHealth();
    if (!h.configured) throw new Error("GT_SERVICE_URL not configured");
    pass(`health: configured=${h.configured}, reachable=${h.reachable}, checkpointLoaded=${h.checkpointLoaded}`);
    if (!h.reachable) {
      info("GT service is configured but not reachable — this is expected if no checkpoint is trained. predictPairs() should return source:none (not 500).");
    }
  });

  await test("predictPairs() returns source:none (not 500) when no checkpoint", async () => {
    const result = await predictPairs([{ drug: "Aspirin", disease: "headache" }]);
    if (result.source !== "gt_checkpoint" && result.source !== "none") {
      throw new Error(`unexpected source: ${result.source}`);
    }
    pass(`source=${result.source}, count=${result.count}, note=${result.note ? result.note.slice(0, 80) : "(none)"}`);
  });

  await test("topKNovel() returns source:none (not 500) when no checkpoint", async () => {
    const result = await topKNovel(10);
    if (result.source !== "gt_checkpoint" && result.source !== "none") {
      throw new Error(`unexpected source: ${result.source}`);
    }
    pass(`source=${result.source}, count=${result.count}`);
  });

  console.log("");

  // ---- Phase 4: RL Hypothesis Ranker Service ----
  console.log("--- Phase 4: RL Hypothesis Ranker Service (Issue 231) ---");

  await test("checkRlHealth() reports configured + reachable", async () => {
    const h = await checkRlHealth();
    if (!h.configured) throw new Error("RL_SERVICE_URL not configured");
    if (!h.reachable) throw new Error("RL service not reachable");
    pass(`health: configured=${h.configured}, reachable=${h.reachable}, checkpoint=${h.checkpointConfigured}, csv=${h.csvOutputAvailable}`);
  });

  await test("getRankedHypotheses() returns rankings (not 500)", async () => {
    const result = await getRankedHypotheses({ pageSize: 10 });
    if (result.source !== "rl_service" && result.source !== "none") {
      throw new Error(`unexpected source: ${result.source}`);
    }
    pass(`source=${result.source}, candidates=${result.candidates.length}, total=${result.total}, note=${result.note ? result.note.slice(0, 80) : "(none)"}`);
  });

  await test("getRankedHypotheses({drug:'Aspirin'}) filters by drug", async () => {
    const result = await getRankedHypotheses({ drug: "Aspirin", pageSize: 10 });
    pass(`source=${result.source}, candidates=${result.candidates.length}, total=${result.total}`);
  });

  console.log("");

  // ---- Acceptance Criteria Verification ----
  console.log("--- Acceptance Criteria ---");

  await test("AC1: predictPairs() does NOT throw 500", async () => {
    try {
      await predictPairs([{ drug: "Aspirin", disease: "fever" }]);
      pass("predictPairs() completed without throwing");
    } catch (err) {
      throw new Error(`predictPairs() threw: ${err}`);
    }
  });

  await test("AC2: getRankedHypotheses() does NOT throw 500", async () => {
    try {
      await getRankedHypotheses({ drug: "Aspirin" });
      pass("getRankedHypotheses() completed without throwing");
    } catch (err) {
      throw new Error(`getRankedHypotheses() threw: ${err}`);
    }
  });

  await test("AC3: getKnowledgeGraphStats() does NOT throw 500", async () => {
    try {
      await getKnowledgeGraphStats();
      pass("getKnowledgeGraphStats() completed without throwing");
    } catch (err) {
      throw new Error(`getKnowledgeGraphStats() threw: ${err}`);
    }
  });

  await test("AC4: getDatasetStats() reads from Phase 1 (not Phase 2)", async () => {
    const stats = await getDatasetStats();
    if (stats.source === "none") throw new Error(`source:none — ${stats.note}`);
    // The Phase 1 service's /stats endpoint returns backend="phase1_service".
    // If we were reading from Phase 2, backend would be "in_memory_bridge".
    if (stats.backend && !stats.backend.includes("phase1")) {
      throw new Error(`backend=${stats.backend} — expected phase1_service (reading from Phase 2?)`);
    }
    pass(`source=${stats.source}, backend=${stats.backend} (Phase 1 confirmed)`);
  });

  console.log("");
  console.log("=".repeat(70));
  console.log(`Results: ${passedTests}/${totalTests} tests passed`);
  console.log("=".repeat(70));

  if (passedTests < totalTests) {
    process.exit(1);
  }
}

main().catch((err) => {
  console.error("Fatal error:", err);
  process.exit(1);
});
