/**
 * Runtime verification: import each fixed service and exercise the
 * pure functions to confirm the REAL CODE (not tests) executes
 * without errors. This is the "run real code" check.
 *
 * This is NOT a test file — it's a one-shot runtime smoke check.
 * Run with: npx tsx scripts/verify-fe-020-to-028-runtime.ts
 */

import { getKnowledgeGraphStats, CANONICAL_NODE_TYPES } from "../src/lib/services/knowledge-graph-stats";
import { getDatasetStats } from "../src/lib/services/dataset-stats";
import { parseRlCsvContent, clearRlCsvCache, getRlCsvCacheState } from "../src/lib/services/rl-csv-cache";
import { truncateAbstract } from "../src/lib/services/pubmed";
import { isOpenfdaApiKeyConfigured } from "../src/lib/services/openfda";
import { RxNormTimeoutError } from "../src/lib/services/rxnorm";
import { searchPatents } from "../src/lib/services/patentsview";
import { buildTreeNumberHierarchy } from "../src/lib/services/mesh";
import { clearDrugMechanismCache, getDrugMechanismCacheState } from "../src/lib/services/drug-mechanism";

let pass = 0;
let fail = 0;

function check(name: string, cond: boolean, extra?: string): void {
  if (cond) {
    console.log(`  ✓ ${name}${extra ? " — " + extra : ""}`);
    pass++;
  } else {
    console.log(`  ✗ ${name}${extra ? " — " + extra : ""}`);
    fail++;
  }
}

async function main(): Promise<void> {
  console.log("=== FE-020 to FE-028 runtime verification ===\n");

  // FE-020: knowledge-graph-stats — read the REAL registry.json
  console.log("FE-020: knowledge-graph-stats.ts + registry.json");
  const kgStats = await getKnowledgeGraphStats();
  check("getKnowledgeGraphStats() executes", kgStats !== null);
  check("canonical node types are the 5 docx types", CANONICAL_NODE_TYPES.length === 5);
  check("response has nodeTypeCounts field", "nodeTypeCounts" in kgStats);
  check("response has edgeTypeCounts field", "edgeTypeCounts" in kgStats);
  check("response has nonCanonicalNodeCounts field", "nonCanonicalNodeCounts" in kgStats);
  check("SIDER AdverseEvent is in nonCanonicalNodeCounts (NOT nodeTypeCounts)", 
    (kgStats.nonCanonicalNodeCounts as any).AdverseEvent === 91926,
    `nonCanonical=${JSON.stringify(kgStats.nonCanonicalNodeCounts)}`);
  check("nodeCount excludes AdverseEvent (0 from this registry)", 
    kgStats.nodeCount === 0,
    `nodeCount=${kgStats.nodeCount}`);
  console.log("");

  // FE-021: dataset-stats — read the REAL checkpoint
  console.log("FE-021: dataset-stats.ts");
  const dsStats = await getDatasetStats();
  check("getDatasetStats() executes", dsStats !== null);
  check("response has status field", "status" in dsStats, `status=${dsStats.status}`);
  check("status is 'ok' (real checkpoint exists)", dsStats.status === "ok", `status=${dsStats.status}`);
  check("nodesLoaded is a number", typeof dsStats.nodesLoaded === "number", `nodesLoaded=${dsStats.nodesLoaded}`);
  console.log("");

  // FE-022: rl-csv-cache
  console.log("FE-022: rl-csv-cache.ts");
  const csv = "drug,disease,gnn_score,safety_score,market_score,reward,rank,policy_prob,confidence,pathway_score,unmet_need_score,efficacy_score,adme_score,literature_support,is_known_positive\naspirin,headache,0.9,0.8,0.7,0.5,1,0.3,0.6,0.4,0.5,0.6,0.5,1,0\n";
  const parsed = parseRlCsvContent(csv);
  check("parseRlCsvContent() parses 1 row", parsed.length === 1);
  check("clearRlCsvCache() is callable", typeof clearRlCsvCache === "function");
  check("getRlCsvCacheState() is callable", typeof getRlCsvCacheState === "function");
  check("getRlCsvCacheState() returns array", Array.isArray(getRlCsvCacheState()));
  clearRlCsvCache();
  check("clearRlCsvCache() empties the cache", getRlCsvCacheState().length === 0);
  console.log("");

  // FE-023: pubmed truncation
  console.log("FE-023: pubmed.ts truncation");
  const short = truncateAbstract("short text", 500);
  check("short text not truncated", !short.truncated && short.text === "short text");
  const long = truncateAbstract("A".repeat(1000), 500);
  check("long text truncated to 500 + ellipsis", long.truncated && long.text?.length === 501 && long.text.endsWith("\u2026"));
  check("fullLength preserved", long.fullLength === 1000);
  console.log("");

  // FE-024: openfda API key
  console.log("FE-024: openfda.ts API key");
  check("isOpenfdaApiKeyConfigured() is callable", typeof isOpenfdaApiKeyConfigured === "function");
  check("isOpenfdaApiKeyConfigured() returns boolean", typeof isOpenfdaApiKeyConfigured() === "boolean");
  console.log("");

  // FE-025: rxnorm timeout
  console.log("FE-025: rxnorm.ts timeout");
  const err = new RxNormTimeoutError("https://rxnav.nlm.nih.gov/REST/test");
  check("RxNormTimeoutError is an Error", err instanceof Error);
  check("RxNormTimeoutError has endpoint field", err.endpoint === "https://rxnav.nlm.nih.gov/REST/test");
  check("RxNormTimeoutError message mentions 3000ms", err.message.includes("3000ms"));
  console.log("");

  // FE-026: patentsview pagination
  console.log("FE-026: patentsview.ts pagination");
  check("searchPatents is callable", typeof searchPatents === "function");
  // Without API key, should return reason gracefully
  const prevKey = process.env.PATENTSVIEW_API_KEY;
  delete process.env.PATENTSVIEW_API_KEY;
  const noKeyResult = await searchPatents({ query: "aspirin" });
  check("returns graceful reason when API key missing", noKeyResult.patents.length === 0 && !!noKeyResult.reason);
  if (prevKey) process.env.PATENTSVIEW_API_KEY = prevKey;
  console.log("");

  // FE-027: mesh tree hierarchy
  console.log("FE-027: mesh.ts tree hierarchy");
  const forest = buildTreeNumberHierarchy(["D03.438.221", "D03.438", "C01.001"]);
  check("buildTreeNumberHierarchy returns 2 roots", forest.length === 2);
  const d03 = forest.find((n) => n.treeNumber === "D03");
  check("D03 root has 1 child (D03.438)", d03?.children.length === 1);
  check("D03.438 has 1 child (D03.438.221)", d03?.children[0].children.length === 1);
  check("D03.438.221 has 0 children", d03?.children[0].children[0].children.length === 0);
  console.log("");

  // FE-028: drug-mechanism cache
  console.log("FE-028: drug-mechanism.ts cache");
  check("clearDrugMechanismCache is callable", typeof clearDrugMechanismCache === "function");
  check("getDrugMechanismCacheState is callable", typeof getDrugMechanismCacheState === "function");
  clearDrugMechanismCache();
  check("clearDrugMechanismCache empties the cache", getDrugMechanismCacheState().length === 0);
  console.log("");

  console.log("=== Runtime verification complete ===");
  console.log(`Passed: ${pass}, Failed: ${fail}`);
  if (fail > 0) {
    process.exit(1);
  }
}

main().catch((e) => {
  console.error("Runtime verification crashed:", e);
  process.exit(1);
});
