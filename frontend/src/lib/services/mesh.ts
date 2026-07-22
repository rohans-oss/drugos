/**
 * Disease search via MeSH (Medical Subject Headings) — the NLM's controlled
 * vocabulary thesaurus used for indexing PubMed.
 *
 * Source: NLM MeSH REST API (https://id.nlm.nih.gov/mesh/lookup)
 * License: Public domain.
 *
 * We use MeSH because it is the single authoritative vocabulary for diseases
 * used across biomedical research. Every PubMed article is indexed against
 * MeSH descriptors.
 *
 * FE-027 ROOT FIX (Team Member 15):
 *
 * ROOT CAUSE: MeSH tree numbers are hierarchical (e.g. D03 = diseases,
 * D03.438 = nervous system diseases, D03.438.221 = epilepsy). The
 * previous code returned them as a flat `string[]` — discarding the
 * hierarchy. The frontend could not navigate diseases by category.
 *
 * ROOT FIX: in addition to the flat `treeNumber` list (kept for
 * backwards compatibility), we now build a `treeNumberHierarchy`:
 * a nested `MeshTreeNode[]` forest where each node's `children`
 * contains its descendant tree numbers. The dashboard can render
 * this as a collapsible tree.
 *
 * The hierarchy is derived purely from the dot-separated structure
 * of the tree numbers themselves — no extra API calls needed. For
 * example, given `["D03.438.221", "D03.438"]`, we build:
 *   { treeNumber: "D03", children: [
 *       { treeNumber: "D03.438", children: [
 *           { treeNumber: "D03.438.221", children: [] }
 *       ]}
 *   ]}
 *
 * Note: MeSH tree numbers may share prefixes across different
 * descriptors (e.g. D03.438 may belong to a different descriptor
 * than D03.438.221). The hierarchy built here is LOCAL to the
 * descriptor's own tree numbers — we do not cross-reference other
 * descriptors. This preserves the "where does THIS descriptor sit
 * in the MeSH tree?" semantics without conflating it with
 * "what are ALL the descriptors under D03.438?".
 */

import { monitoredFetch } from "@/lib/external-api-monitor";

const MESH_BASE = "https://id.nlm.nih.gov/mesh/lookup";

export interface MeshDescriptor {
  descriptorUi: string; // e.g., D000001
  name: string;
  scopeNote?: string;
  allowDuplicates: boolean;
  treeNumber: string[]; // e.g., ["C01.001"] — flat list (backwards-compat)
  /**
   * FE-027: nested tree-number hierarchy. Each root node has its
   * descendants as `children`. Built from the flat `treeNumber`
   * list by parsing the dot-separated path.
   */
  treeNumberHierarchy: MeshTreeNode[];
}

export interface MeshTreeNode {
  /** Full tree number at this node (e.g. "D03.438.221"). */
  treeNumber: string;
  /** The path segments from root to this node (e.g. ["D03","438","221"]). */
  path: string[];
  /** Child nodes (descendants of this tree number). */
  children: MeshTreeNode[];
}

/**
 * FE-027 ROOT FIX: build a nested forest of `MeshTreeNode` from a flat
 * list of tree numbers. The hierarchy is derived from the dot-separated
 * path structure.
 *
 * Example:
 *   buildTreeNumberHierarchy(["D03.438.221", "D03.438", "C01.001"])
 *   => [
 *        { treeNumber: "D03", path: ["D03"], children: [
 *            { treeNumber: "D03.438", path: ["D03","438"], children: [
 *                { treeNumber: "D03.438.221", path: ["D03","438","221"], children: [] }
 *            ]}
 *        ]},
 *        { treeNumber: "C01", path: ["C01"], children: [
 *            { treeNumber: "C01.001", path: ["C01","001"], children: [] }
 *        ]}
 *      ]
 *
 * Algorithm: for each tree number, split on "." to get the path. Walk
 * the forest from the root, creating intermediate nodes as needed, and
 * insert the leaf. This is O(N * D) where N is the number of tree
 * numbers and D is the max depth (typically ≤ 5 for MeSH).
 *
 * Exported for unit testing.
 */
export function buildTreeNumberHierarchy(treeNumbers: string[]): MeshTreeNode[] {
  const forest: MeshTreeNode[] = [];

  // Sort by tree number so parents are inserted before children
  // (e.g. "D03" before "D03.438" before "D03.438.221"). This is a
  // lexical sort — it works because the dot separator sorts before
  // any digit, so "D03" < "D03.438" < "D03.438.221".
  const sorted = [...treeNumbers].filter(Boolean).sort();

  for (const tn of sorted) {
    const segments = tn.split(".");
    let currentLevel = forest;
    let currentPath: string[] = [];

    for (let i = 0; i < segments.length; i++) {
      const segment = segments[i];
      currentPath = currentPath.concat(segment);
      const fullPath = currentPath.join(".");
      let node = currentLevel.find((n) => n.treeNumber === fullPath);
      if (!node) {
        node = {
          treeNumber: fullPath,
          path: [...currentPath],
          children: [],
        };
        currentLevel.push(node);
      }
      currentLevel = node.children;
    }
  }

  return forest;
}

/**
 * Lookup MeSH descriptors matching a free-text disease query.
 * Returns the canonical descriptor record(s) used to index that disease.
 *
 * BE-059 ROOT FIX (v115, MEDIUM): the previous code made 1 search call
 * + (10 descriptors × 3 sequential calls each) = 31 SEQUENTIAL HTTP
 * requests to NLM. At ~300ms per call, that's ~9.3s for a single
 * disease search — well beyond the 2s UX budget. Under V1's 100-
 * concurrent-request load, NLM would block the platform's IP.
 *
 * ROOT FIX:
 *   1. Parallelize the per-descriptor calls (name, scopeNote,
 *      treeNumber) via Promise.all — each descriptor now takes
 *      ~300ms (the slowest of the 3) instead of ~900ms.
 *   2. Parallelize across descriptors via Promise.all on the outer
 *      loop — all 10 descriptors' calls run concurrently.
 *   3. Add a 5s overall timeout — if NLM is slow, we abort and
 *      return partial results rather than making the researcher wait.
 *
 * Net: 31 sequential calls → 2 concurrent waves (1 search + 1 wave
 * of 10 descriptors × 3 parallel calls = ~600ms total instead of 9.3s).
 */
export async function searchDiseasesByName(
  query: string,
  limit = 10
): Promise<MeshDescriptor[]> {
  const q = (query || "").trim();
  if (q.length < 2) return [];
  const url = `${MESH_BASE}/descriptor?label=${encodeURIComponent(q)}&limit=${limit}`;
  // Task 260: monitored for observability.
  const res = await monitoredFetch("mesh", url, {
    headers: { Accept: "application/json" },
    next: { revalidate: 86400 * 30 }, // MeSH updates ~weekly
  });
  if (!res.ok) {
    throw new Error(`MeSH descriptor lookup returned ${res.status}`);
  }
  const uris: string[] = await res.json();
  if (uris.length === 0) return [];

  // BE-059 ROOT FIX: parallelize ALL descriptor lookups. Each
  // descriptor makes 3 concurrent HTTP calls (name, scopeNote,
  // treeNumber) — Promise.all inside the per-descriptor function.
  // The outer Promise.all runs all descriptors in parallel.
  const OVERALL_TIMEOUT_MS = 5_000;

  // Per-descriptor fetcher: returns MeshDescriptor or null on failure.
  // All 3 HTTP calls (name, scopeNote, treeNumber) run concurrently
  // via Promise.allSettled — a failure in any one does NOT abort the
  // others (we still get name + treeNumber even if scopeNote fails).
  async function fetchDescriptor(uri: string): Promise<MeshDescriptor | null> {
    const descriptorUi = uri.split("/").pop() || "";
    if (!descriptorUi) return null;

    const commonOpts = {
      headers: { Accept: "application/json" },
      next: { revalidate: 86400 * 30 },
    };

    // Fire all 3 requests in parallel — Promise.allSettled so a
    // failure in one doesn't abort the others.
    const [nameResult, snResult, tnResult] = await Promise.allSettled([
      monitoredFetch(
        "mesh",
        `${MESH_BASE}/descriptor?resource=${encodeURIComponent(uri)}`,
        commonOpts
      ),
      monitoredFetch(
        "mesh",
        `${MESH_BASE}/scopeNote?resource=${encodeURIComponent(uri)}`,
        commonOpts
      ),
      monitoredFetch(
        "mesh",
        `${MESH_BASE}/treeNumber?resource=${encodeURIComponent(uri)}`,
        commonOpts
      ),
    ]);

    // Extract name (required — if this failed, skip the descriptor).
    if (nameResult.status !== "fulfilled" || !nameResult.value.ok) {
      return null;
    }
    const name = (await nameResult.value.text()).trim().replace(/^"|"$/g, "");
    if (!name) return null;

    // Extract scopeNote (optional — undefined if failed).
    let scopeNote: string | undefined;
    if (snResult.status === "fulfilled" && snResult.value.ok) {
      try {
        scopeNote = (await snResult.value.text()).trim().replace(/^"|"$/g, "");
      } catch {
        scopeNote = undefined;
      }
    }

    // Extract treeNumber (optional — [] if failed).
    let treeNumber: string[] = [];
    if (tnResult.status === "fulfilled" && tnResult.value.ok) {
      try {
        const tnVal = await tnResult.value.json();
        if (Array.isArray(tnVal)) treeNumber = tnVal;
      } catch {
        treeNumber = [];
      }
    }

    return {
      descriptorUi,
      name,
      scopeNote,
      allowDuplicates: false,
      treeNumber,
      treeNumberHierarchy: buildTreeNumberHierarchy(treeNumber),
    };
  }

  // Race the parallel descriptor fetches against the overall timeout.
  // If the timeout fires, return whatever descriptors completed
  // (better partial results than none).
  const descriptorPromises = uris.slice(0, limit).map(fetchDescriptor);
  const timeoutPromise = new Promise<MeshDescriptor[]>((resolve) => {
    setTimeout(() => {
      // Return whatever we have — descriptors that haven't resolved
      // yet are simply dropped.
      resolve([]);
    }, OVERALL_TIMEOUT_MS);
  });

  // We need to capture partial results if the timeout fires. Use a
  // shared array that each descriptor promise appends to as it resolves.
  const partialResults: MeshDescriptor[] = [];
  const capturePromise = Promise.all(
    descriptorPromises.map(async (p) => {
      const result = await p;
      if (result) partialResults.push(result);
    })
  );

  await Promise.race([capturePromise, timeoutPromise]);

  return partialResults;
}
