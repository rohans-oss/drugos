/**
 * FE-069 ROOT FIX tests: rl-ranker.ts CSV cache.
 *
 * These tests verify that getRankedHypotheses() (via readLocalCsv) ACTUALLY
 * caches the parsed CSV result — the second call within TTL returns the SAME
 * array reference (cache hit), and a change to the file's mtime invalidates
 * the cache (cache miss → re-parse).
 *
 * The previous "fix" added a separate rl-csv-cache.ts module with its own
 * cache, but that module was never imported by rl-ranker.ts (the single
 * source of truth for parsing, per FE-019). So the cache was dead code and
 * rl-ranker.ts re-parsed the CSV on every request. These tests catch that
 * by exercising rl-ranker.ts directly with a real tmpdir CSV.
 */

import { promises as fs } from "fs";
import * as path from "path";
import * as os from "os";
import {
  getRankedHypotheses,
  __clearRlRankerCsvCacheForTests,
} from "@/lib/services/rl-ranker";

const SAMPLE_CSV = `drug,disease,gnn_score,safety_score,market_score,reward,rank,policy_prob,literature_support,is_known_positive,confidence,pathway_score,unmet_need_score,efficacy_score,adme_score
metformin,alzheimer,0.85,0.92,0.78,0.83,1,0.71,1,0,0.80,0.82,0.90,0.75,0.88
aspirin,migraine,0.72,0.65,0.55,0.62,2,0.58,0,0,0.70,0.68,0.60,0.65,0.70
losartan,diabetes,0.91,0.95,0.82,0.89,3,0.78,1,1,0.88,0.85,0.92,0.80,0.90
`;

describe("FE-069: rl-ranker.ts readLocalCsv TTL cache", () => {
  const originalEnv = { ...process.env };

  beforeEach(() => {
    __clearRlRankerCsvCacheForTests();
    process.env = { ...originalEnv };
    delete process.env.RL_SERVICE_URL; // force the local-CSV path
  });

  afterAll(() => {
    process.env = originalEnv;
  });

  test("second call within TTL returns the SAME array reference (cache hit)", async () => {
    const tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), "rl-ranker-cache-"));
    const csvPath = path.join(tmpDir, "rl.csv");
    await fs.writeFile(csvPath, SAMPLE_CSV);
    process.env.RL_OUTPUT_CSV_PATH = csvPath;

    try {
      const first = await getRankedHypotheses({ limit: 50 });
      expect(first.candidates.length).toBe(3);

      const second = await getRankedHypotheses({ limit: 50 });
      // CRITICAL: the second call returns the SAME candidate array
      // reference (cache hit) — proving readLocalCsv did NOT re-read or
      // re-parse the CSV.
      expect(second.candidates).toBe(first.candidates);
    } finally {
      await fs.rm(tmpDir, { recursive: true, force: true });
      __clearRlRankerCsvCacheForTests();
    }
  });

  test("cache invalidates when CSV mtime changes (re-parse on file update)", async () => {
    const tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), "rl-ranker-cache-"));
    const csvPath = path.join(tmpDir, "rl.csv");
    await fs.writeFile(csvPath, SAMPLE_CSV);
    process.env.RL_OUTPUT_CSV_PATH = csvPath;

    try {
      const first = await getRankedHypotheses({ limit: 50 });
      expect(first.candidates.length).toBe(3);

      // Wait >1s so the filesystem mtime granularity actually changes.
      await new Promise((r) => setTimeout(r, 1100));

      const NEW_CSV =
        SAMPLE_CSV +
        "ritonavir,hiv,0.99,0.88,0.95,0.91,4,0.82,1,1,0.90,0.88,0.95,0.85,0.92\n";
      await fs.writeFile(csvPath, NEW_CSV);

      const second = await getRankedHypotheses({ limit: 50 });
      // CRITICAL: the cache must have been invalidated → new array with 4 rows.
      expect(second.candidates).not.toBe(first.candidates);
      expect(second.candidates.length).toBe(4);
      expect(second.candidates[3].drug).toBe("ritonavir");
    } finally {
      await fs.rm(tmpDir, { recursive: true, force: true });
      __clearRlRankerCsvCacheForTests();
    }
  });

  test("returns empty list (source: none) when CSV does not exist — NEVER fabricates", async () => {
    process.env.RL_OUTPUT_CSV_PATH = "/tmp/nonexistent-rl-csv-12345.csv";
    const result = await getRankedHypotheses({ limit: 50 });
    expect(result.candidates).toEqual([]);
    expect(result.source).toBe("none");
    // CRITICAL: source is "none", NOT "local_csv" with an empty array —
    // the caller can distinguish "no file" from "empty file".
  });

  test("respects drug filter on cached result (filter applied AFTER cache lookup)", async () => {
    const tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), "rl-ranker-cache-"));
    const csvPath = path.join(tmpDir, "rl.csv");
    await fs.writeFile(csvPath, SAMPLE_CSV);
    process.env.RL_OUTPUT_CSV_PATH = csvPath;

    try {
      // First call: no filter → caches all 3.
      const all = await getRankedHypotheses({ limit: 50 });
      expect(all.candidates.length).toBe(3);

      // Second call: drug filter "asp" → should hit cache then filter.
      const filtered = await getRankedHypotheses({ drug: "asp", limit: 50 });
      expect(filtered.candidates.length).toBe(1);
      expect(filtered.candidates[0].drug).toBe("aspirin");
    } finally {
      await fs.rm(tmpDir, { recursive: true, force: true });
      __clearRlRankerCsvCacheForTests();
    }
  });
});
