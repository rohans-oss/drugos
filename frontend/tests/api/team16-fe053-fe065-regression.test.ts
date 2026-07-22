/**
 * Team 16 — Frontend UI Components & Visualization
 * Regression tests for FE-053, FE-062, FE-063, FE-064, FE-065.
 *
 * These tests verify the ROOT-LEVEL fixes by reading the actual source
 * files and asserting that:
 *   1. The broken pattern is GONE (not just commented out).
 *   2. The correct pattern is PRESENT.
 *
 * This is a static-analysis test — it reads the source files directly
 * rather than rendering components, because the fixes are structural
 * (removed dead code, removed type-safety bypasses, removed mock-data
 * dependencies). A runtime test would not catch a regression where
 * someone re-adds an `as any` cast or re-imports from mock-data.
 */

import * as fs from "fs";
import * as path from "path";

const SRC_ROOT = path.resolve(__dirname, "../../src");

function readFile(rel: string): string {
  return fs.readFileSync(path.join(SRC_ROOT, rel), "utf8");
}

// ─── FE-053: Inline ScoreBar/SafetyBadge removed from core-screens.tsx ───

describe("FE-053: core-screens.tsx uses dedicated ScoreBar/SafetyBadge", () => {
  const coreScreens = readFile("components/drugos/core-screens.tsx");

  test("imports ScoreBar from the dedicated ./score-bar module", () => {
    expect(coreScreens).toContain("import { ScoreBar } from './score-bar'");
  });

  test("imports SafetyBadge from the dedicated ./safety-badge module", () => {
    expect(coreScreens).toContain("import { SafetyBadge } from './safety-badge'");
  });

  test("does NOT define an inline 'function ScoreBar(' — the duplicate is gone", () => {
    // The inline definition would look like: function ScoreBar({ score, size ...
    // We check that no such function DEFINITION exists (usage like <ScoreBar is fine).
    const inlineDef = /function\s+ScoreBar\s*\(/;
    expect(inlineDef.test(coreScreens)).toBe(false);
  });

  test("does NOT define an inline 'function SafetyBadge(' — the duplicate is gone", () => {
    const inlineDef = /function\s+SafetyBadge\s*\(/;
    expect(inlineDef.test(coreScreens)).toBe(false);
  });

  test("scoreColor helper is retained (used by 13+ other call sites)", () => {
    expect(coreScreens).toContain("function scoreColor(");
  });
});

// ─── FE-062: `as any` cast removed for disease name ───

describe("FE-062: core-screens.tsx — no `as any` cast for currentRoute.name", () => {
  const coreScreens = readFile("components/drugos/core-screens.tsx");

  test("does NOT use '(currentRoute as any).name'", () => {
    expect(coreScreens).not.toContain("(currentRoute as any).name");
    expect(coreScreens).not.toContain("currentRoute as any");
  });

  test("uses direct 'currentRoute.name' property access (type-safe)", () => {
    expect(coreScreens).toContain("currentRoute.name");
  });
});

// ─── FE-063: No `catch (e: any)` in rl/route.ts ───

describe("FE-063: rl/route.ts — all catch blocks use `e: unknown`", () => {
  const rlRoute = readFile("app/api/rl/route.ts");

  test("does NOT contain 'catch (e: any)' anywhere in the file", () => {
    expect(rlRoute).not.toContain("catch (e: any)");
    expect(rlRoute).not.toContain("catch(e: any)");
  });

  test("all catch blocks use 'catch (e: unknown)' with proper narrowing", () => {
    // Count catch blocks — there should be at least 3 (2 in route handlers
    // + 1 in persistRlCandidates outer catch). The main branch's version of
    // persistRlCandidates is simpler (no project-creation inner catch), so
    // the total is 3, not 4.
    const catchUnknownCount = (rlRoute.match(/catch\s*\(\s*e:\s*unknown\s*\)/g) || []).length;
    expect(catchUnknownCount).toBeGreaterThanOrEqual(3);

    // Verify narrowing pattern is used: e instanceof Error ? e.message : String(e)
    expect(rlRoute).toContain("e instanceof Error ? e.message : String(e)");
  });
});

// ─── FE-064: No hardcoded chart data in admin-billing-etc-screens.tsx ───

describe("FE-064: admin-billing-etc-screens.tsx — no fabricated chart data", () => {
  const adminScreens = readFile("components/drugos/admin-billing-etc-screens.tsx");

  const fabricatedArrays = [
    "usageTrendData",
    "endpointData",
    "revenueProjectionData",
    "marketSizingData",
    "radarData",
    "comparableData",
    "pipelinePredictData",
    "royaltyData",
    "apiUsageTimeData",
    "moatData",
  ];

  test.each(fabricatedArrays)("does NOT define fabricated dataset '%s'", (name) => {
    // Check that the variable is not defined as a const with data.
    // The fix deleted all 10 arrays; the only mention should be in the
    // explanatory comment listing what was removed.
    const defPattern = new RegExp(`const\\s+${name}\\s*=\\s*\\[`);
    expect(defPattern.test(adminScreens)).toBe(false);
  });

  test("contains a clear comment explaining the fabricated data was deleted (FE-064 ROOT FIX)", () => {
    expect(adminScreens).toContain("FE-064 ROOT FIX");
    expect(adminScreens).toContain("DELETED");
  });
});

// ─── FE-065: app-router.tsx does not import from @/lib/mock-data ───

describe("FE-065: app-router.tsx — no mock-data dependency", () => {
  const appRouter = readFile("components/drugos/app-router.tsx");

  test("does NOT import from '@/lib/mock-data'", () => {
    // The old import was: from '@/lib/mock-data'
    expect(appRouter).not.toContain("from '@/lib/mock-data'");
    expect(appRouter).not.toContain('from "@/lib/mock-data"');
  });

  test("contains the FE-065 ROOT FIX comment explaining the removal", () => {
    expect(appRouter).toContain("FE-065 ROOT FIX");
    expect(appRouter).toContain("23 imports");
  });

  test("defines local empty constants for the 9 used data items (type-safe)", () => {
    // These 9 were used in the file and need local empty replacements.
    const usedNames = [
      "diseases",
      "drugCandidates",
      "notifData",
      "usageMetrics",
      "trendingDiseases",
      "recentQueries",
      "systemStatus",
      "blogPosts",
      "careers",
    ];
    for (const name of usedNames) {
      // Each should be defined as a const with an empty value.
      const defPattern = new RegExp(`const\\s+${name}\\s*[:=]`);
      expect(defPattern.test(appRouter)).toBe(true);
    }
  });

  test("imports types from @/lib/types (Disease, DrugCandidate, AppNotification)", () => {
    expect(appRouter).toContain("import type { Disease, DrugCandidate, AppNotification } from '@/lib/types'");
  });

  test("does NOT reference the 14 unused mock-data names that were removed", () => {
    // These 14 were imported but never used — they should not appear as
    // standalone variable references in actual CODE (only in the
    // explanatory comment block). We strip // and /* */ comments before
    // checking so that comment mentions don't trigger false positives.
    const stripped = appRouter
      .replace(/\/\*[\s\S]*?\*\//g, "")   // strip /* */ block comments
      .replace(/\/\/.*$/gm, "");           // strip // line comments

    const removedUnused = [
      "clinicalTrials",
      "graphNodes",
      "graphEdges",
      "auditLogs",
      "billingHistory",
      "apiKeys",
      "webhooks",
      "dataSources",
      "dealPipeline",
      "featureFlags",
      "savedQueries",
    ];
    for (const name of removedUnused) {
      // The name should NOT appear as a variable reference followed by a
      // dot (property access) or square bracket (index access) — those
      // patterns indicate the variable is being used in code.
      const usagePattern = new RegExp(`\\b${name}\\b\\s*[.\\[]`);
      expect(usagePattern.test(stripped)).toBe(false);
    }
  });
});
