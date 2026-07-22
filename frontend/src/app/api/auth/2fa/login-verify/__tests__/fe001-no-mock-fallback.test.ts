/**
 * FE-001 ROOT FIX (v2) regression test: SearchResultsScreen must NEVER
 * fall back to mock drug candidates.
 *
 * This test would have caught the original FE-001 bug: the SearchResultsScreen
 * had a line `const mockCandidates = drugCandidates.filter(...)` and
 * `const candidates = realCandidates.length > 0 ? realCandidates : mockCandidates;`
 * — i.e. when the RL ranker returned zero candidates, the screen rendered
 * 13 hardcoded fake drugs (Memantine 87, Riluzole 84, …) as if they were
 * real RL predictions.
 *
 * Root fix: the mock fallback line was removed entirely, and a hard EMPTY
 * STATE is rendered in place of the table when realCandidates is empty.
 *
 * This is a SOURCE-LEVEL regression test that reads the actual component
 * source and asserts the forbidden pattern is gone.
 */
import * as fs from "fs";
import * as path from "path";

const COMPONENT_PATH = path.resolve(
  __dirname,
  "../../../../../../components/drugos/core-screens.tsx"
);

function readSource(): string {
  return fs.readFileSync(COMPONENT_PATH, "utf-8");
}

// Extract just the SearchResultsScreen function body.
function extractSearchResultsScreen(src: string): string {
  const start = src.indexOf("function SearchResultsScreen()");
  expect(start).toBeGreaterThan(-1);
  // Find the matching closing brace by scanning for the next "^}" at
  // column 0 (top-level function end).
  const rest = src.slice(start);
  const m = rest.match(/\n\}\n/);
  expect(m).not.toBeNull();
  return rest.slice(0, m!.index! + 3);
}

describe("FE-001 (v2): SearchResultsScreen never falls back to mock data", () => {
  const source = readSource();
  const screen = extractSearchResultsScreen(source);

  test("does NOT assign mockCandidates to candidates (the forbidden fallback)", () => {
    // The exact line that caused the bug:
    //   const candidates = realCandidates.length > 0 ? realCandidates : mockCandidates;
    expect(screen).not.toMatch(
      /const candidates\s*=\s*realCandidates\.length\s*>\s*0\s*\?\s*realCandidates\s*:\s*mockCandidates/
    );
  });

  test("does NOT compute mockCandidates from drugCandidates for fallback", () => {
    // The line that built the fallback:
    //   const mockCandidates = drugCandidates.filter(c => c.diseaseId === diseaseId);
    // We allow `drugCandidates` to be imported (it's used by OTHER screens
    // in the file), but the SearchResultsScreen function body must NOT
    // contain a mockCandidates assignment.
    expect(screen).not.toMatch(
      /const mockCandidates\s*=\s*drugCandidates\.filter/
    );
  });

  test("candidates is assigned DIRECTLY from realCandidates (no ternary)", () => {
    // The root fix:
    //   const candidates = realCandidates;
    expect(screen).toMatch(/const candidates\s*=\s*realCandidates\s*;?/);
  });

  test("renders a hard EMPTY STATE when realCandidates is empty", () => {
    // The empty state must mention "No RL predictions available" so a
    // researcher sees an actionable message instead of a confusing empty
    // table.
    expect(screen).toMatch(/No RL predictions available/);
  });

  test("the table is only rendered when realCandidates.length > 0", () => {
    // The Card containing the results table must be wrapped in a
    // `{realCandidates.length > 0 && (...)}` conditional so it never
    // renders on an empty result set.
    expect(screen).toMatch(/\{realCandidates\.length\s*>\s*0\s*&&\s*\(/);
  });

  test("does NOT render the misleading 'demo data' amber banner", () => {
    // The old banner said "Showing demo data. The Phase 4 RL ranker is
    // not deployed." above an identical-looking table. The new code
    // removes this banner and renders the empty state instead.
    expect(screen).not.toMatch(/Showing demo data\./);
  });
});
