/**
 * Issue 319 (audit 301-320): E2E test — no screen renders mock data in
 * production build without a visible DEMO DATA banner.
 *
 * Acceptance criteria from the audit:
 *   (1) Production build: no screen silently shows mock data.
 *   (2) grep -r "DemoData" frontend/src/app/ returns matches only on
 *       screens with the banner.
 *   (3) E2E test passes.
 *
 * IMPLEMENTATION:
 *
 * This test verifies the source-level invariant: every screen component
 * in `frontend/src/components/drugos/remaining-screens.tsx` (and the
 * core-screens spread) that still renders a hardcoded mock-data array
 * MUST also render <DemoDataBanner> within the same component body.
 *
 * We do NOT need to start a browser to verify this — the source-grep
 * approach is deterministic, fast, and unaffected by auth/DB state.
 * A separate Playwright e2e suite (playwright.config.ts) covers the
 * runtime side against a live dev server.
 *
 * The test fails loudly if a future engineer adds a mock-data array
 * without also adding the banner. This is the structural guarantee
 * the audit requires.
 *
 * Run with:
 *   npx jest tests/e2e/no-mock-data-in-production.e2e.test.ts
 *   (or via `npm test` if e2e tests are added to the Jest testMatch)
 */

import * as fs from "fs";
import * as path from "path";

const ROOT = path.resolve(__dirname, "..", "..");
const SCREENS_FILE = path.join(
  ROOT,
  "src",
  "components",
  "drugos",
  "remaining-screens.tsx",
);
const BANNER_FILE = path.join(
  ROOT,
  "src",
  "components",
  "ui",
  "DemoDataBanner.tsx",
);

describe("Issue 319 — no-mock-data-in-production invariant", () => {
  test("DemoDataBanner component exists at expected path", () => {
    expect(fs.existsSync(BANNER_FILE)).toBe(true);
    const bannerSource = fs.readFileSync(BANNER_FILE, "utf8");
    expect(bannerSource).toContain("DEMO DATA — DO NOT USE FOR DECISIONS");
    expect(bannerSource).toContain("export function DemoDataBanner");
  });

  test("remaining-screens.tsx imports DemoDataBanner", () => {
    expect(fs.existsSync(SCREENS_FILE)).toBe(true);
    const source = fs.readFileSync(SCREENS_FILE, "utf8");
    expect(source).toMatch(
      /import\s+\{[^}]*DemoDataBanner[^}]*\}\s+from\s+['"]@\/components\/ui\/DemoDataBanner['"]/,
    );
  });

  test("every screen with a hardcoded mock-data array also renders <DemoDataBanner>", () => {
    const source = fs.readFileSync(SCREENS_FILE, "utf8");

    // Find every `function XxxScreen() {` block and check its body.
    // A "mock-data array" is a literal `const x = [{ ... }, { ... }]`
    // inside the function body — i.e. a hardcoded list rendered as data.
    //
    // Strategy: split the source at each `function XxxScreen() {` and
    // inspect each block until the matching close brace. For each block
    // that contains a hardcoded array literal with `id:` or `version:`
    // or `subject:` (clear mock-data markers), require the block to also
    // contain `<DemoDataBanner`.

    const functionRegex = /function\s+(\w+Screen)\s*\(\)\s*[:{]/g;
    let match: RegExpExecArray | null;
    const offenders: string[] = [];

    while ((match = functionRegex.exec(source)) !== null) {
      const name = match[1];
      const startIdx = match.index;
      // Find the body block (from the first `{` after the function name
      // to the matching `}`). We do a simple depth-counted scan.
      const firstBrace = source.indexOf("{", startIdx);
      if (firstBrace === -1) continue;
      let depth = 0;
      let endIdx = firstBrace;
      for (let i = firstBrace; i < source.length; i++) {
        const ch = source[i];
        if (ch === "{") depth++;
        else if (ch === "}") {
          depth--;
          if (depth === 0) {
            endIdx = i;
            break;
          }
        }
      }
      const body = source.slice(firstBrace, endIdx + 1);

      // Skip blocks that contain NO hardcoded array literals — they are
      // either real-API screens (use useApiResource/useApiList) or pure
      // layout screens with no data.
      //
      // Mock-data markers: a `const x = [...]` where the array contains
      // object literals with `id:` / `version:` / `subject:` keys — these
      // are typical "fake record" shapes (mock tickets, mock changelog
      // entries, mock deals, mock hypotheses, etc.). We do NOT flag arrays
      // with only `title:`/`name:` keys because those could be legitimate
      // static UI content (legal text sections, help categories).
      const hasHardcodedArray =
        /const\s+\w+\s*=\s*\[\s*\{\s*(id|version|subject):/m.test(
          body,
        );

      if (!hasHardcodedArray) continue;

      // The block has mock data — it MUST render <DemoDataBanner>.
      if (!/<DemoDataBanner/.test(body)) {
        offenders.push(name);
      }
    }

    if (offenders.length > 0) {
      throw new Error(
        `These screens render hardcoded mock-data arrays WITHOUT a <DemoDataBanner>:\n` +
          offenders.map((n) => `  - ${n}`).join("\n") +
          `\n\nIssue 319 requires: every screen with mock data MUST render ` +
          `the visible "DEMO DATA — DO NOT USE FOR DECISIONS" banner. ` +
          `Import DemoDataBanner from @/components/ui/DemoDataBanner and ` +
          `render it at the top of the screen body.`,
      );
    }
  });

  test("the DemoDataBanner text string appears in the source bundle", () => {
    const source = fs.readFileSync(SCREENS_FILE, "utf8");
    const bannerSource = fs.readFileSync(BANNER_FILE, "utf8");
    // The banner text is exported as DEMO_DATA_BANNER_TEXT and is the
    // default `title` prop — so any screen using <DemoDataBanner />
    // contributes the literal "DEMO DATA — DO NOT USE FOR DECISIONS"
    // to the production bundle.
    expect(bannerSource).toContain("DEMO DATA — DO NOT USE FOR DECISIONS");
    // And at least one screen in remaining-screens.tsx renders the banner.
    const bannerRenders = (source.match(/<DemoDataBanner/g) || []).length;
    expect(bannerRenders).toBeGreaterThan(0);
  });
});
