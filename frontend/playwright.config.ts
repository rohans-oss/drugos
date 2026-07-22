/**
 * Playwright E2E test config — root-level (not in uploaded_code).
 *
 * Issue 319 (audit 301-320): The repo previously had playwright.config.ts
 * only inside `uploaded_code/drugos/` — a dead copy that was never loaded.
 * The root project had no Playwright config, so `npx playwright test` did
 * nothing. This file makes the e2e suite actually run.
 *
 * Run with:
 *   npx playwright test
 *
 * Or via the helper script:
 *   node scripts/run-e2e-tests.js
 */

import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./tests/e2e",
  timeout: 60_000,
  expect: { timeout: 10_000 },
  fullyParallel: false, // Sequential to avoid killing the dev server
  retries: 0,
  workers: 1,
  reporter: [["list"], ["html", { open: "never", outputFolder: "playwright-report" }]],
  use: {
    baseURL: process.env.E2E_BASE_URL || "http://localhost:3000",
    trace: "on-first-retry",
    headless: true,
    actionTimeout: 15_000,
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
});
