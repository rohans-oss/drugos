/**
 * Test helpers shared across the TASK-274..278 test suites.
 *
 * TASK-275..278 ROOT FIX: The previous tests crashed at PrismaClient
 * init because the test env had no postgres. The new setup.ts sets a
 * global flag `__DB_AVAILABLE` — these helpers let individual test
 * suites skip DB-dependent tests with a clear message instead of
 * crashing.
 *
 * Usage:
 *   import { describeWithDb, expectDbAvailable } from "@/tests/api/db-helpers";
 *
 *   describeWithDb("DB-dependent test suite", () => {
 *     it("creates a user", async () => {
 *       const user = await db.user.create({ ... });
 *       ...
 *     });
 *   });
 *
 *   // For non-DB tests (auth gates, Zod validation), use the regular
 *   // describe() — these always run.
 */

import { describe, it, expect } from "@jest/globals";

/**
 * Returns true iff the test DB is reachable. Set by tests/api/setup.ts
 * after pinging the DB with SELECT 1.
 */
export function isDbAvailable(): boolean {
  return Boolean((globalThis as any).__DB_AVAILABLE);
}

/**
 * Describe block that SKIPS the entire suite when the test DB is not
 * available. Use this for suites where every test is DB-dependent.
 *
 * The skip message is explicit so operators know the suite needs a
 * real postgres to run.
 */
export function describeWithDb(name: string, fn: () => void): void {
  if (isDbAvailable()) {
    describe(name, fn);
  } else {
    describe.skip(name, fn);
  }
}

/**
 * Single test that skips when the DB is not available. Use this inside
 * a regular describe() block when only some tests are DB-dependent.
 */
export function itWithDb(name: string, fn: () => Promise<void> | void): void {
  if (isDbAvailable()) {
    it(name, fn);
  } else {
    it.skip(name, fn);
  }
}

/**
 * Expectation: the DB is available. Use at the start of a DB-dependent
 * test to fail fast with a clear message if the DB is missing.
 */
export function expectDbAvailable(): void {
  if (!isDbAvailable()) {
    expect(isDbAvailable()).toBe(true);
  }
}
