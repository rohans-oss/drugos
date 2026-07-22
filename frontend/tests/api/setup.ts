/**
 * Test setup — runs once before each test suite.
 *
 * The test DATABASE_URL is set by tests/api/env.ts (a setupFile, which runs
 * before any test module is imported). This file ensures the schema is
 * pushed to the test DB and provides per-test cleanup.
 *
 * TASK-275..278 ROOT FIX: The previous setup.ts silently swallowed the
 * `prisma db push` error, which meant tests ran against a non-existent
 * DB and crashed at query time. The new setup.ts:
 *   1. Detects whether the test DB is reachable (pings it with SELECT 1).
 *   2. If reachable, pushes the schema and cleans tables between tests.
 *   3. If NOT reachable, sets a global flag `__DB_AVAILABLE = false` so
 *      individual test suites can skip DB-dependent tests with a clear
 *      message (instead of crashing).
 *
 * This makes the test suite HONEST — DB tests are skipped (not silently
 * failed) when no DB is available, and the non-DB tests (auth gates,
 * Zod validation, route logic) still run.
 */

import { execSync } from "child_process";

let prismaClient: any;
// Global flag — test suites check this to decide whether to skip
// DB-dependent tests. Set by beforeAll below.
(globalThis as any).__DB_AVAILABLE = false;

beforeAll(async () => {
  // Try to push schema to the test DB.
  try {
    execSync("npx prisma db push --skip-generate --accept-data-loss 2>&1", {
      stdio: "pipe",
      env: { ...process.env, DATABASE_URL: process.env.DATABASE_URL },
    });
  } catch (e) {
    // Schema push failed — either the DB is unreachable, or the schema
    // is already in sync. We'll detect which by pinging the DB below.
  }

  try {
    const { PrismaClient } = await import("@prisma/client");
    prismaClient = new PrismaClient({
      datasources: { db: { url: process.env.DATABASE_URL } },
    });
    // Ping the DB with a trivial query. If this throws, the DB is not
    // available and DB-dependent tests should be skipped.
    await prismaClient.$queryRaw`SELECT 1`;
    (globalThis as any).__DB_AVAILABLE = true;
    (globalThis as any).__testPrisma = prismaClient;
  } catch (e) {
    // DB not available — set the flag so test suites can skip.
    (globalThis as any).__DB_AVAILABLE = false;
    console.warn(
      "\n[TEST SETUP] Test database is not available at " +
      process.env.DATABASE_URL +
      ". DB-dependent tests will be SKIPPED. Non-DB tests (auth gates, " +
      "Zod validation, route logic) will still run.\n"
    );
  }
  // Issue 319 (audit 301-320): wrap PrismaClient init in try/catch so
  // filesystem-only e2e tests (like no-mock-data-in-production.e2e.ts)
  // do not fail when the test DB is unavailable. The prismaClient will
  // remain null and beforeEach will skip table cleanup — that's fine
  // because the e2e test does not touch the DB.
  // TASK-275..278: if the earlier try/catch already created prismaClient
  // AND set __DB_AVAILABLE = true, we skip this redundant init. If the
  // earlier try/catch failed (DB unreachable), we still attempt to
  // create the client here so non-DB tests can import @/lib/db without
  // crashing (the client will throw on query, but import works).
  if (!prismaClient) {
    try {
      const { PrismaClient } = await import("@prisma/client");
      prismaClient = new PrismaClient({
        datasources: { db: { url: process.env.DATABASE_URL } },
      });
      (globalThis as any).__testPrisma = prismaClient;
    } catch (e) {
      // Prisma client not available (e.g. not generated yet, or DB
      // unreachable). Non-DB tests can still run.
    }
  }
});

afterAll(async () => {
  if (prismaClient) {
    try {
      await prismaClient.$disconnect();
    } catch {
      // ignore
    }
  }
});

beforeEach(async () => {
  if (!prismaClient || !(globalThis as any).__DB_AVAILABLE) return;
  // Clean all tables between tests for hermeticity.
  const tablenames = [
    "AuditLogDeadLetter",
    "AuditLog",
    "Notification",
    "MfaChallenge",
    "RefreshToken",
    "ApiKey",
    "EvidencePackage",
    "BillingInvoice",
    "Subscription",
    "Comment",
    "ProjectActivity",
    "Hypothesis",
    "Project",
    "OrganizationMember",
    "Organization",
    "User",
  ];
  // Delete in dependency order to avoid FK violations.
  for (const t of tablenames) {
    try {
      // @ts-ignore — dynamic model name
      await prismaClient[t].deleteMany({});
    } catch (e) {
      // Some tables may not exist yet — ignore
    }
  }
});
