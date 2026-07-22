/**
 * Test env setup — runs BEFORE any test file is loaded.
 *
 * TASK-275..278 ROOT FIX: The previous env.ts set DATABASE_URL to a SQLite
 * file URL (`file:./db/test.db`), but the Prisma schema requires
 * `postgresql://`. This caused EVERY DB-backed test to crash at
 * PrismaClient init with:
 *
 *   "the URL must start with the protocol postgresql:// or postgres://"
 *
 * The prior tests were silently broken — they appeared to "pass" only
 * because the setup.ts `try/catch` swallowed the `prisma db push` error,
 * and the test framework's `beforeEach` hooks caught the PrismaClient
 * init errors and reported them as test failures (not crashes). The
 * net effect was: every DB-backed test failed, but the suite "ran".
 *
 * This commit sets DATABASE_URL to a proper postgres URL. In CI, set
 * `TEST_DATABASE_URL` to a real postgres connection string. In local
 * dev without postgres, the URL is still postgres-formatted (so schema
 * validation passes) but the connection will fail — DB-backed tests
 * will be skipped with a clear message, not crash.
 */

const path = require("path");

(process.env as Record<string, string | undefined>).NODE_ENV = "test";
// Use TEST_DATABASE_URL if set (CI with real postgres), else fall back
// to a localhost postgres URL (works in CI; fails gracefully in local
// dev without postgres).
process.env.DATABASE_URL =
  process.env.TEST_DATABASE_URL ||
  "postgresql://drugos:drugos@localhost:5432/drugos_test?schema=public";
process.env.JWT_SECRET =
  process.env.JWT_SECRET ||
  "test-secret-only-not-for-production-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx";
