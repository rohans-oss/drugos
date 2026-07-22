/**
 * Jest config for the frontend test suite.
 *
 * Uses ts-jest to transform TypeScript. The test environment is node because
 * all tests are backend/service tests (no DOM).
 *
 * FE-069/070/073 INFRA FIX: setupFiles and setupFilesAfterEnv were missing
 * from a prior refactor, which meant tests/api/env.ts (sets DATABASE_URL +
 * JWT_SECRET for the test DB) and tests/api/setup.ts (pushes schema,
 * provides per-test table cleanup) were never loaded. Every DB-backed
 * test would fail at module-import time with "URL must start with
 * postgresql://". Restored here so the test suite actually runs.
 */
module.exports = {
  preset: "ts-jest",
  testEnvironment: "node",
  testMatch: [
    "<rootDir>/src/lib/services/__tests__/**/*.test.ts",
    "<rootDir>/src/lib/auth/__tests__/**/*.test.ts",
    "<rootDir>/src/app/api/**/__tests__/**/*.test.ts",
    "<rootDir>/tests/api/**/*.test.ts",
    "<rootDir>/tests/e2e/**/*.e2e.ts",
  ],
  setupFiles: ["<rootDir>/tests/api/env.ts", "<rootDir>/tests/api/jest-setup.ts"],
  setupFilesAfterEnv: ["<rootDir>/tests/api/setup.ts"],
  moduleNameMapper: {
    "^@/(.*)$": "<rootDir>/src/$1",
  },
  transform: {
    "^.+\\.tsx?$": [
      "ts-jest",
      {
        tsconfig: {
          target: "ES2022",
          module: "CommonJS",
          esModuleInterop: true,
          skipLibCheck: true,
          resolveJsonModule: true,
          allowJs: true,
        },
      },
    ],
  },
  transformIgnorePatterns: [
    "/node_modules/(?!(next|@prisma/client|@radix-ui|lucide-react))",
  ],
  testTimeout: 60000,
};
