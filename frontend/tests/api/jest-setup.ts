/**
 * Jest setup — mocks `next/headers` cookies() so route handlers can be
 * unit-tested in isolation (without a running Next.js server).
 *
 * TASK-274..278 ROOT FIX: The prior tests couldn't call route handlers
 * directly because `cookies()` from `next/headers` throws "called outside
 * a request scope" when there's no Next.js request pipeline. The tests
 * either (a) used source-code regex matching (the "fake test" pattern
 * the user complained about) or (b) skipped route-handler testing
 * entirely.
 *
 * This mock replaces `cookies()` with a function that reads from a
 * global `__TEST_COOKIES` object. The test helper `setTestCookies()`
 * (exported below) lets individual tests set the cookies that
 * `getAuthenticatedUser()` will see.
 *
 * Usage in a test:
 *
 *   import { setTestCookies, clearTestCookies } from "@/tests/api/jest-setup";
 *
 *   beforeEach(() => clearTestCookies());
 *
 *   it("returns 403 for a researcher", async () => {
 *     setTestCookies({ drugos_access: "<jwt>" });
 *     const req = await buildReq("/api/admin/users");
 *     const res = await adminUsersGet(req);
 *     expect(res.status).toBe(403);
 *   });
 *
 * The mock is installed via jest's `setupFilesAfterEnv` config.
 */

import { jest } from "@jest/globals";

// Global cookie store — tests set this via setTestCookies().
(globalThis as any).__TEST_COOKIES = {} as Record<string, string>;

/**
 * Set the cookies that `cookies()` from `next/headers` will return.
 * Pass an object of { name: value } pairs.
 */
export function setTestCookies(cookies: Record<string, string>): void {
  (globalThis as any).__TEST_COOKIES = { ...cookies };
}

/**
 * Clear all test cookies.
 */
export function clearTestCookies(): void {
  (globalThis as any).__TEST_COOKIES = {};
}

// Mock `next/headers` cookies() to read from the global. The mock
// returns an object with `get(name)` and `set(name, value, opts)`
// methods that match the real API.
jest.mock("next/headers", () => ({
  cookies: jest.fn(async () => {
    const store = (globalThis as any).__TEST_COOKIES || {};
    return {
      get: (name: string) => (name in store ? { name, value: store[name] } : undefined),
      set: (name: string, value: string, _opts?: unknown) => {
        store[name] = value;
        (globalThis as any).__TEST_COOKIES = store;
      },
      delete: (name: string) => {
        delete store[name];
        (globalThis as any).__TEST_COOKIES = store;
      },
      getAll: () => Object.entries(store).map(([name, value]) => ({ name, value: value as string })),
    };
  }),
  headers: jest.fn(async () => {
    // Headers mock — tests rarely need to read headers, but some code
    // paths (API key auth) check the Authorization header. We return an
    // empty Headers object by default.
    return new Headers();
  }),
}));
