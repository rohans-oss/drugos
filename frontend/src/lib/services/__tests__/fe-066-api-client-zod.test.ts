/**
 * FE-066 ROOT FIX tests: api-client request<T> zod schema validation.
 *
 * Verifies:
 *   1. When a schema is provided AND the response matches, the parsed body
 *      is returned.
 *   2. When a schema is provided AND the response does NOT match, an
 *      ApiError with error="response_shape_mismatch" and status=0 is thrown.
 *   3. When no schema is provided, the old `as T` behavior is preserved
 *      (backward compat).
 *
 * These tests stub global.fetch so they can run in node without a server.
 */

import { z } from "zod";
import type { ZodType } from "zod";

// The `request` function is not exported from api-client. We test it
// indirectly via the `api.me()` helper, which now passes a schema (FE-066
// demonstration). We stub fetch to return controlled responses.

import { api, ApiError } from "@/lib/api-client";

// api-client uses `window.dispatchEvent` on 401 — guard for node env.
(globalThis as any).window = undefined;

describe("FE-066: api-client request<T> zod schema validation", () => {
  const ORIGINAL_FETCH = global.fetch;

  afterEach(() => {
    global.fetch = ORIGINAL_FETCH;
  });

  test("api.me() returns parsed user when response matches the schema", async () => {
    global.fetch = jest.fn(async () => {
      return new Response(
        JSON.stringify({
          user: {
            id: "cur123",
            email: "test@example.com",
            name: "Test User",
            role: "researcher",
          },
          organizations: [],
          activeOrganizationId: null,
        }),
        {
          status: 200,
          headers: { "Content-Type": "application/json" },
        }
      );
    }) as unknown as typeof fetch;

    const result = await api.me();
    expect(result.user.email).toBe("test@example.com");
    expect(result.user.name).toBe("Test User");
  });

  test("api.me() throws response_shape_mismatch when server returns a different shape", async () => {
    // Server returns `profile` instead of `user` — a contract drift the
    // pre-FE-066 code would have silently swallowed.
    global.fetch = jest.fn(async () => {
      return new Response(
        JSON.stringify({
          profile: { id: "cur123", email: "x@example.com" },
          orgs: [],
        }),
        {
          status: 200,
          headers: { "Content-Type": "application/json" },
        }
      );
    }) as unknown as typeof fetch;

    await expect(api.me()).rejects.toMatchObject({
      error: "response_shape_mismatch",
      status: 0,
    });
  });

  test("api.me() throws response_shape_mismatch when user.email is missing", async () => {
    // Email is required by the schema. Server omits it → mismatch.
    global.fetch = jest.fn(async () => {
      return new Response(
        JSON.stringify({
          user: { id: "cur123", name: "No Email" }, // missing email + role
          organizations: [],
          activeOrganizationId: null,
        }),
        {
          status: 200,
          headers: { "Content-Type": "application/json" },
        }
      );
    }) as unknown as typeof fetch;

    await expect(api.me()).rejects.toMatchObject({
      error: "response_shape_mismatch",
    });
  });

  test("api.logout() (no schema) preserves backward-compat: returns body as-is", async () => {
    global.fetch = jest.fn(async () => {
      return new Response(JSON.stringify({ ok: true }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    }) as unknown as typeof fetch;

    const result = await api.logout();
    expect((result as any).ok).toBe(true);
  });
});
