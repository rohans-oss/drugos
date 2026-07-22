/**
 * FE-047 ROOT FIX: shared pagination helper.
 *
 * The previous list endpoints (/api/evidence-package, /api/notifications,
 * /api/team, /api/auth/activity) each did `take: 50` (or `take: 20`, or
 * no limit at all) with no offset/cursor parameter. Power users with more
 * than 50 records could not access their older data via the API — the list
 * silently truncated. This helper enforces a consistent, safe pagination
 * envelope across all list endpoints:
 *
 *   { items: T[], total: number, hasMore: boolean, limit: number, offset: number }
 *
 * `limit` is capped at 100 (per the issue spec) and defaults to 50. `offset`
 * is a non-negative integer. Callers pass the raw `URLSearchParams` and the
 * helper extracts and validates the params.
 */

export const DEFAULT_PAGE_LIMIT = 50;
export const MAX_PAGE_LIMIT = 100;

export interface PaginatedQuery {
  limit: number;
  offset: number;
}

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  hasMore: boolean;
  limit: number;
  offset: number;
}

/**
 * Extract and validate `limit` + `offset` from a URLSearchParams.
 * Returns sane defaults (limit=50, offset=0) when the params are absent
 * or invalid. Always clamps limit to [1, 100] and offset to [0, 2^31-1].
 *
 * BE-071 ROOT FIX (v115, LOW): the previous code accepted `limit=0`
 * (clamped to default) but did NOT explicitly reject `limit=-1` —
 * `Number.parseInt("-1")` returns -1, which fails the `> 0` check and
 * falls through to the default. The behavior was correct (no bug),
 * but the code was unclear about intent. The fix adds an explicit
 * comment and a stricter check that rejects ANY non-positive limit
 * (including 0, -1, -100, etc.) with the same default fallback.
 *
 * This is a defense-in-depth measure: if a future refactor changes
 * the check to `>= 0` (allowing 0), the explicit `Math.max(1, ...)`
 * clamp below ensures limit is ALWAYS at least 1.
 */
export function parsePagination(params: URLSearchParams): PaginatedQuery {
  const rawLimit = Number.parseInt(params.get("limit") || "", 10);
  const rawOffset = Number.parseInt(params.get("offset") || "", 10);
  // BE-071: explicit rejection of non-positive limits. `Number.isFinite`
  // rules out NaN, Infinity, -Infinity. The `> 0` check rules out 0
  // and negative values. The result is clamped to [1, MAX_PAGE_LIMIT]
  // via Math.min + Math.max so a future refactor that changes the
  // default cannot produce limit=0 (which would break Prisma's `take`).
  const limit = Number.isFinite(rawLimit) && rawLimit > 0
    ? Math.max(1, Math.min(rawLimit, MAX_PAGE_LIMIT))
    : DEFAULT_PAGE_LIMIT;
  const offset = Number.isFinite(rawOffset) && rawOffset >= 0
    ? Math.min(rawOffset, 2_147_483_647) // clamp to INT32_MAX for SQL safety
    : 0;
  return { limit, offset };
}

/**
 * Build the standard paginated envelope from a Prisma findMany result and
 * a total count. `hasMore` is true iff offset + items.length < total.
 */
export function buildPaginatedResponse<T>(
  items: T[],
  total: number,
  query: PaginatedQuery
): PaginatedResponse<T> {
  return {
    items,
    total,
    hasMore: query.offset + items.length < total,
    limit: query.limit,
    offset: query.offset,
  };
}
