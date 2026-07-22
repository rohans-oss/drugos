/**
 * FE-008 ROOT FIX: shared Cypher validator.
 *
 * This module exports validateReadOnlyCypher — the function used by the
 * /api/knowledge-graph POST handler to whitelist read-only Cypher. It is
 * extracted into its own module so unit tests can exercise it without
 * spinning up the Next.js route handler.
 */

// BE-082 ROOT FIX: Align the max-length cap with the Zod schema
// (KnowledgeGraphBody.cypher = z.string().min(1).max(10_000)). The previous
// value of 5000 was inconsistent — a 7000-char query would PASS Zod
// validation but FAIL this validator with a confusing "max 5000 chars"
// error. Complex biomedical queries with multiple MATCH patterns, WHERE
// clauses, and RETURN projections can easily exceed 5000 chars. 10000
// is the agreed-upon contract; both layers now enforce the same limit.
const MAX_CYPHER_LENGTH = 10_000;

// Statements that mutate the graph or call procedures. We reject these
// BEFORE forwarding to the KG service. The check is case-insensitive and
// word-bounded so it doesn't false-positive on legitimate identifiers.
//
// BE-025 ROOT FIX: removed the contradictory `CALL\s+db\.labels` allowance
// from ALLOWED_TOP_LEVEL_VERBS below. The previous code allowed
// `CALL db.labels()` as a top-level verb BUT FORBID `CALL` as a keyword —
// the forbidden check ran first and rejected every `CALL` (including
// `CALL db.labels()`) before the top-level-verb check could match. The
// allowance was dead code, and the misleading regex suggested to
// maintainers that `CALL db.labels` was supported when it wasn't.
// Committing to blocking ALL `CALL` (including `db.labels`) is the
// cleaner choice — the Python KG service's own validator is the real
// gate, and the frontend validator's job is to reject write/mutation
// verbs, not to whitelist introspection procedures.
const FORBIDDEN_CYPHER_KEYWORDS =
  /\b(CREATE|MERGE|DELETE|DETACH\s+DELETE|SET|REMOVE|DROP|CALL|YIELD|UNWIND|FOREACH|LOAD\s+CSV|PERIODIC\s+COMMIT)\b/gi;

// The ONLY top-level verbs allowed in a read-only Cypher query.
// BE-025: `CALL db.labels` removed — see FORBIDDEN_CYPHER_KEYWORDS comment.
const ALLOWED_TOP_LEVEL_VERBS =
  /^\s*(MATCH|OPTIONAL\s+MATCH|WITH|RETURN)\b/i;

export function validateReadOnlyCypher(
  cypher: string
): { ok: boolean; reason?: string } {
  const trimmed = cypher.trim();
  if (!trimmed) return { ok: false, reason: "Cypher query is empty." };
  if (trimmed.length > MAX_CYPHER_LENGTH) {
    return {
      ok: false,
      reason: `Cypher query is too long (max ${MAX_CYPHER_LENGTH} chars).`,
    };
  }
  // Reject any forbidden keyword anywhere in the query.
  const forbiddenMatch = trimmed.match(FORBIDDEN_CYPHER_KEYWORDS);
  if (forbiddenMatch) {
    return {
      ok: false,
      reason: `Cypher contains a forbidden keyword: ${forbiddenMatch[0]}. Only read-only MATCH / OPTIONAL MATCH / WITH / RETURN queries are allowed via this endpoint.`,
    };
  }
  // The first non-comment token must be a read verb.
  if (!ALLOWED_TOP_LEVEL_VERBS.test(trimmed)) {
    return {
      ok: false,
      reason:
        "Cypher must start with MATCH, OPTIONAL MATCH, WITH, or RETURN. " +
        "Write operations (CREATE, DELETE, SET, etc.) are not permitted.",
    };
  }
  // Defensive: reject multiple statements (semicolon-separated). Strip
  // string literals AND backtick-quoted identifiers first so semicolons
  // inside strings or identifiers don't trip us.
  //
  // BE-026 ROOT FIX: the previous implementation stripped ONLY single and
  // double quoted strings before counting semicolons. Cypher supports
  // backtick-quoted identifiers for names with special characters (e.g.
  // `` `drug;DROP DATABASE` `` is a valid identifier — the semicolon
  // inside backticks is part of the identifier, NOT a statement
  // separator). Without stripping backtick-quoted identifiers, the
  // validator would count semicolons inside backticks as statement
  // separators, producing false positives: a legitimate query like
  // `MATCH (n:`my;label`) RETURN n` would be rejected as "multiple
  // statements" when it's actually one. Adding the backtick-stripping
  // regex makes the count consistent with Cypher's actual grammar.
  const stripped = trimmed
    .replace(/'(?:[^'\\]|\\.)*'/g, "''")   // strip single-quoted strings
    .replace(/"(?:[^"\\]|\\.)*"/g, '""')   // strip double-quoted strings
    .replace(/`(?:[^`\\]|\\.)*`/g, "``");  // BE-026: strip backtick-quoted identifiers
  const statementCount = (stripped.match(/;/g) || []).length;
  if (statementCount > 1 || (statementCount === 1 && !stripped.endsWith(";"))) {
    return {
      ok: false,
      reason: "Multiple Cypher statements are not allowed.",
    };
  }
  return { ok: true };
}
