/**
 * FE-008 ROOT FIX unit tests: Cypher read-only validator.
 *
 * Verifies the validateReadOnlyCypher function used by the
 * /api/knowledge-graph POST handler. This function is the second layer
 * of defense (after the role gate) — it rejects any Cypher containing
 * write operations (CREATE, DELETE, SET, REMOVE, MERGE, DROP, CALL, etc.)
 * BEFORE the query is forwarded to the downstream KG service.
 */
import { validateReadOnlyCypher } from "../cypher-validator";

describe("FE-008: Cypher read-only validator", () => {
  test("accepts a simple MATCH/RETURN query", () => {
    const r = validateReadOnlyCypher("MATCH (n:Drug) RETURN n LIMIT 10");
    expect(r.ok).toBe(true);
  });

  test("accepts OPTIONAL MATCH", () => {
    const r = validateReadOnlyCypher(
      "OPTIONAL MATCH (n:Drug)-[:treats]->(d:Disease) RETURN n, d"
    );
    expect(r.ok).toBe(true);
  });

  test("accepts WITH / RETURN composition", () => {
    const r = validateReadOnlyCypher(
      "MATCH (n:Drug) WITH n LIMIT 10 RETURN n"
    );
    expect(r.ok).toBe(true);
  });

  test("rejects CREATE", () => {
    const r = validateReadOnlyCypher("CREATE (n:Drug {name: 'foo'})");
    expect(r.ok).toBe(false);
    expect(r.reason).toMatch(/CREATE/i);
  });

  test("rejects DELETE", () => {
    const r = validateReadOnlyCypher("MATCH (n:Drug) DELETE n");
    expect(r.ok).toBe(false);
    expect(r.reason).toMatch(/DELETE/i);
  });

  test("rejects SET", () => {
    const r = validateReadOnlyCypher("MATCH (n:Drug) SET n.foo = 'bar'");
    expect(r.ok).toBe(false);
    expect(r.reason).toMatch(/SET/i);
  });

  test("rejects MERGE", () => {
    const r = validateReadOnlyCypher("MERGE (n:Drug {name: 'foo'})");
    expect(r.ok).toBe(false);
    expect(r.reason).toMatch(/MERGE/i);
  });

  test("rejects DROP", () => {
    const r = validateReadOnlyCypher("MATCH (n) DROP DATABASE x");
    expect(r.ok).toBe(false);
  });

  test("rejects CALL (procedure invocation)", () => {
    const r = validateReadOnlyCypher(
      "CALL apoc.destroyNode(node) YIELD value RETURN value"
    );
    expect(r.ok).toBe(false);
    expect(r.reason).toMatch(/CALL/i);
  });

  test("rejects DETACH DELETE", () => {
    const r = validateReadOnlyCypher("MATCH (n) DETACH DELETE n");
    expect(r.ok).toBe(false);
  });

  test("rejects UNWIND (used in batch writes)", () => {
    const r = validateReadOnlyCypher(
      "UNWIND $rows AS row CREATE (n:Drug) SET n.name = row.name"
    );
    expect(r.ok).toBe(false);
  });

  test("rejects empty query", () => {
    const r = validateReadOnlyCypher("");
    expect(r.ok).toBe(false);
  });

  test("rejects query that does not start with a read verb", () => {
    const r = validateReadOnlyCypher("WHERE 1=1 RETURN 1");
    expect(r.ok).toBe(false);
    expect(r.reason).toMatch(/MATCH|OPTIONAL MATCH|WITH|RETURN/i);
  });

  test("rejects multiple statements (semicolon-separated)", () => {
    const r = validateReadOnlyCypher("MATCH (n) RETURN n; MATCH (m) RETURN m");
    expect(r.ok).toBe(false);
    expect(r.reason).toMatch(/Multiple/i);
  });

  test("does NOT reject a semicolon at the very end (single statement)", () => {
    const r = validateReadOnlyCypher("MATCH (n) RETURN n;");
    expect(r.ok).toBe(true);
  });

  test("does not false-positive on semicolons inside string literals", () => {
    const r = validateReadOnlyCypher(
      "MATCH (n) WHERE n.name = 'a;b;c' RETURN n"
    );
    expect(r.ok).toBe(true);
  });

  test("rejects oversized query (> 5000 chars)", () => {
    const r = validateReadOnlyCypher("MATCH (n) RETURN n " + "x".repeat(6000));
    expect(r.ok).toBe(false);
    expect(r.reason).toMatch(/too long/i);
  });
});
