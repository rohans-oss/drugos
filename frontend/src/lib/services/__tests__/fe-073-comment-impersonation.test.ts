/**
 * FE-073 ROOT FIX tests: comment impersonation guard.
 *
 * Verifies:
 *   1. addComment(projectId, userId, body) stores the comment with
 *      authorName derived from User.name || User.email — NEVER from a
 *      client-supplied value.
 *   2. The legacy 3-arg signature (projectId, authorName, body) is
 *      rejected — passing an attacker-controlled name in the 2nd-arg
 *      position throws and nothing is persisted.
 *   3. The Comment row stores userId (audit-traceable) and the
 *      ProjectActivity entry's actorName matches the derived authorName.
 *
 * NOTE ON TEST STRATEGY:
 *   The test DB infrastructure in this repo is currently broken — the
 *   Prisma schema is set to `provider = "postgresql"` (FE-019 root fix)
 *   but `tests/api/env.ts` sets `DATABASE_URL = file:...test.db` (sqlite).
 *   Prisma rejects this combination, so EVERY DB-backed test in
 *   `src/lib/services/__tests__/` fails at module-import time. This is a
 *   pre-existing infrastructure issue documented in the PR; it is NOT
 *   caused by my FE-073 fix.
 *
 *   To verify FE-073 WITHOUT depending on the broken DB infra, this test
 *   mocks the `@/lib/db` module. The impersonation-guard logic is what
 *   we're verifying — the DB is just a write, which a mock faithfully
 *   captures.
 */

const dbMock = {
  user: {
    findUnique: jest.fn(),
  },
  comment: {
    create: jest.fn(),
  },
  projectActivity: {
    create: jest.fn(),
  },
};
jest.mock("@/lib/db", () => ({ db: dbMock }));

import { addComment } from "@/lib/services/projects";

describe("FE-073: addComment rejects client-supplied authorName", () => {
  beforeEach(() => {
    dbMock.user.findUnique.mockReset();
    dbMock.comment.create.mockReset();
    dbMock.projectActivity.create.mockReset();
  });

  test("derives authorName from User.name when the user has a name", async () => {
    dbMock.user.findUnique.mockResolvedValue({
      name: "Dr. Legitimate PI",
      email: "pi@example.com",
    });
    dbMock.comment.create.mockImplementation(async ({ data }) => data);

    const comment = await addComment(
      "curproject000000000000001",
      "curuser0000000000000000001",
      "Looks promising — let's run a literature search."
    );

    expect(dbMock.user.findUnique).toHaveBeenCalledWith({
      where: { id: "curuser0000000000000000001" },
      select: { name: true, email: true },
    });
    expect(comment.authorName).toBe("Dr. Legitimate PI");
    expect(comment.userId).toBe("curuser0000000000000000001");
    expect(comment.body).toMatch(/promising/);
  });

  test("falls back to User.email when User.name is null", async () => {
    dbMock.user.findUnique.mockResolvedValue({
      name: null,
      email: "noname@example.com",
    });
    dbMock.comment.create.mockImplementation(async ({ data }) => data);

    const comment = await addComment(
      "curproject000000000000001",
      "curuser0000000000000000002",
      "Comment body"
    );

    expect(comment.authorName).toBe("noname@example.com");
  });

  test("throws when caller passes a legacy authorName in the 2nd-arg position (impersonation guard)", async () => {
    // A legacy caller passing "Dr. Smith (PI)" as the 2nd arg must be
    // rejected — this is the impersonation vector FE-073 closes.
    await expect(
      addComment(
        "curproject000000000000001",
        "Dr. Smith (PI)", // <-- attacker-controlled name in userId slot
        "Trying to impersonate"
      )
    ).rejects.toThrow(/userId/);

    // CRITICAL: NO comment row and NO activity row may have been written.
    expect(dbMock.comment.create).not.toHaveBeenCalled();
    expect(dbMock.projectActivity.create).not.toHaveBeenCalled();
  });

  test("throws when the userId does not correspond to an existing user", async () => {
    dbMock.user.findUnique.mockResolvedValue(null);

    await expect(
      addComment("curproject000000000000001", "curghost000000000000000001", "x")
    ).rejects.toThrow(/not found/);

    expect(dbMock.comment.create).not.toHaveBeenCalled();
  });

  test("emits a ProjectActivity entry with actorName matching the derived authorName", async () => {
    dbMock.user.findUnique.mockResolvedValue({
      name: "Activity Tester",
      email: "activity@example.com",
    });
    dbMock.comment.create.mockImplementation(async ({ data }) => data);

    await addComment(
      "curproject000000000000001",
      "curuser0000000000000000003",
      "Comment that should appear in activity feed."
    );

    expect(dbMock.projectActivity.create).toHaveBeenCalledWith({
      data: {
        projectId: "curproject000000000000001",
        type: "comment_added",
        actorName: "Activity Tester",
        summary: "Comment that should appear in activity feed.".slice(0, 120),
      },
    });
  });
});
