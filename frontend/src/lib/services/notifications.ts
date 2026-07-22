/**
 * TASK-268 ROOT FIX: Notification triggers.
 *
 * The audit (Task 268) found that the platform had NO notification
 * triggers. The Notification table existed and GET /api/notifications
 * was wired to it — but NOTHING ever WROTE to the table. So the
 * notification bell icon in the UI always showed "0 unread" — even
 * when a teammate commented on your project, when your billing invoice
 * was ready, or when your hypothesis validation completed.
 *
 * This module provides typed helpers for the THREE notification
 * triggers required by Task 268:
 *
 *   1. notifyProjectComment — fires when a user comments on a project.
 *      Notifies ALL project members EXCEPT the commenter (no self-notifs).
 *
 *   2. notifyInvoiceReady — fires when a billing invoice is created.
 *      Notifies every org member with the `billing` or `owner` role
 *      (they're the ones who need to pay).
 *
 *   3. notifyHypothesisValidationComplete — fires when a hypothesis
 *      validation completes. Notifies the user who submitted it AND
 *      the project's PI (principal investigator) if different.
 *
 * Each helper:
 *   - Writes a row to the Notification table via Prisma.
 *   - Is NON-BLOCKING (best-effort). A notification failure must NOT
 *     break the user action that triggered it — a comment should still
 *     post even if the notification write fails. We log the failure
 *     to stderr so operators can monitor for systemic issues.
 *   - Is IDEMPOTENT in design (the caller decides when to fire — we
 *     don't dedupe here because the caller's "when" is the source of
 *     truth).
 *
 * SECURITY: every helper takes the ACTOR's userId (the user who
 * triggered the notification) and derives the RECIPIENT list from the
 * DB. The caller CANNOT forge a notification as another user — the
 * `userId` column is set from the recipient's actual User row.
 */

import { db } from "@/lib/db";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type NotificationType = "info" | "success" | "warning" | "error";

export interface CreateNotificationInput {
  userId: string;
  type: NotificationType;
  title: string;
  body: string;
}

// ---------------------------------------------------------------------------
// Core helper — write a single notification, best-effort.
// ---------------------------------------------------------------------------

async function createNotification(input: CreateNotificationInput): Promise<void> {
  try {
    await db.notification.create({
      data: {
        userId: input.userId,
        type: input.type,
        title: input.title.slice(0, 500), // hard cap for storage safety
        body: input.body.slice(0, 10000),
      },
    });
  } catch (e) {
    // Non-blocking — a notification failure must NOT break the user
    // action that triggered it. Log to stderr so operators can monitor
    // for systemic issues (e.g. the Notification table being dropped).
    console.error("[NOTIFICATION] Failed to write notification:", {
      recipientUserId: input.userId,
      type: input.type,
      title: input.title,
      error: e instanceof Error ? e.message : String(e),
    });
  }
}

async function createNotifications(inputs: CreateNotificationInput[]): Promise<void> {
  await Promise.all(inputs.map(createNotification));
}

// ---------------------------------------------------------------------------
// Trigger 1: Project comment
// ---------------------------------------------------------------------------

/**
 * Notify all members of a project (except the commenter) that a new
 * comment was posted.
 *
 * @param projectId — the project the comment was posted on.
 * @param commenterId — the user who posted the comment (excluded from
 *   the recipient list — no self-notifications).
 * @param commentBodyPreview — the first ~200 chars of the comment, for
 *   the notification preview.
 */
export async function notifyProjectComment(
  projectId: string,
  commenterId: string,
  commentBodyPreview: string,
): Promise<void> {
  // Look up the project to get its name + organization.
  const project = await db.project.findUnique({
    where: { id: projectId },
    select: { id: true, name: true, organizationId: true },
  });
  if (!project) {
    // Project was deleted between the comment POST and this notification
    // — nothing to do.
    return;
  }

  // Find all members of the project's org. The project's "members" are
  // the org's members (the current data model has no per-project
  // membership — org membership IS project membership, scoped by
  // organizationId).
  const memberships = await db.organizationMember.findMany({
    where: {
      organizationId: project.organizationId,
      // Exclude the commenter — no self-notifications.
      userId: { not: commenterId },
    },
    select: { userId: true },
  });

  // Look up the commenter's name for the notification body.
  const commenter = await db.user.findUnique({
    where: { id: commenterId },
    select: { name: true, email: true },
  });
  const commenterName = commenter?.name || commenter?.email || "A teammate";

  const preview = commentBodyPreview.length > 200
    ? commentBodyPreview.slice(0, 200) + "…"
    : commentBodyPreview;

  await createNotifications(
    memberships.map((m) => ({
      userId: m.userId,
      type: "info" as NotificationType,
      title: `New comment on "${project.name}"`,
      body: `${commenterName} commented: ${preview}`,
    })),
  );
}

// ---------------------------------------------------------------------------
// Trigger 2: Billing invoice ready
// ---------------------------------------------------------------------------

/**
 * Notify every billing/owner member of an org that a new invoice is ready.
 *
 * @param organizationId — the org the invoice belongs to.
 * @param invoiceNumber — the human-readable invoice number (e.g. INV-2026-01-ABC123).
 * @param amountCents — the invoice amount in cents (e.g. 9900 = $99.00).
 * @param currency — the ISO currency code (e.g. "usd").
 */
export async function notifyInvoiceReady(
  organizationId: string,
  invoiceNumber: string,
  amountCents: number,
  currency: string,
): Promise<void> {
  // Find every member of the org with the `billing` or `owner` role —
  // they're the ones who need to see the invoice.
  const memberships = await db.organizationMember.findMany({
    where: {
      organizationId,
      role: { in: ["billing", "owner"] },
    },
    select: { userId: true },
  });

  const amountFormatted = new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: currency.toUpperCase(),
  }).format(amountCents / 100);

  await createNotifications(
    memberships.map((m) => ({
      userId: m.userId,
      type: "info" as NotificationType,
      title: `Invoice ${invoiceNumber} is ready`,
      body: `A new invoice for ${amountFormatted} is ready. View it in Billing → Invoices.`,
    })),
  );
}

// ---------------------------------------------------------------------------
// Trigger 3: Hypothesis validation complete
// ---------------------------------------------------------------------------

/**
 * Notify the user who submitted a hypothesis validation that it's complete.
 *
 * Also notify the org's PIs (principal investigators) so they can review
 * the validation result.
 *
 * @param submitterId — the user who submitted the validation.
 * @param organizationId — the org the validation belongs to.
 * @param drug — the drug name (for the notification body).
 * @param disease — the disease name (for the notification body).
 * @param outcome — the validation outcome (validated_positive, etc.).
 */
export async function notifyHypothesisValidationComplete(
  submitterId: string,
  organizationId: string,
  drug: string,
  disease: string,
  outcome: string,
): Promise<void> {
  // Format the outcome for human readers.
  const outcomeLabels: Record<string, string> = {
    validated_positive: "validated (positive)",
    validated_negative: "validated (negative)",
    validated_toxic: "validated (toxicity signal)",
    invalidated: "invalidated",
  };
  const outcomeLabel = outcomeLabels[outcome] || outcome;

  // Notify the submitter.
  await createNotification({
    userId: submitterId,
    type: outcome === "validated_positive" ? "success" : outcome === "validated_toxic" ? "warning" : "info",
    title: `Hypothesis validation complete: ${drug} → ${disease}`,
    body: `Your hypothesis for ${drug} → ${disease} has been ${outcomeLabel}. The result has been written back to Phase 1 (dataset), Phase 2 (Neo4j edge), and Phase 3 (retrain trigger).`,
  });

  // Notify the org's PIs (principal investigators) — they need to
  // review the validation result. We use the `role` field (functional
  // role), not `platformRole`.
  const pis = await db.user.findMany({
    where: {
      organizationMemberships: {
        some: { organizationId },
      },
      role: "pi",
    },
    select: { id: true },
  });
  // Exclude the submitter if they're a PI (they already got notified).
  const piRecipients = pis.filter((p) => p.id !== submitterId);

  await createNotifications(
    piRecipients.map((p) => ({
      userId: p.id,
      type: "info" as NotificationType,
      title: `Hypothesis validation ready for review: ${drug} → ${disease}`,
      body: `A teammate submitted a validation for ${drug} → ${disease} (${outcomeLabel}). Review the writeback in the hypothesis detail view.`,
    })),
  );
}
