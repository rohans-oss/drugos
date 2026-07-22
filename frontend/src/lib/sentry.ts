/**
 * IN-043 ROOT FIX (v115, MEDIUM): opt-in Sentry integration.
 *
 * ROOT CAUSE: the previous codebase had NO error tracking SDK. When
 * an unhandled exception occurred in production (e.g., a Prisma
 * query threw because the DB was down, or a torch.load failed
 * because the checkpoint was corrupted), the error went to
 * stdout/stderr — captured by Docker's json-file log driver but
 * NOT aggregated, NOT deduplicated, NOT alerted on. For a patient-
 * safety platform, an unhandled error in /api/predict that returns
 * a 500 to a clinician is a P0 incident — but without Sentry, no
 * one knows it happened until the clinician reports it.
 *
 * ROOT FIX: this module provides a LAZY-LOAD Sentry integration.
 * It's opt-in — operators set `SENTRY_DSN` in the env to enable it.
 * If SENTRY_DSN is unset, the module is a no-op (all functions
 * return immediately without doing anything). If SENTRY_DSN is set
 * but `@sentry/nextjs` is not installed, the module logs a warning
 * and falls back to no-op — it does NOT crash the app.
 *
 * This pattern is the safest way to add an observability layer
 * without making it a hard runtime dependency. Operators who want
 * Sentry install the package and set the env var; operators who
 * don't want Sentry see no impact.
 *
 * Usage (in any route handler or service):
 *   import { captureException, setTag } from "@/lib/sentry";
 *   try {
 *     await riskyOperation();
 *   } catch (e) {
 *     captureException(e, { tags: { route: "/api/predict" } });
 *     throw e;
 *   }
 *
 * The module is safe to import at the top of any file — it only
 * initializes Sentry on first use, and only if SENTRY_DSN is set.
 */

let sentryInitialized = false;
let sentryAvailable = false;

interface SentryLike {
  captureException: (err: unknown, opts?: { tags?: Record<string, string>; extra?: Record<string, unknown> }) => string;
  setTag: (key: string, value: string) => void;
  setUser: (user: { id?: string; email?: string } | null) => void;
  addBreadcrumb: (crumb: { message: string; level?: "info" | "warning" | "error"; data?: Record<string, unknown> }) => void;
}

let sentryClient: SentryLike | null = null;

/**
 * Lazily initialize Sentry on first use. Safe to call multiple times —
 * the init check is idempotent. Returns the Sentry client if it's
 * available, or null if it's not (SENTRY_DSN unset or package missing).
 */
async function getSentry(): Promise<SentryLike | null> {
  // Fast path — already initialized.
  if (sentryInitialized) {
    return sentryClient;
  }

  sentryInitialized = true;

  const dsn = process.env.SENTRY_DSN;
  if (!dsn) {
    // SENTRY_DSN is unset — Sentry is disabled. This is the default
    // for local dev and for deployments that don't want Sentry.
    return null;
  }

  try {
    // Dynamic import so the @sentry/nextjs package is only loaded
    // when SENTRY_DSN is set. If the package is not installed, this
    // throws — we catch and log a warning. The TypeScript types for
    // @sentry/nextjs are stubbed in src/types/optional-deps.d.ts so
    // the build does NOT fail when the package is absent.
    const Sentry = await import("@sentry/nextjs");
    const initOpts: Record<string, unknown> = {
      dsn,
      // Sample 10% of transactions for performance monitoring. Full
      // tracing is expensive — 10% is enough to spot trends without
      // overwhelming the Sentry quota.
      tracesSampleRate: 0.1,
      // Sample 100% of errors — these are the events we care about.
      // Errors are rare and high-signal; sampling them would lose
      // critical incidents.
      sampleRate: 1.0,
      // Release: git SHA if available, else the app version.
      release: process.env.SENTRY_RELEASE || process.env.npm_package_version,
      // Environment: production / staging / development.
      environment: process.env.SENTRY_ENVIRONMENT || process.env.NODE_ENV || "development",
      // Don't send PII — HIPAA / GDPR compliance. Sentry's default
      // already strips most PII, but we explicitly enable it.
      beforeSend(event: { request?: { headers?: Record<string, string> } }) {
        if (event?.request?.headers) {
          // Strip auth headers from error reports — never send
          // Authorization, Cookie, or X-CSRF-Token to Sentry.
          delete event.request.headers["authorization"];
          delete event.request.headers["Authorization"];
          delete event.request.headers["cookie"];
          delete event.request.headers["Cookie"];
          delete event.request.headers["x-csrf-token"];
          delete event.request.headers["X-CSRF-Token"];
        }
        return event;
      },
    };
    if (typeof Sentry.init === "function") {
      Sentry.init(initOpts);
    }
    sentryClient = {
      captureException: (err, opts) => {
        try {
          return Sentry.captureException(err, opts as Record<string, unknown>);
        } catch {
          return "";
        }
      },
      setTag: (key, value) => {
        try {
          Sentry.setTag(key, value);
        } catch {
          // no-op
        }
      },
      setUser: (user) => {
        try {
          Sentry.setUser(user as Record<string, unknown>);
        } catch {
          // no-op
        }
      },
      addBreadcrumb: (crumb) => {
        try {
          Sentry.addBreadcrumb(crumb as Record<string, unknown>);
        } catch {
          // no-op
        }
      },
    };
    sentryAvailable = true;
    console.info("[sentry] initialized — error tracking enabled.");
    return sentryClient;
  } catch (e) {
    // @sentry/nextjs is not installed. Log a clear warning so the
    // operator knows to either install the package OR unset SENTRY_DSN.
    console.warn(
      "[sentry] SENTRY_DSN is set but @sentry/nextjs is not installed.",
      "To enable error tracking: npm install @sentry/nextjs",
      "Or unset SENTRY_DSN to silence this warning.",
      "Error:",
      e instanceof Error ? e.message : String(e)
    );
    return null;
  }
}

/**
 * Capture an exception and send it to Sentry (if configured).
 * Safe to call from any context — never throws.
 *
 * @param err The error to capture. Accepts Error, string, or unknown.
 * @param opts Optional tags and extra context for the Sentry event.
 */
export async function captureException(
  err: unknown,
  opts?: { tags?: Record<string, string>; extra?: Record<string, unknown> }
): Promise<void> {
  const client = await getSentry();
  if (!client) return;
  client.captureException(err, opts);
}

/**
 * Synchronously capture an exception. Use this from non-async contexts
 * where you can't await (e.g., a top-level error handler). Falls back
 * to console.error if Sentry is not yet initialized — the next async
 * captureException call will properly initialize Sentry.
 */
export function captureExceptionSync(
  err: unknown,
  opts?: { tags?: Record<string, string>; extra?: Record<string, unknown> }
): void {
  if (sentryAvailable && sentryClient) {
    sentryClient.captureException(err, opts);
  } else {
    // Sentry not yet initialized — log to stderr and fire-and-forget
    // an async captureException call to initialize it.
    console.error("[sentry-not-yet-initialized]", err);
    void captureException(err, opts);
  }
}

/**
 * Set a Sentry tag (key-value pair attached to all subsequent events).
 * Useful for tagging events with the route name, user role, or org ID.
 */
export async function setTag(key: string, value: string): Promise<void> {
  const client = await getSentry();
  if (!client) return;
  client.setTag(key, value);
}

/**
 * Set the current user on Sentry events. Pass null to clear the user
 * (e.g., on logout).
 */
export async function setUser(user: { id?: string; email?: string } | null): Promise<void> {
  const client = await getSentry();
  if (!client) return;
  client.setUser(user);
}

/**
 * Add a breadcrumb — a lightweight event that's attached to the next
 * error report. Useful for tracing the sequence of operations that
 * led up to an error (e.g., "user clicked Validate", "API called",
 * "DB query executed", "error thrown").
 */
export async function addBreadcrumb(crumb: {
  message: string;
  level?: "info" | "warning" | "error";
  data?: Record<string, unknown>;
}): Promise<void> {
  const client = await getSentry();
  if (!client) return;
  client.addBreadcrumb(crumb);
}

/**
 * Returns true if Sentry is initialized and available. Useful for
 * conditional logging (e.g., only log a breadcrumb if Sentry is on).
 */
export function isSentryAvailable(): boolean {
  return sentryAvailable;
}
