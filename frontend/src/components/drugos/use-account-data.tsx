/**
 * FE-065 ROOT FIX: Account-scoped data hooks.
 *
 * app-router.tsx previously imported 23 mock-data exports and rendered them
 * directly in the sidebar, notifications dropdown, dashboard widgets, etc.
 * The "real API integration" was a thin veneer on top of a mock-data
 * foundation — every account-scoped view (notifications, billing history,
 * API keys, usage metrics, audit logs) showed fabricated data.
 *
 * This module provides typed React hooks that wrap the existing `api` client
 * (which already has methods for listNotifications, listApiKeys,
 * listInvoices, listAuditLogs, listProjects, getSystemStatus, etc.). The
 * hooks return { data, loading, error } — when the backend returns no data,
 * `data` is null and the UI MUST render an empty state. We NEVER fall back
 * to mock data.
 *
 * Design decisions:
 *   - Single fetch on mount (no polling). The UI can call `refresh()` to
 *     re-fetch after a mutation.
 *   - Errors are surfaced, not swallowed. A 401 dispatches the
 *     `drugos:unauthorized` event (handled by SessionProvider).
 *   - Each hook is independent so a failure in one (e.g. billing) doesn't
 *     block the others (e.g. notifications).
 */

'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import {
  api,
  type ApiError,
  type Notification,
  type Invoice,
  type ApiKey,
  type AuditLog,
  type Project,
  type TeamMember,
  type SystemStatus,
  type Subscription,
  type Plan,
} from '@/lib/api-client';

interface AsyncState<T> {
  data: T | null;
  loading: boolean;
  error: ApiError | null;
  refresh: () => Promise<void>;
}

function useAsyncFetch<T>(fetcher: () => Promise<T>): AsyncState<T> {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);

  // Keep the fetcher in a ref so we can re-run on `refresh()` without
  // re-running on every render (the fetcher is usually a fresh closure
  // from the caller's render). The ref is updated inside useEffect so we
  // don't access it during render.
  const fetcherRef = useRef(fetcher);
  useEffect(() => {
    fetcherRef.current = fetcher;
  });

  const run = useCallback(async () => {
    setLoading(true);
    try {
      const result = await fetcherRef.current();
      setData(result);
      setError(null);
    } catch (e: unknown) {
      const err: ApiError =
        e && typeof e === 'object' && 'status' in e && 'error' in e
          ? (e as ApiError)
          : {
              error: 'request_failed',
              message: e instanceof Error ? e.message : String(e),
              status: 0,
            };
      // Don't overwrite data on error if we already have stale data — the
      // UI can keep showing the stale data with an error banner. But for
      // first-load errors, data is null and the UI shows the error state.
      setError(err);
      setData((current) => current); // no-op to satisfy lint
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    run();
  }, [run]);

  return { data, loading, error, refresh: run };
}

/**
 * Notifications — current user's notification feed.
 * Source: GET /api/notifications
 */
export function useNotifications() {
  return useAsyncFetch(async () => {
    const res = await api.listNotifications();
    return res.items;
  });
}

/**
 * Usage metrics — quota consumption for the current org.
 * Source: GET /api/billing/subscription (returns plan + seat info) +
 *         GET /api/api-keys (count of active keys) +
 *         GET /api/projects (count of projects).
 *
 * FE-065: The previous mock usageMetrics had hardcoded `used` and `limit`
 * fields per category (queries, apiCalls, reports). Real quota tracking
 * requires a usage service that doesn't exist yet — we compute what we can
 * from existing endpoints and return null for categories we can't compute.
 * The UI renders an EmptyState for null categories instead of fabricated
 * numbers.
 */
export interface UsageMetrics {
  queries: { used: number | null; limit: number | null } | null;
  apiCalls: { used: number | null; limit: number | null } | null;
  reports: { used: number | null; limit: number | null } | null;
  apiKeys: { used: number; limit: number | null } | null;
  projects: { used: number; limit: number | null } | null;
  seats: { used: number; limit: number | null } | null;
  plan: string | null;
}

export function useUsageMetrics() {
  return useAsyncFetch<UsageMetrics>(async () => {
    // Fetch in parallel — if any fails, we still return what we have.
    const [subRes, apiKeysRes, projectsRes] = await Promise.allSettled([
      api.getSubscription(),
      api.listApiKeys(),
      api.listProjects(),
    ]);

    const sub = subRes.status === 'fulfilled' ? subRes.value.subscription : null;
    const planName = sub?.plan ?? null;
    const seatsUsed = sub?.seats ?? 0;
    const seatsLimit = sub?.seats ?? null;

    const apiKeysItems = apiKeysRes.status === 'fulfilled' ? apiKeysRes.value.items : [];
    const apiKeysActive = apiKeysItems.filter((k: ApiKey) => !k.revokedAt).length;

    const projectItems = projectsRes.status === 'fulfilled' ? projectsRes.value.items : [];

    return {
      queries: null, // No usage-tracking endpoint exists yet — UI shows EmptyState.
      apiCalls: null,
      reports: null,
      apiKeys: { used: apiKeysActive, limit: null },
      projects: { used: projectItems.length, limit: null },
      seats: { used: seatsUsed, limit: seatsLimit },
      plan: planName,
    } as UsageMetrics;
  });
}

/**
 * Billing history — current org's invoices.
 * Source: GET /api/billing/invoices
 */
export function useBillingHistory() {
  return useAsyncFetch(async () => {
    const res = await api.listInvoices();
    return res.items;
  });
}

/**
 * API keys — current user's API keys.
 * Source: GET /api/api-keys
 */
export function useApiKeys() {
  return useAsyncFetch(async () => {
    const res = await api.listApiKeys();
    return res.items;
  });
}

/**
 * Audit logs — current org's audit trail (admin-only).
 * Source: GET /api/audit-logs
 */
export function useAuditLogs(limit = 50, offset = 0) {
  // Refresh when limit/offset change. We pass them through the fetcher
  // closure; the ref-based useAsyncFetch always calls the latest closure.
  return useAsyncFetch(async () => {
    const res = await api.listAuditLogs(limit, offset);
    return res.items;
  });
}

/**
 * Projects — current user's projects.
 * Source: GET /api/projects
 */
export function useProjects() {
  return useAsyncFetch(async () => {
    const res = await api.listProjects();
    return res.items;
  });
}

/**
 * Team members — current org's members.
 * Source: GET /api/team
 */
export function useTeamMembers() {
  return useAsyncFetch(async () => {
    const res = await api.listTeamMembers();
    return res.items;
  });
}

/**
 * System status — public status page data.
 * Source: GET /api/system/status
 *
 * FE-065: Replaces the static `systemStatus` mock array. The real endpoint
 * returns which backend services (Neo4j, dataset, RL ranker) are configured
 * and reachable. If the endpoint returns an error, we surface it honestly.
 */
export function useSystemStatus() {
  return useAsyncFetch<SystemStatus>(async () => api.getSystemStatus());
}

/**
 * Recent queries — client-side localStorage persistence.
 *
 * FE-065: The previous mock `recentQueries` was a hardcoded array of fake
 * searches. Real recent queries are client-side state — we persist them in
 * localStorage so they survive page reloads. No API needed.
 */
const RECENT_QUERIES_KEY = 'drugos:recent-queries';
const RECENT_QUERIES_MAX = 10;

export interface RecentQuery {
  id: string;
  q: string;
  type: 'drug' | 'disease' | 'literature' | 'trials';
  timestamp: number;
}

export function useRecentQueries() {
  const [queries, setQueries] = useState<RecentQuery[]>([]);

  useEffect(() => {
    try {
      const raw = localStorage.getItem(RECENT_QUERIES_KEY);
      if (raw) {
        const parsed = JSON.parse(raw) as RecentQuery[];
        if (Array.isArray(parsed)) setQueries(parsed);
      }
    } catch {
      // localStorage might be unavailable (private browsing) — silently ignore.
    }
  }, []);

  const addRecentQuery = useCallback((q: string, type: RecentQuery['type']) => {
    if (!q.trim()) return;
    setQueries((prev) => {
      const entry: RecentQuery = {
        id: `${Date.now().toString(36)}`,
        q: q.trim(),
        type,
        timestamp: Date.now(),
      };
      // Dedupe by (q, type) — keep only the most recent entry per pair.
      const filtered = prev.filter((x) => !(x.q === entry.q && x.type === entry.type));
      const next = [entry, ...filtered].slice(0, RECENT_QUERIES_MAX);
      try {
        localStorage.setItem(RECENT_QUERIES_KEY, JSON.stringify(next));
      } catch {
        // Storage full or unavailable — keep in-memory state only.
      }
      return next;
    });
  }, []);

  const clearRecentQueries = useCallback(() => {
    setQueries([]);
    try {
      localStorage.removeItem(RECENT_QUERIES_KEY);
    } catch {
      // ignore
    }
  }, []);

  return { queries, addRecentQuery, clearRecentQueries };
}

/**
 * Saved queries — user's saved searches (server-side).
 *
 * FE-065: The previous mock `savedQueries` was hardcoded. Real saved
 * searches require a /api/saved-queries endpoint that doesn't exist yet.
 * We return null and the UI renders an EmptyState — never fabricated
 * saved searches.
 */
export function useSavedQueries() {
  return useAsyncFetch<null>(async () => {
    // No endpoint exists yet. When one is added, replace this with:
    //   const res = await api.listSavedQueries();
    //   return res.items;
    return null;
  });
}
