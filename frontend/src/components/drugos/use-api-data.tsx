'use client';

/**
 * FE-001 ROOT FIX: Real API integration hooks for the core drug-repurposing
 * screens.
 *
 * The previous core-screens.tsx imported mock data and rendered it directly.
 * None of the screens called the real API routes. The api-client.ts defined
 * api.searchDrugs, api.searchDiseases, api.getSafety, api.buildEvidencePackage,
 * etc. — but a repo-wide grep confirmed these were NEVER called from any
 * component.
 *
 * This module provides React hooks that wrap those api-client methods with
 * proper loading/error states. The core screens now use these hooks instead
 * of importing mock data.
 *
 * Design decisions:
 *   - Every hook returns { data, loading, error }.
 *   - On error, we surface the real error message — we NEVER silently fall
 *     back to mock data. A pharma researcher seeing fabricated "Memantine 87
 *     for Huntington's" could act on fake data — patient safety violation.
 *   - The hooks are typed against the api-client interfaces so the UI gets
 *     compile-time safety.
 *   - Debouncing is built in for search-as-you-type hooks.
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import type { ReactNode } from 'react';
import { api, type ApiError } from '@/lib/api-client';
import { Card, CardContent } from '@/components/ui/card';

export interface AsyncState<T> {
  data: T | null;
  loading: boolean;
  error: ApiError | null;
}

/**
 * Debounce a fast-changing value (e.g. a search input) so we don't fire
 * an API request on every keystroke.
 */
function useDebouncedValue<T>(value: T, delayMs: number): T {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const t = setTimeout(() => setDebounced(value), delayMs);
    return () => clearTimeout(t);
  }, [value, delayMs]);
  return debounced;
}

/**
 * Search diseases via the real /api/diseases/search endpoint (backed by
 * NLM MeSH). Debounced by 300ms.
 */
export function useDiseaseSearch(query: string, minLength = 2) {
  const debounced = useDebouncedValue(query, 300);
  const [state, setState] = useState<AsyncState<Awaited<ReturnType<typeof api.searchDiseases>>>>({
    data: null,
    loading: false,
    error: null,
  });

  useEffect(() => {
    if (debounced.trim().length < minLength) {
      setState({ data: null, loading: false, error: null });
      return;
    }
    let cancelled = false;
    setState((s) => ({ ...s, loading: true, error: null }));
    api
      .searchDiseases(debounced)
      .then((data) => {
        if (!cancelled) setState({ data, loading: false, error: null });
      })
      .catch((error: ApiError) => {
        if (!cancelled) setState({ data: null, loading: false, error });
      });
    return () => {
      cancelled = true;
    };
  }, [debounced, minLength]);

  return state;
}

/**
 * Search drugs via the real /api/drugs/search endpoint (backed by RxNorm).
 */
export function useDrugSearch(query: string, minLength = 2) {
  const debounced = useDebouncedValue(query, 300);
  const [state, setState] = useState<AsyncState<Awaited<ReturnType<typeof api.searchDrugs>>>>({
    data: null,
    loading: false,
    error: null,
  });

  useEffect(() => {
    if (debounced.trim().length < minLength) {
      setState({ data: null, loading: false, error: null });
      return;
    }
    let cancelled = false;
    setState((s) => ({ ...s, loading: true, error: null }));
    api
      .searchDrugs(debounced)
      .then((data) => {
        if (!cancelled) setState({ data, loading: false, error: null });
      })
      .catch((error: ApiError) => {
        if (!cancelled) setState({ data: null, loading: false, error });
      });
    return () => {
      cancelled = true;
    };
  }, [debounced, minLength]);

  return state;
}

/**
 * Fetch safety data for a drug via the real /api/safety/[drug] endpoint
 * (backed by openFDA adverse event reports).
 */
export function useDrugSafety(drug: string | null) {
  const [state, setState] = useState<AsyncState<Awaited<ReturnType<typeof api.getSafety>>>>({
    data: null,
    loading: false,
    error: null,
  });

  useEffect(() => {
    if (!drug || drug.trim().length < 2) {
      setState({ data: null, loading: false, error: null });
      return;
    }
    let cancelled = false;
    setState({ data: null, loading: true, error: null });
    api
      .getSafety(drug)
      .then((data) => {
        if (!cancelled) setState({ data, loading: false, error: null });
      })
      .catch((error: ApiError) => {
        if (!cancelled) setState({ data: null, loading: false, error });
      });
    return () => {
      cancelled = true;
    };
  }, [drug]);

  return state;
}

/**
 * Search clinical trials via the real /api/clinical-trials/search endpoint
 * (backed by ClinicalTrials.gov v2).
 */
export function useClinicalTrialsSearch(params: {
  condition?: string;
  intervention?: string;
  limit?: number;
  pageToken?: string;
}) {
  const [state, setState] = useState<AsyncState<Awaited<ReturnType<typeof api.searchClinicalTrials>>>>({
    data: null,
    loading: false,
    error: null,
  });

  // Stringify params for the dep array so we re-fetch when any changes.
  const paramsKey = JSON.stringify(params);
  useEffect(() => {
    if (!params.condition && !params.intervention) {
      setState({ data: null, loading: false, error: null });
      return;
    }
    let cancelled = false;
    setState((s) => ({ ...s, loading: true, error: null }));
    // The api-client's searchClinicalTrials takes a single `q` string;
    // the underlying API route accepts condition + intervention + pageToken.
    // We build the query string manually to pass all params.
    const qs = new URLSearchParams();
    if (params.condition) qs.set("condition", params.condition);
    if (params.intervention) qs.set("intervention", params.intervention);
    if (params.limit) qs.set("limit", String(params.limit));
    if (params.pageToken) qs.set("pageToken", params.pageToken);
    fetch(`/api/clinical-trials/search?${qs.toString()}`, { credentials: "include" })
      .then(async (res) => {
        const text = await res.text();
        let body: any = null;
        if (text) {
          try { body = JSON.parse(text); } catch { body = { raw: text }; }
        }
        if (!res.ok) {
          throw {
            error: body?.error || "request_failed",
            message: body?.message || `Request failed with status ${res.status}`,
            status: res.status,
          } as ApiError;
        }
        return body as Awaited<ReturnType<typeof api.searchClinicalTrials>>;
      })
      .then((data) => {
        if (!cancelled) setState({ data, loading: false, error: null });
      })
      .catch((error: ApiError) => {
        if (!cancelled) setState({ data: null, loading: false, error });
      });
    return () => {
      cancelled = true;
    };
  }, [paramsKey]);

  return state;
}

/**
 * Search PubMed literature via the real /api/literature/search endpoint.
 */
export function useLiteratureSearch(query: string, minLength = 3) {
  const debounced = useDebouncedValue(query, 400);
  const [state, setState] = useState<AsyncState<Awaited<ReturnType<typeof api.searchLiterature>>>>({
    data: null,
    loading: false,
    error: null,
  });

  useEffect(() => {
    if (debounced.trim().length < minLength) {
      setState({ data: null, loading: false, error: null });
      return;
    }
    let cancelled = false;
    setState((s) => ({ ...s, loading: true, error: null }));
    api
      .searchLiterature(debounced)
      .then((data) => {
        if (!cancelled) setState({ data, loading: false, error: null });
      })
      .catch((error: ApiError) => {
        if (!cancelled) setState({ data: null, loading: false, error });
      });
    return () => {
      cancelled = true;
    };
  }, [debounced, minLength]);

  return state;
}

/**
 * FE-027 ROOT FIX: Fetch the knowledge graph subgraph for a drug or disease
 * via the real /api/knowledge-graph endpoint.
 *
 * Previously: the hook short-circuited when no drug/disease was provided,
 * so the KG explorer showed a blank graph on initial load. It also only
 * handled the `{ nodes, edges }` subgraph response shape — but the stats
 * endpoint (no params) returns `{ sources, nodeCount, edgeCount, ... }`.
 *
 * Fix: The hook ALWAYS fires. When drug/disease are provided, it calls
 * `GET /api/knowledge-graph?drug=X&disease=Y` which returns `{ nodes, edges }`
 * (subgraph). When no params are provided, it calls `GET /api/knowledge-graph`
 * which returns stats — we normalize stats to `{ nodes: [], edges: [] }` so
 * the KG explorer doesn't crash. The stats can be fetched separately via
 * `api.getKnowledgeGraphStats()` if needed.
 */
export function useKnowledgeGraph(params: { drug?: string; disease?: string }) {
  const [state, setState] = useState<
    AsyncState<{ nodes: any[]; edges: any[] }>
  >({ data: null, loading: false, error: null });

  const paramsKey = JSON.stringify(params);
  useEffect(() => {
    let cancelled = false;
    setState({ data: null, loading: true, error: null });
    const qs = new URLSearchParams();
    if (params.drug) qs.set("drug", params.drug);
    if (params.disease) qs.set("disease", params.disease);
    qs.set("limit", "100");

    // Always use drug/disease params to trigger the explore endpoint
    // which returns {nodes, edges}. Without params, the route returns stats
    // which has no nodes/edges — the graph would be blank.
    const hasParams = !!(params.drug || params.disease);
    if (!hasParams) {
      // No params: fetch stats for display purposes only (no graph data)
      fetch(`/api/knowledge-graph`, { credentials: "include" })
        .then(async (res) => {
          const text = await res.text();
          let body: any = null;
          if (text) { try { body = JSON.parse(text); } catch { body = { raw: text }; } }
          if (!res.ok) throw { error: body?.error || "request_failed", message: body?.message || `Request failed with status ${res.status}`, status: res.status } as ApiError;
          // Stats response — normalize to empty graph
          return { nodes: [], edges: [], _stats: body } as { nodes: any[]; edges: any[] };
        })
        .then((data) => { if (!cancelled) setState({ data, loading: false, error: null }); })
        .catch((error: ApiError) => { if (!cancelled) setState({ data: null, loading: false, error }); });
      return () => { cancelled = true; };
    }

    fetch(`/api/knowledge-graph?${qs.toString()}`, { credentials: "include" })
      .then(async (res) => {
        const text = await res.text();
        let body: any = null;
        if (text) {
          try { body = JSON.parse(text); } catch { body = { raw: text }; }
        }
        if (!res.ok) {
          throw {
            error: body?.error || "request_failed",
            message: body?.message || `Request failed with status ${res.status}`,
            status: res.status,
          } as ApiError;
        }
        // Normalize response: /kg/explore returns {nodes, edges, backend}
        // Ensure we always return the correct shape
        if (body && 'nodes' in body) {
          return body as { nodes: any[]; edges: any[] };
        }
        // Stats response (shouldn't happen with params, but guard anyway)
        if (body && 'sources' in body && !('nodes' in body)) {
          return { nodes: [], edges: [], _stats: body } as { nodes: any[]; edges: any[] };
        }
        return body as { nodes: any[]; edges: any[] };
      })
      .then((data) => {
        if (!cancelled) setState({ data, loading: false, error: null });
      })
      .catch((error: ApiError) => {
        if (!cancelled) setState({ data: null, loading: false, error });
      });
    return () => {
      cancelled = true;
    };
  }, [paramsKey]);

  return state;
}

/**
 * Build an evidence package via the real /api/evidence-package endpoint.
 * This is a mutation hook (not a query hook) — it returns a `build` function
 * plus the current state.
 */
export function useBuildEvidencePackage() {
  const [state, setState] = useState<
    AsyncState<Awaited<ReturnType<typeof api.buildEvidencePackage>>>
  >({ data: null, loading: false, error: null });

  const build = useCallback(
    (body: { drug: string; disease: string; notes?: string; literatureLimit?: number; trialsLimit?: number }) => {
      setState({ data: null, loading: true, error: null });
      return api
        .buildEvidencePackage(body)
        .then((data) => {
          setState({ data, loading: false, error: null });
          return data;
        })
        .catch((error: ApiError) => {
          setState({ data: null, loading: false, error });
          throw error;
        });
    },
    []
  );

  return { ...state, build };
}

/**
 * Fetch RL-ranked candidates via the real /api/rl endpoint.
 *
 * FE-067 ROOT FIX: When `drug` and `disease` are both unset, the hook now
 * issues a GET /api/rl (default top-N list) instead of short-circuiting
 * with no fetch. This lets the Knowledge Graph Explorer look up real RL
 * candidates by drug name when the user clicks a drug node — previously
 * the lookup hit the mock `drugCandidates` array and silently failed for
 * any drug that wasn't in the mock set.
 */
export function useRlCandidates(params: { drug?: string; disease?: string; limit?: number }) {
  const [state, setState] = useState<AsyncState<{ candidates: any[]; source?: string; total?: number }>>({
    data: null,
    loading: false,
    error: null,
  });

  const paramsKey = JSON.stringify(params);
  useEffect(() => {
    let cancelled = false;
    setState({ data: null, loading: true, error: null });

    // FE-067: if no drug/disease filter is provided, fetch the default
    // top-N list via GET. Otherwise POST with the filter params.
    const hasFilter = !!(params.drug || params.disease);
    
    // Grab the CSRF token from document.cookie, just like apiFetch does
    let csrfToken = "";
    if (typeof document !== "undefined") {
      const match = document.cookie.match(new RegExp("(^| )drugos_csrf=([^;]+)"));
      if (match) csrfToken = match[2];
    }
    
    const fetchInit: RequestInit = hasFilter
      ? {
          method: "POST",
          credentials: "include",
          headers: { 
            "Content-Type": "application/json",
            ...(csrfToken ? { "x-csrf-token": csrfToken } : {})
          },
          body: JSON.stringify(params),
        }
      : {
          method: "GET",
          credentials: "include",
          headers: { "Content-Type": "application/json" },
        };

    fetch(`/api/rl`, fetchInit)
      .then(async (res) => {
        const text = await res.text();
        let body: any = null;
        if (text) {
          try { body = JSON.parse(text); } catch { body = { raw: text }; }
        }
        if (!res.ok) {
          throw {
            error: body?.error || "request_failed",
            message: body?.message || `Request failed with status ${res.status}`,
            status: res.status,
          } as ApiError;
        }
        return body as { candidates: any[]; source?: string; total?: number };
      })
      .then((data) => {
        if (!cancelled) setState({ data, loading: false, error: null });
      })
      .catch((error: ApiError) => {
        if (!cancelled) setState({ data: null, loading: false, error });
      });
    return () => {
      cancelled = true;
    };
  }, [paramsKey]);

  return state;
}

/**
 * FE-029 ROOT FIX: Real notifications hook for the app shell.
 *
 * The previous app-shell.tsx imported a hardcoded `notifications` array from
 * `mock-data.ts` (now deleted). The array was empty, but the NAME invited
 * future engineers to add fabricated "Dr. Sarah Chen published a hypothesis"
 * entries — which would have been presented to researchers as real activity.
 *
 * This hook calls the real /api/notifications endpoint, which returns the
 * authenticated user's actual notification feed from the database. The hook
 * polls every 60 seconds so the bell-icon badge stays fresh without requiring
 * a manual refresh. On error or empty, it returns an empty array — the UI
 * renders an honest "No notifications" state, never fabricated data.
 *
 * Returns:
 *   - notifications: Notification[]  — the user's real notifications (newest first)
 *   - unreadCount: number           — count of notifications with readAt === null
 *   - loading: boolean              — true during the initial fetch
 *   - error: ApiError | null        — surfaced honestly, never swallowed
 *   - refetch: () => void           — trigger a manual refresh
 */
export function useNotifications(options: { pollMs?: number } = {}) {
  const [state, setState] = useState<{
    notifications: import("@/lib/api-client").Notification[];
    unreadCount: number;
    loading: boolean;
    error: ApiError | null;
  }>({
    notifications: [],
    unreadCount: 0,
    loading: true,
    error: null,
  });
  const [refetchCounter, setRefetchCounter] = useState(0);
  const refetch = () => setRefetchCounter((c) => c + 1);

  // Initial fetch + refetch when refetchCounter changes.
  useEffect(() => {
    let cancelled = false;
    setState((s) => ({ ...s, loading: true, error: null }));
    api
      .listNotifications()
      .then((res) => {
        if (cancelled) return;
        const items = res?.items ?? [];
        const unread = items.filter((n) => n.readAt === null).length;
        setState({ notifications: items, unreadCount: unread, loading: false, error: null });
      })
      .catch((error: ApiError) => {
        if (cancelled) return;
        // On error, render an empty feed — NEVER fabricated notifications.
        setState({ notifications: [], unreadCount: 0, loading: false, error });
      });
    return () => {
      cancelled = true;
    };
  }, [refetchCounter]);

  // Optional polling. Default 60s. Caller can disable by passing { pollMs: 0 }.
  const pollMs = options.pollMs ?? 60_000;
  useEffect(() => {
    if (!pollMs || pollMs <= 0) return;
    const id = setInterval(() => setRefetchCounter((c) => c + 1), pollMs);
    return () => clearInterval(id);
  }, [pollMs]);

  return { ...state, refetch };
}

/**
 * FE-030 ROOT FIX: Real team-activity feed hook for dashboard "Recent Activity".
 *
 * The previous all-screens.tsx / remaining-screens.tsx rendered hardcoded
 * arrays of fake colleagues ("Dr. Sarah Chen", "James Wilson", "Dr. Priya
 * Patel", "Dr. Lisa Kim") in the "Shared Queries", "Team Comments", and
 * "Recent Feedback" sections. A researcher believed these were real colleagues
 * and could not tell the dashboard was empty.
 *
 * This hook calls /api/team to fetch the REAL organization members. It does
 * NOT fabricate colleagues. On error or empty, it returns an empty array —
 * the UI renders an honest "No team members yet" state.
 */
export function useTeamMembers() {
  const [state, setState] = useState<{
    members: import("@/lib/api-client").TeamMember[];
    loading: boolean;
    error: ApiError | null;
  }>({
    members: [],
    loading: true,
    error: null,
  });
  const [refetchCounter, setRefetchCounter] = useState(0);
  const refetch = () => setRefetchCounter((c) => c + 1);

  useEffect(() => {
    let cancelled = false;
    setState((s) => ({ ...s, loading: true, error: null }));
    api
      .listTeamMembers()
      .then((res) => {
        if (cancelled) return;
        setState({ members: res?.items ?? [], loading: false, error: null });
      })
      .catch((error: ApiError) => {
        if (cancelled) return;
        setState({ members: [], loading: false, error });
      });
    return () => {
      cancelled = true;
    };
  }, [refetchCounter]);

  return { ...state, refetch };
}

/**
 * FE-024 ROOT FIX: Batch-fetch real drug mechanisms via the
 * /api/drugs/mechanism endpoint (backed by the ChEMBL service in
 * lib/services/drug-mechanism.ts). The UI uses this hook to display
 * the actual mechanism of action (e.g. "NMDA receptor antagonist")
 * instead of fake RL debug values like "RL reward: 0.234".
 *
 * Returns a Map keyed by lowercase drug name. Lookup is O(1).
 * The hook re-fetches whenever the set of drug names changes.
 */
export function useDrugMechanisms(drugNames: string[]) {
  const [state, setState] = useState<
    AsyncState<Map<string, import("@/lib/services/drug-mechanism").DrugMechanismResult>>
  >({ data: null, loading: false, error: null });

  // Sort + dedupe + join to make a stable dep key.
  const key = Array.from(new Set(drugNames.map((n) => (n || "").trim()).filter(Boolean)))
    .sort()
    .join("|");

  useEffect(() => {
    if (!key) {
      setState({ data: null, loading: false, error: null });
      return;
    }
    let cancelled = false;
    setState((s) => ({ ...s, loading: true, error: null }));
    const names = key.split("|");
    fetch(`/api/drugs/mechanism`, {
      method: "POST",
      credentials: "include",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ drugNames: names }),
    })
      .then(async (res) => {
        const text = await res.text();
        let body: any = null;
        if (text) {
          try { body = JSON.parse(text); } catch { body = { raw: text }; }
        }
        if (!res.ok) {
          throw {
            error: body?.error || "request_failed",
            message: body?.message || `Request failed with status ${res.status}`,
            status: res.status,
          } as ApiError;
        }
        const map = new Map<string, import("@/lib/services/drug-mechanism").DrugMechanismResult>();
        for (const r of body?.results || []) {
          if (r?.drugName) map.set(r.drugName.toLowerCase(), r);
        }
        return map;
      })
      .then((data) => {
        if (!cancelled) setState({ data, loading: false, error: null });
      })
      .catch((error: ApiError) => {
        if (!cancelled) setState({ data: null, loading: false, error });
      });
    return () => {
      cancelled = true;
    };
  }, [key]);

  return state;
}

/**
 * Reusable loading spinner component.
 */
export function LoadingSpinner({ label = "Loading..." }: { label?: string }) {
  return (
    <div className="flex items-center justify-center py-12 text-muted-foreground">
      <RefreshCw className="h-5 w-5 animate-spin mr-2" />
      <span className="text-sm">{label}</span>
    </div>
  );
}

/**
 * Reusable error display component.
 */
export function ErrorDisplay({ error, onRetry }: { error: ApiError; onRetry?: () => void }) {
  return (
    <div className="flex flex-col items-center justify-center py-12 text-center">
      <AlertCircle className="h-8 w-8 text-red-500 mb-2" />
      <p className="text-sm font-medium text-foreground">{error.message || error.error}</p>
      {error.status === 503 && (
        <p className="text-xs text-muted-foreground mt-1 max-w-md">
          The backend service is not deployed. Set the relevant environment variable
          (KG_SERVICE_URL, DATASET_SERVICE_URL, RL_SERVICE_URL) to enable this feature.
        </p>
      )}
      {onRetry && (
        <button
          onClick={onRetry}
          className="mt-4 text-xs text-primary hover:underline"
        >
          Try again
        </button>
      )}
    </div>
  );
}

// We import these here so the icons used by LoadingSpinner/ErrorDisplay are
// always available without each screen importing them separately.
import { RefreshCw, AlertCircle } from 'lucide-react';

// ---------------------------------------------------------------------------
// FE-009 ROOT FIX: Generic hooks for admin/dashboard screens.
//
// The previous all-screens.tsx and admin-billing-etc-screens.tsx rendered
// hardcoded mock data ("Dr. Sarah Chen", "James Wilson", etc.) for ~38
// admin/dashboard screens. An admin viewing "User Management" thought they
// saw the real user list but actually saw 6 fake users. The platform was
// non-functional for its stated admin/billing/collab use cases.
//
// The hooks below let every admin screen call the real API client with
// proper loading / error / empty states, so a researcher never sees
// fabricated data presented as real.
// ---------------------------------------------------------------------------

/**
 * Generic "fetch a list endpoint" hook. Returns { data, loading, error }
 * plus a `refetch` callback. The fetch is fired on mount and whenever
 * `refetchToken` changes (so callers can trigger a refresh).
 *
 * Usage:
 *   const { data, loading, error, refetch } = useApiList(
 *     () => api.listUsers(50, 0),
 *     []
 *   );
 */
export function useApiList<T>(
  fetcher: () => Promise<T>,
  deps: unknown[] = [],
  options: { refetchToken?: unknown } = {}
): {
  data: T | null;
  loading: boolean;
  error: ApiError | null;
  refetch: () => void;
} {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<ApiError | null>(null);
  const [refetchCounter, setRefetchCounter] = useState(0);

  // We deliberately stringify deps to avoid identity churn. The linter
  // can't statically verify that `fetcher` is stable, so we ignore it.
  const depsKey = JSON.stringify(deps);
  const refetchToken = options.refetchToken;

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    fetcher()
      .then((result) => {
        if (!cancelled) {
          setData(result);
          setLoading(false);
        }
      })
      .catch((err: ApiError) => {
        if (!cancelled) {
          setError(err);
          setLoading(false);
        }
      });
    return () => {
      cancelled = true;
    };
     
  }, [depsKey, refetchToken, refetchCounter]);

  const refetch = () => setRefetchCounter((c) => c + 1);
  return { data, loading, error, refetch };
}

/**
 * Generic "fetch a single resource" hook. Same semantics as useApiList but
 * for endpoints that return a single object (e.g. api.getSystemStatus()).
 */
export function useApiResource<T>(
  fetcher: () => Promise<T>,
  deps: unknown[] = []
): {
  data: T | null;
  loading: boolean;
  error: ApiError | null;
  refetch: () => void;
} {
  return useApiList(fetcher, deps);
}

/**
 * FE-009 ROOT FIX: DemoDataBanner.
 *
 * For admin/dashboard screens where the backend API has NOT been implemented
 * yet (RolesScreen, SSOScreen, FeatureFlagsScreen, etc.), we render this
 * banner above the illustrative data. It tells the admin honestly:
 *   "This screen shows illustrative demo data. The backend API is not yet
 *    implemented. Do not make business decisions based on these numbers."
 *
 * This is the production-grade way to handle "we don't have a real API for
 * this screen yet" — we DO NOT silently render mock data as if it were
 * real (that was the original FE-009 bug).
 */
export function DemoDataBanner({ screenName }: { screenName: string }) {
  return (
    <div className="mb-4 rounded-md border border-amber-300 bg-amber-50 px-4 py-3 text-amber-900">
      <div className="flex items-start gap-2">
        <AlertCircle className="h-5 w-5 flex-shrink-0 mt-0.5" />
        <div className="text-sm">
          <p className="font-semibold">Illustrative demo data — {screenName}</p>
          <p className="mt-1 text-xs">
            The backend API for this screen has not been implemented yet. The
            numbers and entries below are illustrative only and may not match
            your actual organization&rsquo;s state. <strong>Do not make business,
            billing, or compliance decisions based on this view.</strong>
          </p>
        </div>
      </div>
    </div>
  );
}

/**
 * FE-009 ROOT FIX: Generic empty state for list-based admin screens.
 * Used when a real API returns an empty list.
 */
export function EmptyState({
  title,
  description,
  action,
}: {
  title: string;
  description?: string;
  action?: ReactNode;
}) {
  return (
    <Card>
      <CardContent className="py-12 text-center text-muted-foreground">
        <AlertCircle className="h-8 w-8 mx-auto mb-2 opacity-50" />
        <p className="text-sm font-medium">{title}</p>
        {description && (
          <p className="text-xs mt-1 max-w-md mx-auto">{description}</p>
        )}
        {action && <div className="mt-4">{action}</div>}
      </CardContent>
    </Card>
  );
}
