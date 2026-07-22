# External APIs — Rate Limits & Integration Reference

> **Task 259 root fix.** This document is the single source of truth for
> every external API the DruGOS frontend proxies to. Operators use it to
> provision API keys, set quotas, and debug rate-limit 429s. The
> `frontend/src/lib/external-api-monitor.ts` module logs every call to
> these APIs with its duration and status — correlate this doc with the
> `external_api_call` log stream when investigating upstream issues.

## Summary table

| Provider            | Base URL                                          | Auth                          | Free quota                                | Env var                  | Frontend service file          |
|---------------------|---------------------------------------------------|-------------------------------|-------------------------------------------|--------------------------|--------------------------------|
| RxNorm (NLM)        | https://rxnav.nlm.nih.gov/REST                    | None                          | No hard limit; be nice (~1 req/sec)       | —                        | `lib/services/rxnorm.ts`       |
| MeSH (NLM)          | https://id.nlm.nih.gov/mesh/lookup                | None                          | No hard limit; be nice (~1 req/sec)       | —                        | `lib/services/mesh.ts`         |
| ChEMBL (EBI)        | https://www.ebi.ac.uk/chembl/api/data             | None                          | ~5 req/sec shared                         | —                        | `lib/services/drug-mechanism.ts` |
| openFDA (FAERS)     | https://api.fda.gov                                | API key optional              | 240 req/min shared; 120,000/min with key  | `OPENFDA_API_KEY`        | `lib/services/openfda.ts`      |
| ClinicalTrials.gov  | https://clinicaltrials.gov/api/v2                 | None                          | No published limit; burst-friendly        | —                        | `lib/services/clinical-trials.ts` |
| USPTO PatentsView   | https://search.patentsview.org/api/v1/patent      | API key REQUIRED              | Per-key (request at /apis/keyrequest)     | `PATENTSVIEW_API_KEY`    | `lib/services/patentsview.ts`  |
| Phase 2 KG service  | ${KG_SERVICE_URL}/kg/stats, /query, /cypher       | Internal (CORS whitelist)     | Bounded by Neo4j; 30s timeout per query   | `KG_SERVICE_URL`         | `lib/services/knowledge-graph-stats.ts` |

## Per-user rate limit (frontend enforcement)

All 7 public-API-proxy routes (`/api/drugs/search`, `/api/drugs/mechanism`,
`/api/diseases/search`, `/api/safety/[drug]`, `/api/clinical-trials/search`,
`/api/patents/search`, and the literature route) enforce a **5 req/sec per
user** sliding-window rate limit at the Next.js layer (see
`lib/auth/rate-limit.ts` -> `checkUserApiRateLimitV2`). This is IN ADDITION
to the upstream provider's quota — even if a single user exhausts their
5 req/sec allowance, the upstream API never sees more than 5 req/sec
from our backend.

When the per-user limit is hit, the route returns:

```json
{
  "error": "rate_limited",
  "message": "Too many requests. Try again in N second(s).",
  "retryAfterSeconds": 1
}
```

with HTTP status 429 and a `Retry-After: 1` header.

## Provider-specific notes

### RxNorm (drug name normalization)
- **No API key needed.** NLM asks that you "be reasonable" — keep under
  ~1 req/sec average. Our backend caches responses for 24h
  (`next: { revalidate: 86400 }`) so repeat searches for the same drug
  hit the cache, not NLM.
- **Timeout:** 3 seconds (AbortController). On timeout the route returns
  503 with a "RxNorm lookup timed out — please retry" message.

### MeSH (disease descriptors)
- **No API key needed.** Same "be reasonable" guideline as RxNorm.
- **Caching:** 30 days (`next: { revalidate: 86400 * 30 }`) because MeSH
  updates ~weekly.
- **N+1 calls:** a single disease search triggers 1 descriptor-list
  call + up to N descriptor-detail calls (capped at `limit`). Operators
  should expect ~5 upstream calls per user search.

### ChEMBL (drug mechanism of action)
- **No API key needed.** Shared rate limit ~5 req/sec across all callers
  on EBI's infrastructure. Our backend enforces client-side concurrency
  of 5 in `lookupDrugMechanisms()` to avoid 429s.
- **Caching:** 1-hour in-memory TTL (see `drug-mechanism.ts`).
  Operators can force-refresh via `POST /api/drugs/mechanism/refresh`.

### openFDA (FAERS adverse events)
- **API key OPTIONAL but strongly recommended.** Without a key, our
  backend shares a 240 req/min pool with every other unauthenticated
  caller on the internet — demos will be slow. Register a free key at
  https://open.fda.gov/api/reference/ and set `OPENFDA_API_KEY` to
  raise the limit to 120,000 req/min per key.
- **404 is normal:** openFDA returns 404 when zero reports match the
  drug. Our service treats 404 as "no data" and returns a zero-report
  summary (NOT an error).
- **Scientific caveat:** FAERS reports are spontaneous. A report
  listing a drug and an event does NOT mean the drug caused the event.
  The `disclaimer` field in every response MUST be displayed alongside
  the data.

### ClinicalTrials.gov v2
- **No API key needed.** CT.gov v2 is cursor-paginated — pass the
  `nextPageToken` from one response as `pageToken` on the next request.
  Numeric offsets are NOT supported; the route rejects `page=2` style
  pagination.
- **Caching:** 1 hour (`next: { revalidate: 3600 }`).

### USPTO PatentsView
- **API key REQUIRED.** Without `PATENTSVIEW_API_KEY`, the service
  returns an empty result set with a `reason` field explaining the
  missing key. Request a free key at
  https://patentsview.org/apis/keyrequest.
- **Pagination:** 100 patents per page. The service auto-paginates up
  to 1,000 patents per `searchPatents()` call (safety cap). Callers
  needing more should make multiple calls with different query terms.

### Phase 2 KG service
- **Internal service.** Set `KG_SERVICE_URL` (e.g.
  `http://localhost:8002`) to enable proxying KG queries to the Python
  Phase 2 service. Without it, the frontend reads the local Phase 2
  registry at `../phase2/data/registry.json` for stats, but cannot
  answer structured drug/disease queries (returns 503).
- **CORS:** the Phase 2 service whitelists origins via
  `KG_CORS_ORIGINS` (default: `http://localhost:3000`).
- **Timeout:** 30s per Cypher query. The frontend aborts and returns
  504 if the upstream does not respond in time.

## Monitoring

Every external API call is logged by
`lib/external-api-monitor.ts:monitoredFetch()`. Each log line is JSON:

```json
{
  "event": "external_api_call",
  "provider": "openfda",
  "url": "https://api.fda.gov/drug/event.json?search=...",
  "method": "GET",
  "status": 200,
  "durationMs": 412,
  "ok": true,
  "timestamp": "2026-07-16T01:55:00.000Z"
}
```

Slow (>3s) or failed calls are also logged at WARN level with event
`external_api_call_slow_or_failed` so operators can alert on them
without grepping the INFO stream.

To inspect recent calls programmatically (e.g. from an admin endpoint):

```ts
import { __getRecentCalls } from "@/lib/external-api-monitor";
const recent = __getRecentCalls(100);
```
