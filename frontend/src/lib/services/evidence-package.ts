/**
 * Evidence Package service.
 *
 * Assembles a scientifically-grounded evidence package for a (drug, disease)
 * hypothesis by aggregating REAL data from:
 *   - PubMed literature search (NCBI E-utilities)
 *   - ClinicalTrials.gov interventional studies
 *   - FDA adverse event reports (openFDA)
 *
 * We never invent data. If a service is unavailable or returns no results,
 * we say so explicitly. The package is serializable to JSON and to a
 * human-readable PDF-ready markdown blob.
 */

import { searchPubMed, type PubMedArticle } from "./pubmed";
import { searchClinicalTrials, type ClinicalTrial } from "./clinical-trials";
import { getDrugSafetySummary, type DrugSafetySummary } from "./openfda";

export interface EvidencePackage {
  drug: string;
  disease: string;
  generatedAt: string;
  literature: {
    total: number;
    articles: PubMedArticle[];
  };
  clinicalTrials: {
    total: number;
    trials: ClinicalTrial[];
  };
  safety: DrugSafetySummary | null;
  notes: string;
  /**
   * BE-018 ROOT FIX: per-service status so the caller can distinguish
   * "0 results because the service has no data" from "0 results because
   * the service was unreachable". A pharma partner making a go/no-go
   * decision MUST know whether "0 clinical trials" means "no trials are
   * registered" (real signal) or "CT.gov was down" (incomplete data).
   *
   * The UI should display a warning banner for any service marked "failed".
   * The PDF export should include a "Data Completeness" section listing
   * which sources succeeded and which failed.
   */
  serviceStatus: {
    literature: "ok" | "failed";
    clinicalTrials: "ok" | "failed";
    safety: "ok" | "failed";
  };
}

export interface BuildEvidencePackageInput {
  drug: string;
  disease: string;
  literatureLimit?: number;
  trialsLimit?: number;
  notes?: string;
}

export async function buildEvidencePackage(input: BuildEvidencePackageInput): Promise<EvidencePackage> {
  const drug = (input.drug || "").trim();
  const disease = (input.disease || "").trim();
  if (!drug || !disease) {
    throw new Error("Both drug and disease must be provided to build an evidence package.");
  }

  // BE-052 ROOT FIX (v115, LOW): PubMed query injection. The previous
  // code interpolated `${drug} AND ${disease}` directly into the
  // PubMed query string. PubMed's query syntax supports boolean
  // operators (AND, OR, NOT), field qualifiers ([Title], [MeSH]), and
  // parentheses. An attacker (or careless researcher) passing
  // drug="aspirin OR cancer" would craft the query
  // "aspirin OR cancer AND <disease>" — PubMed would interpret this
  // as (aspirin) OR (cancer AND <disease>), returning all aspirin-
  // related articles PLUS all cancer-and-disease articles. The
  // evidence package would contain irrelevant articles, inflating
  // the "literature support" metric and potentially making a weak
  // hypothesis look well-supported.
  //
  // ROOT FIX: validate drug and disease against the biomedical-name
  // whitelist before interpolation. The whitelist allows alphanumerics,
  // spaces, hyphens, and apostrophes (e.g. "St John's Wort") — these
  // are the only characters in legitimate drug/disease names. Any
  // input that doesn't match is rejected up-front with a clear error
  // message. This is the same whitelist used by openfda.ts and the
  // /api/safety / /api/drugs/search routes.
  const BIOMEDICAL_NAME_WHITELIST = /^[A-Za-z0-9 \-']{2,128}$/;
  if (!BIOMEDICAL_NAME_WHITELIST.test(drug)) {
    throw new Error(
      `Invalid drug name "${drug.slice(0, 64)}": only alphanumerics, spaces, hyphens, and apostrophes are allowed (BE-052 root fix: prevents PubMed query injection).`
    );
  }
  if (!BIOMEDICAL_NAME_WHITELIST.test(disease)) {
    throw new Error(
      `Invalid disease name "${disease.slice(0, 64)}": only alphanumerics, spaces, hyphens, and apostrophes are allowed (BE-052 root fix: prevents PubMed query injection).`
    );
  }

  // BE-051 ROOT FIX (v115, MEDIUM): per-call timeout.
  //
  // ROOT CAUSE: the previous code used Promise.allSettled without a
  // timeout. If PubMed was slow (10s) but CT.gov and openFDA returned
  // in 1s, the overall request took 10s. Under V1's 100-concurrent-
  // request load, 100 evidence-package builds each taking 10s = 1000s
  // of accumulated external API time → external API rate limits
  // exceeded (NCBI: 3 req/sec without key) → all subsequent requests
  // failed.
  //
  // ROOT FIX: wrap each external call in a per-call 5s timeout via
  // Promise.race with a timeout promise. If a call exceeds 5s, it
  // rejects with a timeout error — Promise.allSettled captures the
  // rejection as "failed" status, and the evidence package is built
  // with whatever data DID come back. The UI shows a "FAILED" badge
  // for the timed-out source so the researcher knows the data is
  // incomplete.
  //
  // We don't need a separate overall timeout because each per-call
  // timeout caps the worst case at 5s — Promise.allSettled waits for
  // all three to settle, which is at most 5s.
  const PER_CALL_TIMEOUT_MS = 5_000;

  // Helper: wrap a promise with a timeout. Returns the original
  // promise's result if it resolves before the timeout, else rejects
  // with a timeout error. The timeout timer is cleared on settlement
  // to avoid leaking a long-lived timer handle.
  function withTimeout<T>(p: Promise<T>, ms: number, label: string): Promise<T> {
    return new Promise<T>((resolve, reject) => {
      const timer = setTimeout(() => {
        reject(new Error(`${label} timed out after ${ms}ms`));
      }, ms);
      p.then(
        (val) => {
          clearTimeout(timer);
          resolve(val);
        },
        (err) => {
          clearTimeout(timer);
          reject(err);
        }
      );
    });
  }

  // Run all three lookups concurrently — they are independent.
  // Each is wrapped in a per-call 5s timeout. The overall
  // Promise.allSettled therefore completes in at most 5s (not the
  // sum of all three call times).
  const [literature, clinicalTrials, safety] = await Promise.allSettled([
    withTimeout(
      searchPubMed({
        query: `${drug} AND ${disease}`,
        limit: input.literatureLimit ?? 15,
        sort: "relevance",
      }),
      PER_CALL_TIMEOUT_MS,
      "PubMed search"
    ),
    withTimeout(
      searchClinicalTrials({
        condition: disease,
        intervention: drug,
        limit: input.trialsLimit ?? 10,
      }),
      PER_CALL_TIMEOUT_MS,
      "ClinicalTrials.gov search"
    ),
    withTimeout(
      getDrugSafetySummary(drug),
      PER_CALL_TIMEOUT_MS,
      "openFDA safety summary"
    ),
  ]);

  // BE-018 ROOT FIX: capture per-service status. A "rejected" promise means
  // the service was unreachable (network error, 5xx, parse failure). A
  // "fulfilled" promise with 0 results means the service is up but has no
  // data for this query. The two cases MUST be distinguishable in the
  // response — otherwise a pharma partner could make a go/no-go decision
  // on incomplete data, believing "0 trials" means "no trials exist"
  // when in fact CT.gov was down.
  const literatureStatus = literature.status === "fulfilled" ? "ok" : "failed" as const;
  const clinicalTrialsStatus = clinicalTrials.status === "fulfilled" ? "ok" : "failed" as const;
  const safetyStatus = safety.status === "fulfilled" ? "ok" : "failed" as const;

  // Log failures loudly so operators see upstream outages in the platform log.
  if (literatureStatus === "failed") {
    console.error(`[evidence-package] PubMed lookup failed for "${drug}+${disease}":`, (literature as PromiseRejectedResult).reason);
  }
  if (clinicalTrialsStatus === "failed") {
    console.error(`[evidence-package] ClinicalTrials.gov lookup failed for "${drug}+${disease}":`, (clinicalTrials as PromiseRejectedResult).reason);
  }
  if (safetyStatus === "failed") {
    console.error(`[evidence-package] openFDA lookup failed for "${drug}":`, (safety as PromiseRejectedResult).reason);
  }

  // Build a notes string that includes a data-completeness warning when any
  // service failed. The notes are persisted in the EvidencePackage DB row
  // and exported to the PDF, so the warning is visible to pharma partners.
  const failedServices: string[] = [];
  if (literatureStatus === "failed") failedServices.push("PubMed");
  if (clinicalTrialsStatus === "failed") failedServices.push("ClinicalTrials.gov");
  if (safetyStatus === "failed") failedServices.push("openFDA");
  const completenessWarning = failedServices.length > 0
    ? ` WARNING: ${failedServices.join(", ")} ${failedServices.length === 1 ? "was" : "were"} unreachable when this package was generated. ` +
      `The corresponding section may show 0 results due to the outage, not due to absence of data. ` +
      `Re-generate this package later when the service(s) recover.`
    : "";

  return {
    drug,
    disease,
    generatedAt: new Date().toISOString(),
    literature: {
      total: literature.status === "fulfilled" ? literature.value.total : 0,
      articles: literature.status === "fulfilled" ? literature.value.articles : [],
    },
    clinicalTrials: {
      total: clinicalTrials.status === "fulfilled" ? clinicalTrials.value.total : 0,
      trials: clinicalTrials.status === "fulfilled" ? clinicalTrials.value.trials : [],
    },
    safety: safety.status === "fulfilled" ? safety.value : null,
    serviceStatus: {
      literature: literatureStatus,
      clinicalTrials: clinicalTrialsStatus,
      safety: safetyStatus,
    },
    notes:
      input.notes ||
      `Evidence package assembled for ${drug} as a candidate for ${disease}. ` +
        "All data is sourced from authoritative public databases (PubMed, " +
        "ClinicalTrials.gov, openFDA). This package does NOT contain any " +
        "model predictions — those are owned by the standalone RL agent." +
        completenessWarning,
  };
}

/**
 * Convert the evidence package to a markdown document suitable for PDF export.
 */
export function evidencePackageToMarkdown(pkg: EvidencePackage): string {
  const lines: string[] = [];
  lines.push(`# Evidence Package`);
  lines.push(``);
  lines.push(`- **Drug**: ${pkg.drug}`);
  lines.push(`- **Disease**: ${pkg.disease}`);
  lines.push(`- **Generated**: ${pkg.generatedAt}`);
  lines.push(`- **Notes**: ${pkg.notes}`);
  lines.push(``);
  // BE-018: Data Completeness section — surfaces per-service status so a
  // pharma partner reading the PDF knows whether "0 trials" means "no
  // trials registered" or "CT.gov was down when this was generated".
  //
  // BE-070 ROOT FIX (v115, LOW): the previous code numbered this section
  // "## 0. Data Completeness" — but PDF renderers (and most markdown
  // TOC generators) start at 1, not 0. The "Section 0" header broke
  // the PDF's table of contents (the section appeared before the TOC
  // start, or was skipped entirely). The fix renumbers to a
  // non-numeric heading so it doesn't interfere with the numbered
  // sections that follow (PubMed = 1, Clinical Trials = 2, Safety = 3).
  lines.push(`## Data Completeness`);
  lines.push(``);
  const status = pkg.serviceStatus ?? { literature: "ok", clinicalTrials: "ok", safety: "ok" };
  const anyFailed = status.literature === "failed" || status.clinicalTrials === "failed" || status.safety === "failed";
  if (anyFailed) {
    lines.push(`> ⚠ **WARNING**: One or more data sources were unreachable when this package was generated.`);
    lines.push(`> Sections marked "failed" may show 0 results due to the outage, NOT due to absence of data.`);
    lines.push(`> Re-generate this package after the failed service(s) recover to get a complete picture.`);
    lines.push(``);
  }
  lines.push(`| Source | Status |`);
  lines.push(`|--------|--------|`);
  lines.push(`| PubMed Literature | ${status.literature === "ok" ? "✅ OK" : "❌ FAILED"} |`);
  lines.push(`| ClinicalTrials.gov | ${status.clinicalTrials === "ok" ? "✅ OK" : "❌ FAILED"} |`);
  lines.push(`| openFDA Safety | ${status.safety === "ok" ? "✅ OK" : "❌ FAILED"} |`);
  lines.push(``);
  lines.push(`---`);
  lines.push(``);
  lines.push(`## 1. PubMed Literature (${pkg.literature.total} total matches)`);
  lines.push(``);
  if (pkg.literature.articles.length === 0) {
    if (status.literature === "failed") {
      lines.push(`❌ PubMed lookup FAILED — the service was unreachable when this package was generated.`);
      lines.push(`"0 articles" reflects the outage, not the absence of literature. Re-generate later.`);
    } else {
      lines.push(`No articles returned. PubMed search returned zero results for this drug-disease pair.`);
    }
  } else {
    for (const a of pkg.literature.articles) {
      lines.push(`### ${a.title}`);
      lines.push(`- **PMID**: ${a.pmid}`);
      lines.push(`- **Journal**: ${a.journal}`);
      lines.push(`- **Authors**: ${a.authors.slice(0, 5).join(", ")}${a.authors.length > 5 ? ", et al." : ""}`);
      lines.push(`- **Date**: ${a.pubDate}`);
      if (a.doi) lines.push(`- **DOI**: ${a.doi}`);
      lines.push(`- **URL**: ${a.url}`);
      lines.push(``);
    }
  }
  lines.push(`---`);
  lines.push(``);
  lines.push(`## 2. Clinical Trials (${pkg.clinicalTrials.total} total matches)`);
  lines.push(``);
  if (pkg.clinicalTrials.trials.length === 0) {
    if (status.clinicalTrials === "failed") {
      lines.push(`❌ ClinicalTrials.gov lookup FAILED — the service was unreachable when this package was generated.`);
      lines.push(`"0 trials" reflects the outage, not the absence of registered trials. Re-generate later.`);
    } else {
      lines.push(`No registered clinical trials returned for this drug-disease pair.`);
    }
  } else {
    for (const t of pkg.clinicalTrials.trials) {
      lines.push(`### ${t.title}`);
      lines.push(`- **NCT ID**: ${t.nctId}`);
      lines.push(`- **Status**: ${t.status}`);
      lines.push(`- **Phase**: ${t.phase}`);
      lines.push(`- **Sponsor**: ${t.sponsor || "N/A"}`);
      lines.push(`- **Enrollment**: ${t.enrollment ?? "N/A"}`);
      lines.push(`- **Start**: ${t.startDate || "N/A"}`);
      lines.push(`- **Completion**: ${t.completionDate || "N/A"}`);
      lines.push(`- **URL**: ${t.url}`);
      lines.push(``);
    }
  }
  lines.push(`---`);
  lines.push(``);
  lines.push(`## 3. FDA Adverse Event Profile`);
  lines.push(``);
  if (!pkg.safety) {
    if (status.safety === "failed") {
      lines.push(`❌ openFDA lookup FAILED — the service was unreachable when this package was generated.`);
      lines.push(`"No safety data" reflects the outage, not a clean safety record. Re-generate later.`);
    } else {
      lines.push(`openFDA returned no safety data for this drug (no adverse event reports on file).`);
    }
  } else {
    lines.push(`- **Generic Name**: ${pkg.safety.genericName}`);
    lines.push(`- **Brand Name**: ${pkg.safety.brandName}`);
    lines.push(`- **Total Reports Returned**: ${pkg.safety.totalReports}`);
    lines.push(`- **Serious Reports**: ${pkg.safety.seriousReports}`);
    lines.push(`- **Serious Reports Involving Death**: ${pkg.safety.seriousReportsWithDeath}`);
    lines.push(``);
    lines.push(`### Top Reported Reactions`);
    if (pkg.safety.topReactions.length === 0) {
      lines.push(`No reaction frequency data available in the returned reports.`);
    } else {
      for (const r of pkg.safety.topReactions) {
        lines.push(`- ${r.term}: ${r.count} reports`);
      }
    }
    lines.push(``);
    lines.push(`> **Disclaimer**: ${pkg.safety.disclaimer}`);
  }
  return lines.join("\n");
}
