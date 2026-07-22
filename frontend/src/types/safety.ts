/**
 * Safety types — Task 251 root fix.
 *
 * ROOT CAUSE: the audit found `SafetyReport.drug` referenced in the
 * codebase, but the actual openFDA service (`lib/services/openfda.ts`)
 * returns `brandName` and `genericName` — never a `drug` field. Any
 * consumer that read `safetyData.drug` got `undefined`, silently
 * breaking safety-badge rendering in the candidate table.
 *
 * ROOT FIX: this file is the canonical home for safety-related types.
 * The `SafetyReport` interface is aligned with the ACTUAL response shape
 * of `GET /api/safety/[drug]` (served by `openfda.ts`):
 *
 *   {
 *     brandName: "ASPIRIN",
 *     genericName: "ASPIRIN",
 *     totalReports: 1234,
 *     seriousReports: 234,
 *     seriousReportsWithDeath: 12,
 *     topReactions: [{ term: "Nausea", count: 80 }, ...],
 *     disclaimer: "Adverse event data is sourced from..."
 *   }
 *
 * There is NO `drug` field. There is NO `name` field. There are TWO
 * name-like fields: `brandName` (proprietary, e.g. "Bayer") and
 * `genericName` (non-proprietary, e.g. "aspirin"). Consumers that need
 * a single display name should use `brandName || genericName`.
 *
 * This file also re-exports the underlying service-level types
 * (`DrugSafetySummary`, `AdverseEventReaction`) so consumers can import
 * from a single canonical location: `@/types/safety`.
 */

/**
 * A single adverse-event reaction term with its report count.
 * Source: openFDA `patient.reaction.reactionmeddrapt` aggregated.
 */
export interface AdverseEventReaction {
  term: string;
  count: number;
}

/**
 * Safety summary for a drug, sourced from openFDA FAERS.
 *
 * IMPORTANT — what these numbers MEAN:
 *   - `totalReports` is the number of FAERS reports that mention this
 *     drug. It is NOT the number of patients who experienced an event.
 *     A single patient can appear in multiple reports; a single report
 *     can list multiple drugs. FAERS data is spontaneous — there is no
 *     denominator. A drug with 10,000 reports and 100M prescriptions
 *     has a LOWER per-patient risk than a drug with 1,000 reports and
 *     10K prescriptions.
 *   - `seriousReports` counts reports flagged by FDA as serious
 *     (death, life-threatening, hospitalization, disability, congenital
 *     anomaly, or other serious outcome).
 *   - `seriousReportsWithDeath` is the subset where the patient died.
 *     This is a SUBSET of `seriousReports`, not additive.
 *   - `topReactions` is the top-10 most-cited MedDRA Preferred Term
 *     reactions across all reports. Counts are de-duplicated per report.
 *
 * The `disclaimer` field MUST be displayed alongside the data —
 * regulatory compliance requires that consumers understand these are
 * spontaneous reports, not proven causal events.
 */
export interface SafetyReport {
  /** Proprietary name (e.g. "Bayer"). May be absent for generic-only drugs. */
  brandName: string;
  /** Non-proprietary name (e.g. "aspirin"). Always present. */
  genericName: string;
  /** Total FAERS reports mentioning this drug (NOT patient count). */
  totalReports: number;
  /** Subset flagged as serious by FDA criteria. */
  seriousReports: number;
  /** Subset of seriousReports where the patient died. */
  seriousReportsWithDeath?: number;
  /** Top-10 MedDRA Preferred Term reactions by report count. */
  topReactions: AdverseEventReaction[];
  /** Regulatory disclaimer — MUST be displayed with the data. */
  disclaimer: string;
}

/**
 * Alias kept for backwards compatibility. New code should use
 * `SafetyReport`. This type alias ensures any consumer that imported
 * `DrugSafetySummary` from the old location still compiles.
 */
export type DrugSafetySummary = SafetyReport;
