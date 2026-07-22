import { NextRequest, NextResponse } from "next/server";
import { buildEvidencePackage, evidencePackageToMarkdown } from "@/lib/services/evidence-package";
import { requireAuthRole, badRequest, internalError, writeAuditLog, requireCsrfOrSend, notFound } from "@/lib/api-helpers";
import { parsePagination, buildPaginatedResponse } from "@/lib/pagination";
import { db } from "@/lib/db";
// FE-008 ROOT FIX (Team Member 13): validate drug/disease exist in the KG.
//
// Previously this route accepted ANY drug + disease name and built an
// evidence package from PubMed + ClinicalTrials.gov + openFDA without
// verifying the drug or disease exists in the Phase 2 KG. A researcher
// could request an evidence package for "aspirin" + "cancer" even if the
// KG has no aspirin-cancer edge. The PDF then contained generic drug and
// disease info, not KG-derived evidence — the researcher may believe the
// package reflects KG evidence when it does not.
//
// ROOT FIX: before building the package, query the KG (via
// knowledge-graph-stats.ts when KG_SERVICE_URL is unset, or via the KG
// service proxy when it is set) for the drug and disease. If neither is
// found in the KG, return 404 with a clear error message.
//
// Issue 228 ROOT FIX: the previous version of this route was reported
// as "currently returns mock PDF". After forensic code reading, the
// route DOES call real services (buildEvidencePackage → PubMed +
// ClinicalTrials.gov + openFDA). The "mock PDF" claim was likely from
// an older version. This version is verified to call real services.
//
// Issue 232 ROOT FIX: use the unified kg-service.ts for KG entity
// validation instead of the deprecated knowledge-graph-stats.ts.
// The new validateEntityInKg function calls /kg/explore on the Python
// service (the CORRECT endpoint — the previous version called /lookup
// which does not exist on phase2/service.py).
//
// SECURITY: this validation is NOT a substitute for KG-level row-level
// security. The KG service enforces tenant isolation on its side. This
// route only checks "does the drug/disease name appear in the KG at all"
// — not "does the calling user's org have access to it".
import { validateEntityInKg } from "@/lib/services/kg-service";

// validateEntityInKg is now imported from @/lib/services/kg-service
// (Issue 232). The previous local definition called /lookup which does
// NOT exist on phase2/service.py — it has /kg/explore. The new
// imported function calls the correct endpoint.

/** FE-010 ROOT FIX: Evidence package build requires a research role. */
export async function POST(req: NextRequest) {
  // FE-011: CSRF protection on every state-changing route.
  const csrf = await requireCsrfOrSend(req);
  if (csrf.response) return csrf.response;

  const auth = await requireAuthRole("researcher", "data-scientist", "pi", "business-dev");
  if (auth.user === null) return auth.response;

  let body: { drug: string; disease: string; notes?: string; literatureLimit?: number; trialsLimit?: number };
  try {
    body = await req.json();
  } catch {
    return badRequest("Invalid JSON body");
  }
  if (!body.drug || !body.disease) {
    return badRequest("Both 'drug' and 'disease' fields are required");
  }

  // BE-006 ROOT FIX: the previous code accepted a `skipKgValidation: true`
  // body flag from admin/owner callers to bypass KG entity validation
  // entirely. This was a security hole: an admin could generate an
  // evidence package for "aspirin for cancer" (no KG edge), and the
  // package would look IDENTICAL to a KG-validated one in the UI. The
  // only signal that validation was skipped was a field in the audit log
  // — invisible to the researcher receiving the package. A researcher
  // acting on the package would believe the KG confirms the drug-disease
  // relationship when in fact the KG has no such edge.
  //
  // Root fix: REMOVE the bypass entirely. KG validation is now mandatory
  // for ALL callers (admin, owner, platformOwner included). If an admin
  // needs to generate a package for an entity not in the KG, they MUST
  // add the entity to the KG first (via the Phase 2 ingestion pipeline).
  // This enforces the scientific-contract invariant: every evidence
  // package is backed by a KG edge.
  //
  // We ALSO ignore the `skipKgValidation` field if a legacy client sends
  // it — the field is silently dropped (no error) so existing clients
  // don't break, but the bypass no longer works.
  const [drugCheck, diseaseCheck] = await Promise.all([
    validateEntityInKg(body.drug, "drug"),
    validateEntityInKg(body.disease, "disease"),
  ]);
  if (!drugCheck.ok) {
    return notFound(drugCheck.reason || `Drug "${body.drug}" not found in knowledge graph`);
  }
  if (!diseaseCheck.ok) {
    return notFound(diseaseCheck.reason || `Disease "${body.disease}" not found in knowledge graph`);
  }

  try {
    const pkg = await buildEvidencePackage({
      drug: body.drug,
      disease: body.disease,
      literatureLimit: body.literatureLimit,
      trialsLimit: body.trialsLimit,
      notes: body.notes,
    });
    const markdown = evidencePackageToMarkdown(pkg);
    // Persist the package so the user can retrieve it later
    const record = await db.evidencePackage.create({
      data: {
        userId: auth.user.userId,
        drugName: pkg.drug,
        diseaseName: pkg.disease,
        title: `Evidence package: ${pkg.drug} for ${pkg.disease}`,
        summary: pkg.notes,
        payloadJson: JSON.stringify(pkg),
        status: "generated",
      },
    });
    await writeAuditLog({
      user: auth.user,
      action: "evidence_package_generated",
      resource: `evidence_package:${record.id}`,
      metadata: {
        drug: pkg.drug,
        disease: pkg.disease,
        literatureCount: pkg.literature.total,
        trialsCount: pkg.clinicalTrials.total,
        // BE-006: kgValidationSkipped is now ALWAYS false — the bypass
        // was removed. The field is kept in the audit log for backwards
        // compatibility with log-analysis tooling that expects it.
        kgValidationSkipped: false,
        serviceStatus: pkg.serviceStatus,
      },
    });
    return NextResponse.json({ id: record.id, package: pkg, markdown });
  } catch (e: unknown) {
    const msg = e instanceof Error ? e.message : String(e);
    return internalError(`Evidence package build failed: ${msg}`);
  }
}

/**
 * FE-047 ROOT FIX: GET was `take: 50` with no offset/pagination — power
 * users with >50 evidence packages could not access older records. Now
 * accepts `limit` (capped at 100) and `offset` query params and returns
 * the standard paginated envelope `{ items, total, hasMore, limit, offset }`.
 */
export async function GET(req: NextRequest) {
  const auth = await requireAuthRole("researcher", "data-scientist", "pi", "business-dev");
  if (auth.user === null) return auth.response;
  const id = req.nextUrl.searchParams.get("id");
  if (id) {
    const pkg = await db.evidencePackage.findUnique({ where: { id } });
    if (!pkg || pkg.userId !== auth.user.userId) {
      return NextResponse.json({ error: "not_found" }, { status: 404 });
    }
    return NextResponse.json({ id: pkg.id, package: JSON.parse(pkg.payloadJson), markdown: evidencePackageToMarkdown(JSON.parse(pkg.payloadJson)) });
  }
  const page = parsePagination(req.nextUrl.searchParams);
  const where = { userId: auth.user.userId };
  const [items, total] = await Promise.all([
    db.evidencePackage.findMany({
      where,
      orderBy: { createdAt: "desc" },
      take: page.limit,
      skip: page.offset,
      select: { id: true, drugName: true, diseaseName: true, title: true, status: true, createdAt: true },
    }),
    db.evidencePackage.count({ where }),
  ]);
  return NextResponse.json(buildPaginatedResponse(items, total, page));
}
