import { NextResponse } from "next/server";
// FE-009 ROOT FIX (Team Member 13): use requireAuthRole('billing') instead
// of requireAuth().
//
// Previously this route used `requireAuth()` — ANY logged-in user could
// read the org's full invoice history, including read-only viewers. This
// violated the RBAC policy in lib/rbac.ts which defines a 'billing' role
// for invoice access. The invoices section in the sidebar is restricted
// to ["owner", "admin", "billing"] — the API route was the only path
// that didn't enforce the same restriction.
//
// ROOT FIX: replace requireAuth() with requireAuthRole("billing").
// requireAuthRole implicitly allows "admin" and "owner" (superuser
// bypass — see lib/api-helpers.ts:requireRole). Non-billing users
// (viewer, researcher, data-scientist, pi, business-dev, developer)
// now get 403 forbidden, matching the sidebar's RBAC gate.
//
// FINANCIAL COMPLIANCE: invoice data is regulated by GDPR (Article 30
// records of processing activities), SOC 2 (CC6.1 logical access
// controls), and depending on jurisdiction, financial-services laws
// (e.g., NYDFS 23 NYCRR 500 for NY-regulated insurers). A read-only
// viewer scraping the org's invoice history is a compliance violation.
import { requireAuthRole, badRequest } from "@/lib/api-helpers";
import { listOrganizationInvoices } from "@/lib/services/billing";

export async function GET() {
  // FE-009: only billing / admin / owner can read invoices.
  const auth = await requireAuthRole("billing");
  if (auth.user === null) return auth.response;
  if (!auth.user.orgId) return badRequest("No active organization");
  const invoices = await listOrganizationInvoices(auth.user.orgId);
  return NextResponse.json({ items: invoices });
}
