import { NextRequest, NextResponse } from "next/server";
import { checkDatasetAvailability } from "@/lib/services/ml-stubs";

/**
 * Dataset statistics endpoint.
 *
 * V100 ROOT FIX (BUG #14, P0 CRITICAL): the previous code returned
 * `501 not_implemented` UNCONDITIONALLY — even when `DATASET_SERVICE_URL`
 * was set. The Phase 1 Airflow ETL pipeline was unreachable from the dashboard.
 *
 * Root fix: when `DATASET_SERVICE_URL` is set, forward the request to the
 * Airflow dataset service. We NEVER fabricate dataset statistics — if the
 * service is not deployed, we return 503 with a clear message.
 */
export async function GET(req: NextRequest) {
  const availability = checkDatasetAvailability();
  if (!availability.available) {
    return NextResponse.json(
      {
        error: "service_not_deployed",
        service: availability.service,
        description: availability.description,
        reason: availability.reason,
        documentation: "See Phase 1 of the build plan (Data Ingestion & Pipeline Setup).",
      },
      { status: 503 },
    );
  }
  // V100 BUG #14: proxy to the real dataset service.
  const dsUrl = process.env.DATASET_SERVICE_URL!;
  const { search } = new URL(req.url);
  try {
    const upstream = await fetch(
      `${dsUrl.replace(/\/$/, "")}/stats${search || ""}`,
      { method: "GET", headers: { "Accept": "application/json" } },
    );
    const text = await upstream.text();
    return new NextResponse(text, {
      status: upstream.status,
      headers: { "Content-Type": upstream.headers.get("Content-Type") ?? "application/json" },
    });
  } catch (err) {
    return NextResponse.json(
      { error: "dataset_service_unreachable", message: String(err) },
      { status: 502 },
    );
  }
}
