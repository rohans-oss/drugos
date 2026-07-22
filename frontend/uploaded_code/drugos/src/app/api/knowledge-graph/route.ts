import { NextRequest, NextResponse } from "next/server";
import { checkKnowledgeGraphAvailability } from "@/lib/services/ml-stubs";

/**
 * Knowledge Graph query endpoint.
 *
 * V100 ROOT FIX (BUG #14, P0 CRITICAL): the previous code returned
 * `501 not_implemented` UNCONDITIONALLY — even when `KG_SERVICE_URL`
 * was set. The Phase 2 Neo4j graph was unreachable from the dashboard.
 *
 * Root fix: when `KG_SERVICE_URL` is set, forward the request to the
 * Neo4j query service. We NEVER fabricate graph data — if the service
 * is not deployed, we return 503 with a clear message.
 */
export async function GET(req: NextRequest) {
  const availability = checkKnowledgeGraphAvailability();
  if (!availability.available) {
    return NextResponse.json(
      {
        error: "service_not_deployed",
        service: availability.service,
        description: availability.description,
        reason: availability.reason,
        documentation: "See Phase 2 of the build plan (Neo4j Knowledge Graph Construction).",
      },
      { status: 503 },
    );
  }
  // V100 BUG #14: proxy to the real KG service.
  const kgUrl = process.env.KG_SERVICE_URL!;
  const { search } = new URL(req.url);
  try {
    const upstream = await fetch(
      `${kgUrl.replace(/\/$/, "")}/graph${search || ""}`,
      { method: "GET", headers: { "Accept": "application/json" } },
    );
    const text = await upstream.text();
    return new NextResponse(text, {
      status: upstream.status,
      headers: { "Content-Type": upstream.headers.get("Content-Type") ?? "application/json" },
    });
  } catch (err) {
    return NextResponse.json(
      { error: "kg_service_unreachable", message: String(err) },
      { status: 502 },
    );
  }
}

export async function POST(req: NextRequest) {
  const availability = checkKnowledgeGraphAvailability();
  if (!availability.available) {
    return NextResponse.json(
      {
        error: "service_not_deployed",
        service: availability.service,
        description: availability.description,
        reason: availability.reason,
        documentation: "See Phase 2 of the build plan (Neo4j Knowledge Graph Construction).",
      },
      { status: 503 },
    );
  }
  const kgUrl = process.env.KG_SERVICE_URL!;
  const body = await req.text();
  try {
    const upstream = await fetch(
      `${kgUrl.replace(/\/$/, "")}/query`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json", "Accept": "application/json" },
        body,
      },
    );
    const text = await upstream.text();
    return new NextResponse(text, {
      status: upstream.status,
      headers: { "Content-Type": upstream.headers.get("Content-Type") ?? "application/json" },
    });
  } catch (err) {
    return NextResponse.json(
      { error: "kg_service_unreachable", message: String(err) },
      { status: 502 },
    );
  }
}
