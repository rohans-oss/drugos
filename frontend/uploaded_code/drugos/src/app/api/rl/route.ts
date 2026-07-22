import { NextRequest, NextResponse } from "next/server";
import { checkRlAvailability } from "@/lib/services/ml-stubs";
import { promises as fs } from "fs";
import path from "path";

/**
 * RL hypothesis ranking endpoint.
 *
 * V100 ROOT FIX (BUG #14, P0 CRITICAL): the previous code returned
 * `501 not_implemented` UNCONDITIONALLY — even when `RL_SERVICE_URL`
 * was set. This meant the Phase 4 RL ranker's output NEVER reached the
 * dashboard. The Phase 4 → API handoff was 100% absent.
 *
 * Root fix: implement a real proxy that supports TWO modes:
 *   1. HTTP proxy: when `RL_SERVICE_URL` is set, forward the request to
 *      the Phase 4 RL service (a FastAPI service that exposes
 *      `/candidates?drug=<name>&limit=<N>` etc.).
 *   2. File-based: when `RL_OUTPUT_DIR` is set (the common case for this
 *      project — the pipeline produces `top_candidates_*.csv` files),
 *      read the latest CSV and return its contents as JSON.
 *
 * We NEVER fabricate predictions. If neither env var is set, we return
 * 503 with a clear message.
 */
export async function POST(req: NextRequest) {
  const availability = checkRlAvailability();
  const body = await req.json().catch(() => ({}));
  const { drug, disease, limit = 50 } = body as {
    drug?: string; disease?: string; limit?: number;
  };

  // Mode 1: HTTP proxy to a standalone RL service.
  if (availability.available) {
    const rlUrl = process.env.RL_SERVICE_URL!;
    try {
      const params = new URLSearchParams();
      if (drug) params.set("drug", drug);
      if (disease) params.set("disease", disease);
      params.set("limit", String(limit));
      const upstream = await fetch(
        `${rlUrl.replace(/\/$/, "")}/candidates?${params.toString()}`,
        { method: "GET", headers: { "Accept": "application/json" } },
      );
      const text = await upstream.text();
      return new NextResponse(text, {
        status: upstream.status,
        headers: { "Content-Type": upstream.headers.get("Content-Type") ?? "application/json" },
      });
    } catch (err) {
      return NextResponse.json(
        { error: "rl_service_unreachable", message: String(err) },
        { status: 502 },
      );
    }
  }

  // Mode 2: file-based — read the latest top_candidates CSV from disk.
  // This is the common case: the pipeline produces CSVs, not a live service.
  const outputDir = process.env.RL_OUTPUT_DIR;
  if (outputDir) {
    try {
      const candidates = await readLatestCandidatesCsv(outputDir, { drug, disease, limit });
      if (candidates.length === 0) {
        return NextResponse.json(
          { error: "no_candidates", message: "No candidates found in RL output directory." },
          { status: 404 },
        );
      }
      return NextResponse.json({ candidates, count: candidates.length, source: "csv" });
    } catch (err) {
      return NextResponse.json(
        { error: "rl_output_read_failed", message: String(err) },
        { status: 500 },
      );
    }
  }

  // Neither mode available — return 503 (NOT 501).
  return NextResponse.json(
    {
      error: "service_not_deployed",
      service: availability.service,
      description: availability.description,
      reason: availability.reason,
      documentation: "See Phase 4 of the build plan (RL-Driven Hypothesis Ranking).",
    },
    { status: 503 },
  );
}

export async function GET(req: NextRequest) {
  const { searchParams } = new URL(req.url);
  const drug = searchParams.get("drug") ?? undefined;
  const disease = searchParams.get("disease") ?? undefined;
  const limit = Number(searchParams.get("limit") ?? 50);
  // Reuse POST logic by constructing a synthetic request.
  return POST(new NextRequest(req.url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ drug, disease, limit }),
  }));
}

/**
 * Read the latest top_candidates_*.csv from the output directory and
 * filter by drug/disease. Returns an array of candidate objects.
 */
async function readLatestCandidatesCsv(
  outputDir: string,
  filter: { drug?: string; disease?: string; limit: number },
): Promise<Array<Record<string, unknown>>> {
  const entries = await fs.readdir(outputDir);
  const csvFiles = entries
    .filter((f) => f.startsWith("top_candidates_") && f.endsWith(".csv"))
    .sort()
    .reverse();
  if (csvFiles.length === 0) {
    // Fall back to any candidates CSV.
    const anyCsv = entries
      .filter((f) => f.includes("candidate") && f.endsWith(".csv"))
      .sort()
      .reverse();
    if (anyCsv.length === 0) return [];
    csvFiles.push(anyCsv[0]);
  }
  const latest = csvFiles[0];
  const fullPath = path.join(outputDir, latest);
  const content = await fs.readFile(fullPath, "utf-8");
  const rows = parseCsv(content);
  let filtered = rows;
  if (filter.drug) {
    filtered = filtered.filter((r) =>
      String(r["drug"] ?? r["drug_name"] ?? "").toLowerCase() === filter.drug!.toLowerCase());
  }
  if (filter.disease) {
    filtered = filtered.filter((r) =>
      String(r["disease"] ?? r["disease_name"] ?? "").toLowerCase() === filter.disease!.toLowerCase());
  }
  return filtered.slice(0, filter.limit);
}

/** Minimal CSV parser (handles quoted fields with commas). */
function parseCsv(text: string): Array<Record<string, string>> {
  const lines = text.split(/\r?\n/).filter((l) => l.trim().length > 0);
  if (lines.length === 0) return [];
  const headers = parseCsvLine(lines[0]);
  return lines.slice(1).map((line) => {
    const values = parseCsvLine(line);
    const row: Record<string, string> = {};
    headers.forEach((h, i) => { row[h] = values[i] ?? ""; });
    return row;
  });
}

function parseCsvLine(line: string): string[] {
  const result: string[] = [];
  let current = "";
  let inQuotes = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (inQuotes) {
      if (ch === '"') {
        if (line[i + 1] === '"') { current += '"'; i++; }
        else { inQuotes = false; }
      } else { current += ch; }
    } else {
      if (ch === '"') { inQuotes = true; }
      else if (ch === ",") { result.push(current); current = ""; }
      else { current += ch; }
    }
  }
  result.push(current);
  return result;
}
