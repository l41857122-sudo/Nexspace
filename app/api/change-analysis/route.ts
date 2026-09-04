import { NextResponse } from "next/server";
import type { NexSpaceChangeAnalysisResponse } from "../../types/nexspace";

const ML_BACKEND_URL = process.env.ML_BACKEND_URL || "http://localhost:8000";

export async function POST(req: Request) {
  let body: any = {};
  try {
    body = await req.json();
  } catch (e) {
    return NextResponse.json({ detail: "Malformed JSON body" }, { status: 400 });
  }

  try {
    const res = await fetch(`${ML_BACKEND_URL}/api/change-analysis`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(60000)
    });

    if (res.ok) {
      const data: NexSpaceChangeAnalysisResponse = await res.json();
      data.backend_status = "online";
      return NextResponse.json(data);
    } else {
      const err = await res.json().catch(() => ({ detail: "Change analysis error" }));
      return NextResponse.json(err, { status: res.status });
    }
  } catch (err: any) {
    console.warn("[Next.js Proxy] FastAPI ML backend offline for change-analysis:", err?.message);
  }

  const fallback: NexSpaceChangeAnalysisResponse = {
    request_id: "req_offline_fallback",
    status: "error",
    summary: "ML backend offline. Start Python server (`python ml_backend/server.py`) on port 8000 for live pixel differencing & anomaly extraction.",
    changed_fraction: 0.0,
    mean_intensity_delta: 0.0,
    anomalies: [],
    anomaly_summary: {
      total_anomalies: 0,
      high_severity: 0,
      medium_severity: 0,
      low_severity: 0
    },
    overlay_image: null,
    heatmap_image: null,
    evidence: [],
    backend_status: "offline_fallback"
  };

  return NextResponse.json(fallback);
}
