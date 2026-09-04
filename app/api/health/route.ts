import { NextResponse } from "next/server";
import type { NexSpaceHealthResponse } from "../../types/nexspace";

const ML_BACKEND_URL = process.env.ML_BACKEND_URL || "http://localhost:8000";

export async function GET() {
  try {
    const res = await fetch(`${ML_BACKEND_URL}/api/health`, {
      method: "GET",
      signal: AbortSignal.timeout(3000)
    });

    if (res.ok) {
      const data: NexSpaceHealthResponse = await res.json();
      data.ml_backend_url = ML_BACKEND_URL;
      return NextResponse.json(data);
    }
  } catch (err: any) {
    // Backend offline
  }

  const offlineResponse: NexSpaceHealthResponse = {
    request_id: "req_health_offline",
    status: "degraded",
    service: "NexSpace Next.js (FastAPI Backend Offline)",
    version: "2.5.0",
    capabilities: {
      captioning: "offline",
      grounding: "offline",
      vqa: "offline",
      change_analysis: "offline",
      anomaly_extraction: "offline",
      optical_sar_fusion: "offline",
      geospatial: "offline"
    },
    ml_backend_url: ML_BACKEND_URL
  };

  return NextResponse.json(offlineResponse);
}
