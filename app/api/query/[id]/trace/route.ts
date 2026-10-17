import { NextResponse } from "next/server";

const ML_BACKEND_URL = process.env.ML_BACKEND_URL || "http://127.0.0.1:8000";

export async function GET(req: Request, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;

  try {
    const res = await fetch(`${ML_BACKEND_URL}/api/investigation/${id}/status`, {
      signal: AbortSignal.timeout(3000),
    });

    if (res.ok) {
      const data = await res.json();
      return NextResponse.json({
        queryId: id,
        status: (data.status || "ACTIVE").toUpperCase(),
        totalProgressPct: typeof data.progress === "number" ? Math.round(data.progress * 100) : 100,
        currentStage: data.current_stage || "COMPLETED",
        stages: Array.isArray(data.stages) ? data.stages : [],
        provenance: data.provenance || "UNAVAILABLE",
      });
    }
  } catch {
    // Backend unavailable or request timed out
  }

  return NextResponse.json({
    queryId: id,
    status: "UNAVAILABLE",
    totalProgressPct: 0,
    currentStage: "NONE",
    stages: [],
    provenance: "UNAVAILABLE",
    message: "No live telemetry available for this investigation ID.",
  });
}
