import { NextResponse } from "next/server";

const ML_BACKEND_URL = process.env.ML_BACKEND_URL || "http://127.0.0.1:8000";

export async function GET(req: Request, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;

  try {
    const res = await fetch(`${ML_BACKEND_URL}/api/investigation/${id}/result`, {
      signal: AbortSignal.timeout(3000),
    });

    if (res.ok) {
      const data = await res.json();
      const trace = data.execution_trace || [];
      const logLines = trace.map((st: any, idx: number) => ({
        time: typeof st.duration_ms === "number" ? `+${st.duration_ms.toFixed(0)}ms` : `00:0${idx + 1}.00`,
        tag: (st.status || "INFO").toUpperCase(),
        text: `[${st.stage || st.name}] ${st.details || st.message || ""}`.trim(),
        color: st.status === "failed" ? "text-rose-400" : "text-emerald-400",
      }));

      return NextResponse.json({
        queryId: id,
        pid: "AGENT-CONTROLLER-ML",
        status: data.status || "completed",
        provenance: data.provenance || "UNAVAILABLE",
        logLines,
      });
    }
  } catch {
    // Backend offline or not found
  }

  return NextResponse.json({
    queryId: id,
    pid: "AGENT-CONTROLLER-ML",
    status: "UNAVAILABLE",
    provenance: "UNAVAILABLE",
    logLines: [],
    message: "No live execution logs available for this query ID.",
  });
}
