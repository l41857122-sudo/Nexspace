import { NextResponse } from "next/server";

const ML_BACKEND_URL = process.env.ML_BACKEND_URL || "http://localhost:8000";

export async function GET(
  req: Request,
  { params }: { params: Promise<{ id: string }> | { id: string } }
) {
  const resolvedParams = await Promise.resolve(params);
  const investigationId = resolvedParams.id;
  try {
    const res = await fetch(
      `${ML_BACKEND_URL}/api/investigation/${encodeURIComponent(investigationId)}/status`,
      {
        method: "GET",
        headers: { "Content-Type": "application/json" },
      }
    );

    const data = await res.json();
    return NextResponse.json(data, { status: res.status });
  } catch (err: any) {
    return NextResponse.json(
      {
        investigation_id: investigationId,
        status: "failed",
        error: `ML Backend unreachable: ${err?.message || "Connection error"}`,
      },
      { status: 503 }
    );
  }
}
