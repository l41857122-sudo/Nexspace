import { NextResponse } from "next/server";

export async function GET(req: Request, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;

  return NextResponse.json({
    queryId: id,
    status: "active",
    message: "Investigation telemetry and results are managed dynamically per session.",
    entities: []
  });
}
