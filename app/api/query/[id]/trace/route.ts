import { NextResponse } from "next/server";
import { generateExecutionStages } from "@/server/services/queryService";

export async function GET(req: Request, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const stages = generateExecutionStages(id);
  return NextResponse.json({
    queryId: id,
    targetRegion: "Quadrant 7A, Sector North",
    operationName: "Geospatial Feature Extraction",
    status: "ACTIVE",
    totalProgressPct: 75,
    eta: "02m 34s",
    throughput: "4.2 GB/s",
    activeNodes: "2 Compute",
    stages
  });
}
