import { NextResponse } from "next/server";
import { parseQueryFilters } from "@/server/services/queryService";

export async function GET(req: Request, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const parsedFilters = await parseQueryFilters("Identify maritime vessels in Sector 9B");

  return NextResponse.json({
    id,
    rawText: "Identify maritime vessels over 50m length in the Malacca Strait using SAR imagery.",
    parsedFilters,
    status: "completed",
    execTimeMs: 1240,
    createdAt: new Date().toISOString()
  });
}
