import { NextResponse } from "next/server";
import { createOrGetComparison } from "@/server/services/comparisonService";

export async function GET(req: Request, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const result = await createOrGetComparison();
  return NextResponse.json({ ...result, id });
}
