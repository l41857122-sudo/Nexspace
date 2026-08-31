import { NextResponse } from "next/server";
import { getUploadJobStatus } from "@/server/services/ingestionService";

export async function GET(req: Request, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const status = getUploadJobStatus(id);
  return NextResponse.json(status);
}
