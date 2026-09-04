import { NextResponse } from "next/server";

export async function GET(req: Request, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;

  return NextResponse.json({
    id,
    evidenceId: `EVD-${id}`,
    status: "UNVERIFIED"
  });
}

export async function POST(req: Request, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const body = await req.json();
  const action = body.action || "confirm"; // "confirm" | "flag" | "reject"

  return NextResponse.json({
    id,
    action,
    message: `Evidence verification action '${action}' recorded successfully.`
  });
}
