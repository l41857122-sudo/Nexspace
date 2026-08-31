import { NextResponse } from "next/server";
import { createOrGetComparison } from "@/server/services/comparisonService";

export async function POST(req: Request) {
  try {
    const body = await req.json();
    const result = await createOrGetComparison(body.sceneBeforeId, body.sceneAfterId);
    return NextResponse.json(result);
  } catch (err: any) {
    return NextResponse.json({ error: { code: "SERVER_ERROR", message: err.message } }, { status: 500 });
  }
}
