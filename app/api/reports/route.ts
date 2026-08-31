import { NextResponse } from "next/server";

export async function POST(req: Request) {
  try {
    const body = await req.json();
    const id = `rep_${Date.now()}`;
    return NextResponse.json({
      id,
      title: body.title || "Vessel Activity Assessment Report",
      comparisonId: body.comparisonId || "comp_b492_xt_p",
      pdfUrl: `/api/reports/${id}/pdf`,
      createdAt: new Date().toISOString()
    });
  } catch (err: any) {
    return NextResponse.json({ error: { code: "SERVER_ERROR", message: err.message } }, { status: 500 });
  }
}

export async function GET() {
  return NextResponse.json([
    {
      id: "SQ-REP-2023-11A",
      title: "Vessel Activity Assessment Report",
      target: "Sector 7B (South China Sea)",
      generatedAt: "2023-10-27T08:14Z",
      pdfUrl: "/api/reports/SQ-REP-2023-11A/pdf"
    }
  ]);
}
