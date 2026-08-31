import { NextResponse } from "next/server";

export async function GET() {
  return NextResponse.json({
    status: "ok",
    app: "SatQuery AI Next.js Interface",
    ml_backend_url: process.env.ML_BACKEND_URL || "http://localhost:8000"
  });
}
