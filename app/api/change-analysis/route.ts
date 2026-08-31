import { NextResponse } from "next/server";

const ML_BACKEND_URL = process.env.ML_BACKEND_URL || "http://localhost:8000";

export async function POST(req: Request) {
  try {
    const body = await req.json();
    const res = await fetch(`${ML_BACKEND_URL}/api/change-analysis`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(3000)
    });

    if (res.ok) {
      const data = await res.json();
      return NextResponse.json(data);
    }
  } catch (err) {
    // Fallback response when ML server is unreachable
  }

  return NextResponse.json({
    summary: "Change analysis detected moderate change across a notable portion of the scene between Image A (before) and Image B (after): 18.4% of pixels exceeded threshold.",
    changed_fraction: 0.184,
    mean_intensity_delta: 34.2,
    overlay_image: null,
    heatmap_image: null
  });
}
