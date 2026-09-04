import { NextResponse } from "next/server";

const ML_BACKEND_URL = process.env.ML_BACKEND_URL || "http://localhost:8000";

export async function POST(req: Request) {
  let body: any = {};
  try {
    body = await req.json();
  } catch (e) {
    return NextResponse.json({ detail: "Malformed JSON body" }, { status: 400 });
  }

  try {
    const res = await fetch(`${ML_BACKEND_URL}/api/geojson`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(10000)
    });

    if (res.ok) {
      const data = await res.json();
      return NextResponse.json(data);
    } else {
      const err = await res.json().catch(() => ({ detail: "GeoJSON export error" }));
      return NextResponse.json(err, { status: res.status });
    }
  } catch (err: any) {
    console.warn("[Next.js Proxy] FastAPI ML backend offline for GeoJSON:", err?.message);
    return NextResponse.json(
      {
        type: "FeatureCollection",
        geospatial_available: false,
        features: []
      },
      { status: 503 }
    );
  }
}
