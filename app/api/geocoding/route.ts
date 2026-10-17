import { NextResponse } from "next/server";

const ML_BACKEND_URL = process.env.ML_BACKEND_URL || "http://localhost:8000";

export async function POST(req: Request) {
  try {
    const body = await req.json();

    const response = await fetch(`${ML_BACKEND_URL}/api/geocoding`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
    });

    const data = await response.json();
    return NextResponse.json(data, { status: response.status });
  } catch (error: any) {
    return NextResponse.json(
      {
        query: "",
        display_name: "Geocoding service unavailable",
        latitude: 0,
        longitude: 0,
        bbox: [],
        source: "error",
        geocoding_available: false,
        error: error?.message || "Connection refused",
      },
      { status: 503 }
    );
  }
}
