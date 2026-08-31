import { NextResponse } from "next/server";

export async function GET(req: Request, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;

  return NextResponse.json({
    id,
    evidenceId: "SQ-2023-X992",
    sensor: "Sentinel-2B",
    coordinates: "LAT: 34.9522° N · LON: 118.2437° W · ELEV: 412M",
    spectralBands: [
      { band: "B04 (Red 665nm)", value: 0.142, color: "bg-red-400" },
      { band: "B08 (NIR 842nm)", value: 0.875, color: "bg-cyan-400" },
      { band: "B11 (SWIR 1610nm)", value: 0.329, color: "bg-amber-400" }
    ],
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
