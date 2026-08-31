import { NextResponse } from "next/server";

const scenes = [
  {
    id: "sc_01",
    source: "Sentinel-2 MSI",
    capturedAt: "2024-03-15T10:53:41Z",
    gsdMeters: 10,
    cloudCoverPct: 4.2,
    bands: ["RGB", "NIR", "SWIR"],
    status: "indexed",
    thumbnailUrl: "/sample_thumb_1.jpg"
  },
  {
    id: "sc_02",
    source: "Landsat-8 OLI",
    capturedAt: "2024-03-14T14:20:00Z",
    gsdMeters: 30,
    cloudCoverPct: 1.2,
    bands: ["RGB", "NIR"],
    status: "indexed",
    thumbnailUrl: "/sample_thumb_2.jpg"
  }
];

export async function GET() {
  return NextResponse.json(scenes);
}
