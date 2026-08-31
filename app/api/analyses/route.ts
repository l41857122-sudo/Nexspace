import { NextResponse } from "next/server";

const analysesData = [
  {
    id: "an_01",
    name: "Synthetic Aperture Radar (SAR) Analysis - Suez Canal",
    type: "Vessel Wake Detection",
    status: "Processing",
    detail: "82%",
    metadata: "GSD: 8.5m | Cloud: 0%",
    createdAt: "2024-03-15T10:00:00Z"
  },
  {
    id: "an_02",
    name: "Sentinel-2 Multispectral Ingestion",
    type: "Vegetation Index (NDVI)",
    status: "Completed",
    detail: "",
    metadata: "GSD: 10m | Cloud: 12%",
    createdAt: "2024-03-15T09:30:00Z"
  },
  {
    id: "an_03",
    name: "Urban Sprawl Mapping - Lagos",
    type: "Change Detection",
    status: "Awaiting QA",
    detail: "",
    metadata: "GSD: 3m | Cloud: 5%",
    createdAt: "2024-03-15T08:15:00Z"
  },
  {
    id: "an_04",
    name: "Thermal Anomaly Scan - Eastern Europe",
    type: "Infrared Analysis",
    status: "Error",
    detail: "Corrupt Metadata",
    metadata: "N/A",
    createdAt: "2024-03-14T22:00:00Z"
  }
];

export async function GET(req: Request) {
  const { searchParams } = new URL(req.url);
  const status = searchParams.get("status");
  const limit = searchParams.get("limit");

  let filtered = [...analysesData];
  if (status) {
    filtered = filtered.filter((a) => a.status.toLowerCase() === status.toLowerCase());
  }
  if (limit) {
    filtered = filtered.slice(0, parseInt(limit, 10));
  }

  return NextResponse.json(filtered);
}
