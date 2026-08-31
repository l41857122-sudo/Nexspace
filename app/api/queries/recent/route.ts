import { NextResponse } from "next/server";

const recentQueries = [
  {
    id: "q_778_delta",
    text: "Detect new construction activities near coordinates 25.276987, 55.296249 in the last 36 days.",
    time: "1.2s",
    status: "Completed",
    createdAt: "2024-03-15T10:14:00Z"
  },
  {
    id: "q_9482_a",
    text: "Identify maritime vessels over 50m length in the Malacca Strait using SAR imagery.",
    time: "4.5s",
    status: "Completed",
    createdAt: "2024-03-15T09:42:00Z"
  },
  {
    id: "q_1092_c",
    text: "Show me NDVI changes in the Amazon basin during Q3 stream analysis...",
    time: null,
    status: "Running",
    createdAt: "2024-03-15T10:20:00Z"
  }
];

export async function GET() {
  return NextResponse.json(recentQueries);
}
