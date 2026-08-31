import { NextResponse } from "next/server";

export async function GET(req: Request, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;

  return NextResponse.json({
    id,
    reportId: "SQ-REP-2023-11A",
    title: "Vessel Activity Assessment Report",
    target: "Sector 7B (South China Sea)",
    generatedAt: "2023-10-27T08:14Z",
    sections: {
      executiveSummary: [
        "Analysis of multispectral imagery acquired between Oct 15 and Oct 25 indicates a 14.2% increase in large vessel concentration within the designated AoI (Area of Interest). Automated neural models identified a clustering pattern consistent with recent maritime fleet movements in adjacent international shipping corridors.",
        "Confidence levels for primary detection models exceed 94%, with minimal cloud occlusion (avg 3.2%) during the observation window. Strategic recommendation involves continued daily automated tasking over coordinates 12°34'N 114°21'E."
      ],
      spatialAnalysis: {
        targetAoi: "Sector 7B Target AOI",
        timestampUtc: "21:14:07 UTC",
        cloudCoverPct: 3.2
      },
      temporalTrends: {
        totalDetected: 342,
        avgSizeMeters: 114,
        monthlyData: [38, 52, 46, 71, 90, 100, 82]
      },
      technicalAppendix: [
        {
          id: "SQ-IMG-0812",
          timestamp: "2023-10-25T14:32:11",
          sensor: "SAR-X Band",
          coords: "12.56N, 114.35E",
          resolution: "0.5m GSD",
          confidence: "98.2%"
        },
        {
          id: "SQ-IMG-0813",
          timestamp: "2023-10-26T02:11:45",
          sensor: "Sentinel-2 MSI",
          coords: "12.55N, 114.33E",
          resolution: "10m GSD",
          confidence: "94.1%"
        },
        {
          id: "SQ-IMG-0814",
          timestamp: "2023-10-27T08:00:02",
          sensor: "WorldView-3",
          coords: "12.58N, 114.37E",
          resolution: "0.3m GSD",
          confidence: "99.0%"
        }
      ]
    }
  });
}
