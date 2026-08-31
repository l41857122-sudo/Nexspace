import { NextResponse } from "next/server";
import { generateReportPDF } from "@/server/services/reportService";

export async function GET(req: Request, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const { searchParams } = new URL(req.url);
  const format = searchParams.get("format") || "pdf";

  if (format === "geojson") {
    return NextResponse.json({
      type: "FeatureCollection",
      features: [
        {
          type: "Feature",
          geometry: { type: "Point", coordinates: [-118.241, 34.951] },
          properties: { id: "veg", type: "Vegetation Loss", area_km2: 4.2 }
        },
        {
          type: "Feature",
          geometry: { type: "Point", coordinates: [-118.239, 34.958] },
          properties: { id: "struct", type: "New Structure", area_km2: 0.8 }
        }
      ]
    });
  }

  // Generate PDF
  const pdfBuffer = await generateReportPDF({
    id,
    title: "Temporal Assessment Comparison Export",
    reportId: `COMP-EXP-${id}`,
    target: "Sector 12 Bi-Temporal Pair",
    generatedAt: new Date().toISOString(),
    sections: {
      executiveSummary: ["Bi-temporal change comparison rendered a 24.8% delta across sector footprints."],
      spatialAnalysis: { targetAoi: "Sector 12", timestampUtc: "02:12Z", cloudCoverPct: 1.2 },
      temporalTrends: { totalDetected: 2, avgSizeMeters: 250, monthlyData: [2, 4, 1] },
      technicalAppendix: [
        { id: "ANOMALY-01", timestamp: "T1", sensor: "SAR_COSMO_X", coords: "34.951N, -118.241W", resolution: "0.5m", confidence: "96%" }
      ]
    }
  });

  return new Response(new Uint8Array(pdfBuffer), {
    headers: {
      "Content-Type": "application/pdf",
      "Content-Disposition": `attachment; filename="Temporal_Assessment_${id}.pdf"`
    }
  });
}
