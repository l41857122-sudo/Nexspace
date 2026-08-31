// @ts-ignore
import PDFDocument from "pdfkit";

export interface ReportData {
  id: string;
  title: string;
  reportId: string;
  target: string;
  generatedAt: string;
  sections: {
    executiveSummary: string[];
    spatialAnalysis: { targetAoi: string; timestampUtc: string; cloudCoverPct: number };
    temporalTrends: { totalDetected: number; avgSizeMeters: number; monthlyData: number[] };
    technicalAppendix: Array<{ id: string; timestamp: string; sensor: string; coords: string; resolution: string; confidence: string }>;
  };
}

export function generateReportPDF(data: ReportData): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    try {
      const doc = new PDFDocument({ margin: 40 });
      const buffers: Buffer[] = [];

      doc.on("data", (chunk: any) => buffers.push(chunk));
      doc.on("end", () => resolve(Buffer.concat(buffers)));

      // Title & Header
      doc.fontSize(18).fillColor("#06b6d4").text("SatQuery AI — Orbital Intelligence Report", { align: "left" });
      doc.fontSize(14).fillColor("#1e293b").text(data.title);
      doc.fontSize(9).fillColor("#64748b").text(`ID: ${data.reportId} | Generated: ${data.generatedAt} | Target: ${data.target}`);
      doc.moveDown(1.5);

      // Section 1.0 Executive Summary
      doc.fontSize(12).fillColor("#0f172a").text("1.0 Executive Summary", { underline: true });
      doc.moveDown(0.5);
      data.sections.executiveSummary.forEach((paragraph) => {
        doc.fontSize(10).fillColor("#334155").text(paragraph, { align: "justify" });
        doc.moveDown(0.5);
      });
      doc.moveDown(1);

      // Section 2.0 Spatial Analysis
      doc.fontSize(12).fillColor("#0f172a").text("2.0 Spatial Analysis", { underline: true });
      doc.fontSize(10).fillColor("#334155").text(`Target AOI: ${data.sections.spatialAnalysis.targetAoi}`);
      doc.fontSize(10).fillColor("#334155").text(`Acquisition Time: ${data.sections.spatialAnalysis.timestampUtc}`);
      doc.fontSize(10).fillColor("#334155").text(`Cloud Occlusion: ${data.sections.spatialAnalysis.cloudCoverPct}%`);
      doc.moveDown(1.5);

      // Section 3.0 Temporal Trends
      doc.fontSize(12).fillColor("#0f172a").text("3.0 Temporal Trends", { underline: true });
      doc.fontSize(10).fillColor("#334155").text(`Total Detected Entities: ${data.sections.temporalTrends.totalDetected}`);
      doc.fontSize(10).fillColor("#334155").text(`Average Feature Size: ${data.sections.temporalTrends.avgSizeMeters}m`);
      doc.fontSize(10).fillColor("#334155").text(`Monthly Detection Distribution: ${data.sections.temporalTrends.monthlyData.join(", ")}`);
      doc.moveDown(1.5);

      // Section 4.0 Technical Appendix
      doc.fontSize(12).fillColor("#0f172a").text("4.0 Technical Appendix", { underline: true });
      doc.moveDown(0.5);

      data.sections.technicalAppendix.forEach((item) => {
        doc.fontSize(9).fillColor("#0f172a").text(`${item.id} | Sensor: ${item.sensor} | Coords: ${item.coords} | Res: ${item.resolution} | Conf: ${item.confidence}`);
      });

      doc.end();
    } catch (err) {
      reject(err);
    }
  });
}
