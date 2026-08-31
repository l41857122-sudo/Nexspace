import { NextResponse } from "next/server";

export async function GET(req: Request, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  
  const detectedEntities = [
    {
      id: "entity-1",
      name: "Vessel_Panamax_01",
      type: "Cargo/Container",
      meta: "294 × 32m",
      confidence: "94%",
      statusColor: "bg-emerald-400",
      badgeBorder: "border-emerald-500/30",
      badgeBg: "bg-emerald-500/10",
      badgeText: "text-emerald-400",
      x: 48,
      y: 38,
      width: 140,
      height: 48
    },
    {
      id: "entity-2",
      name: "Infra_Tank_Farm_B",
      type: "Storage/Liquid",
      meta: "14,500 m²",
      confidence: "82%",
      statusColor: "bg-amber-400",
      badgeBorder: "border-amber-500/30",
      badgeBg: "bg-amber-500/10",
      badgeText: "text-amber-300",
      x: 64,
      y: 56,
      width: 110,
      height: 60
    },
    {
      id: "entity-3",
      name: "Vessel_Feeder_12",
      type: "Cargo/Breakbulk",
      meta: "142 × 22m",
      confidence: "91%",
      statusColor: "bg-emerald-400",
      badgeBorder: "border-emerald-500/30",
      badgeBg: "bg-emerald-500/10",
      badgeText: "text-emerald-400",
      x: 28,
      y: 28,
      width: 120,
      height: 44
    }
  ];

  return NextResponse.json({
    queryId: id,
    areaName: "Maasvlakte Industrial Zone",
    targetCount: detectedEntities.length,
    meanPrecisionPct: 88,
    cargoShipsCount: 12,
    infraCount: 4,
    entities: detectedEntities
  });
}
