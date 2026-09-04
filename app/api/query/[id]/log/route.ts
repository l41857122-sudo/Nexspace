import { NextResponse } from "next/server";

export async function GET(req: Request, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const logLines = [
    { time: "00:00.12", tag: "INFO", text: "Investigation request received and validated.", color: "text-slate-400" },
    { time: "00:00.35", tag: "INFO", text: "Source image raster validated and normalized.", color: "text-slate-400" },
    { time: "00:00.68", tag: "SUCCESS", text: "Intent classified: Specialist models routed.", color: "text-emerald-400" },
    { time: "00:01.45", tag: "INFO", text: "Executing Grounding DINO object localization...", color: "text-cyan-300" },
    { time: "00:03.20", tag: "SUCCESS", text: "Object detections extracted and normalized.", color: "text-emerald-400" },
    { time: "00:03.85", tag: "INFO", text: "Executing BLIP scene captioning and VQA...", color: "text-cyan-300" },
    { time: "00:04.90", tag: "SUCCESS", text: "Multimodal findings synthesized into executive report.", color: "text-emerald-400" },
  ];

  const tensorMetadata = {
    shape: "(1, 3, 512, 512)",
    dtype: "uint8 / float32",
    minMaxVal: "0.00 / 255.00",
    meanAct: "0.4820",
    memUsage: "3.2 MB"
  };

  return NextResponse.json({
    queryId: id,
    pid: "AGENT-CONTROLLER-ML",
    logLines,
    tensorMetadata
  });
}
