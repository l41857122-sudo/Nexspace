import { NextResponse } from "next/server";

export async function GET(req: Request, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const logLines = [
    { time: "14:02:11", tag: "INFO", text: "Initializing pipeline...", color: "text-slate-400" },
    { time: "14:02:12", tag: null, text: "Loading source array TRQ_64A0_RAW (34.2 GB)", color: "text-slate-500" },
    { time: "14:02:18", tag: "SUCCESS", text: "Data ingestion complete. Checksum matched.", color: "text-emerald-400" },
    { time: "14:02:19", tag: null, text: "Applying radiometric calibration profile...", color: "text-slate-500" },
    { time: "14:02:35", tag: "WARN", text: "Cloud cover detected in sector 9 (coverage ~12%)", color: "text-amber-400" },
    { time: "14:02:40", tag: "SUCCESS", text: "Radiometric correction applied. Tensor shape: [1024, 1024, 6]", color: "text-emerald-400" },
    { time: "14:02:41", tag: "INFO", text: "Booting Neural Extraction Engine (GPU:0, GPU:1)...", color: "text-slate-400" },
    { time: "14:02:43", tag: null, text: "Allocating VRAM... 16000MB reserved", color: "text-slate-500" },
    { time: "14:02:45", tag: null, text: "Commencing deep feature extraction using model RESNET_SAT_v4", color: "text-slate-500" },
    { time: "14:02:50", tag: null, text: "Processing batch 1/64 ...", color: "text-slate-500" },
    { time: "14:02:55", tag: null, text: "Processing batch 18/64 ...", color: "text-slate-500" },
    { time: "14:03:02", tag: null, text: "Processing batch 42/64 ...", color: "text-slate-500" },
    { time: "14:03:09", tag: null, text: "Processing batch 49/64 ...", color: "text-slate-500" }
  ];

  const tensorMetadata = {
    shape: "(1, 3, 1024, 1024)",
    dtype: "float32",
    minMaxVal: "-0.984 / 1.000",
    meanAct: "0.1425",
    memUsage: "12.0 MB"
  };

  return NextResponse.json({
    queryId: id,
    pid: "9021-CUDA",
    logLines,
    tensorMetadata
  });
}
