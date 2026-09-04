"use client";

import { useState, useMemo } from "react";
import Link from "next/link";
import {
  Download,
  Info,
  Layers,
  CheckCircle2,
  Cpu,
  Search,
  Crosshair,
  Sparkles,
  ArrowRight,
} from "lucide-react";
import Sidebar from "./Sidebar";
import type { CanonicalSourceImage, CanonicalInvestigationState } from "../types/nexspace";
import { SAMPLE_OPTICAL_PORT } from "../utils/sampleImages";
import { normalizeBox } from "./QueryPage";
import {
  getCurrentInvestigation,
  getActiveSourceImage,
  DEFAULT_DEMO_SOURCE,
} from "../utils/investigationStorage";

// ---------------------------------------------
// Header
// ---------------------------------------------
function Header({
  investigation,
  sourceImage,
}: {
  investigation: CanonicalInvestigationState | null;
  sourceImage: CanonicalSourceImage;
}) {
  const isUpload = sourceImage.source === "upload";

  const handleExportLog = () => {
    if (!investigation) return;
    const logData = {
      investigation_id: investigation.investigation_id,
      query: investigation.query,
      timestamp: investigation.timestamp,
      source_image: {
        filename: sourceImage.filename,
        mediaType: sourceImage.mediaType,
        source: sourceImage.source,
      },
      response: investigation.response,
    };
    const blob = new Blob([JSON.stringify(logData, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `execution_log_${investigation.investigation_id || "active"}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 px-4 sm:px-6 lg:px-8 py-4 border-b border-slate-800/80 bg-[#0d1826]/40 backdrop-blur-md">
      <div>
        <div className="flex flex-wrap items-center gap-2 mb-1.5 font-mono">
          <span className="text-[10px] bg-slate-800/60 text-slate-300 rounded-full px-2 py-0.5 tracking-wide border border-slate-700/60">
            EXECUTION TRACE
          </span>
          <span className="flex items-center gap-1 text-[10px] bg-emerald-500/10 text-emerald-300 rounded-full px-2 py-0.5 border border-emerald-500/20">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
            {investigation ? "COMPLETED" : "IDLE"}
          </span>
          <span className={`text-[10px] px-2 py-0.5 rounded-full border ${
            isUpload
              ? "text-emerald-300 bg-emerald-500/15 border-emerald-500/30"
              : "text-blue-300 bg-blue-500/15 border-blue-500/30"
          }`}>
            Source: {sourceImage.filename} ({isUpload ? "User Upload" : "Demo Tile"})
          </span>
        </div>
        <p className="text-base font-semibold text-white tracking-tight">
          Query ID: <span className="font-mono text-cyan-300">{investigation?.investigation_id || "INV-STANDBY"}</span>
        </p>
        <p className="text-[11px] text-slate-400 mt-0.5 font-mono">
          Query: &ldquo;{investigation?.query || "No query executed yet"}&rdquo;
        </p>
      </div>
      <button
        onClick={handleExportLog}
        disabled={!investigation}
        className="self-start sm:self-auto flex items-center gap-1.5 bg-slate-900/60 hover:bg-slate-800/80 disabled:opacity-40 border border-slate-800 hover:border-cyan-500/30 transition-all text-xs text-slate-300 hover:text-cyan-300 rounded-lg px-3 py-1.5 font-mono active:scale-95 cursor-pointer"
      >
        <Download size={12} />
        <span>Export Log (JSON)</span>
      </button>
    </div>
  );
}

// ---------------------------------------------
// Key-value detail box
// ---------------------------------------------
function DetailBox({
  label,
  value,
  valueColor,
}: {
  label: string;
  value: string;
  valueColor?: string;
}) {
  return (
    <div className="bg-slate-900/60 border border-slate-800/80 rounded-lg px-3 py-2 flex-1 min-w-[140px]">
      <p className="text-[9px] text-slate-400 uppercase tracking-widest font-mono">
        {label}
      </p>
      <p
        className={`text-xs font-mono mt-0.5 truncate ${
          valueColor ?? "text-slate-200"
        }`}
      >
        {value}
      </p>
    </div>
  );
}

// ---------------------------------------------
// Pipeline step component
// ---------------------------------------------
function PipelineStep({
  icon: Icon,
  title,
  duration,
  children,
}: {
  icon: React.ElementType;
  title: string;
  duration?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="flex gap-3.5">
      <div className="flex flex-col items-center">
        <div className="w-8 h-8 rounded-full bg-slate-900/80 border border-cyan-500/30 flex items-center justify-center text-cyan-400 shrink-0 shadow-[0_0_10px_rgba(6,182,212,0.2)]">
          <Icon size={14} />
        </div>
        <div className="flex-1 w-px bg-slate-800 my-1.5" />
      </div>

      <div className="flex-1 pb-6">
        <div className="flex items-center justify-between mb-2">
          <p className="text-xs sm:text-sm text-slate-100 font-semibold tracking-tight">{title}</p>
          {duration && (
            <span className="text-[10px] text-cyan-400 font-mono bg-cyan-500/10 px-2 py-0.5 rounded border border-cyan-500/20">
              {duration}
            </span>
          )}
        </div>
        {children}
      </div>
    </div>
  );
}

// ---------------------------------------------
// Dynamic Pipeline Execution Steps List
// ---------------------------------------------
function PipelineSteps({
  investigation,
  sourceImage,
}: {
  investigation: CanonicalInvestigationState | null;
  sourceImage: CanonicalSourceImage;
}) {
  const detections = investigation?.response?.grounding?.detections || [];
  const caption = typeof investigation?.response?.optical_caption === "string"
    ? investigation.response.optical_caption
    : (investigation?.response?.optical_caption as unknown as { caption?: string })?.caption || "Scene captioning completed.";
  const vqa = investigation?.response?.vqa_results || [];
  const hasSar = Boolean(investigation?.sar_image);

  if (!investigation) {
    return (
      <div className="w-full border border-slate-800/90 bg-[#0c1624]/60 backdrop-blur-md rounded-xl p-8 text-center shadow-[0_4px_24px_rgba(0,0,0,0.3)] space-y-4">
        <Layers size={32} className="text-slate-600 mx-auto" />
        <h3 className="text-base font-semibold text-white">No Active Investigation Trace</h3>
        <p className="text-xs text-slate-400 font-mono max-w-md mx-auto">
          Execute a query with an uploaded satellite image in the Query Terminal to view the live 5-stage agentic execution trace.
        </p>
        <Link
          href="/query"
          className="inline-flex items-center gap-2 bg-cyan-500 hover:bg-cyan-400 text-[#0a1420] text-xs font-bold px-4 py-2 rounded-lg transition-all"
        >
          <Search size={14} />
          <span>Open Query Terminal</span>
        </Link>
      </div>
    );
  }

  return (
    <div className="w-full border border-slate-800/90 bg-[#0c1624]/60 backdrop-blur-md rounded-xl p-5 shadow-[0_4px_24px_rgba(0,0,0,0.3)]">
      <p className="text-xs font-semibold text-white tracking-tight uppercase tracking-wider font-mono mb-5 flex items-center gap-2">
        <Layers size={13} className="text-cyan-400" />
        <span>Live Agentic Execution Pipeline (5 Stages)</span>
      </p>

      <div className="flex-1">
        {/* Stage 1: Ingestion */}
        <PipelineStep icon={CheckCircle2} title="Stage 1: Input Ingestion & Raster Validation" duration="✓ Verified">
          <div className="flex flex-wrap gap-3">
            <DetailBox label="Input Raster" value={sourceImage.filename} />
            <DetailBox label="Source Origin" value={sourceImage.source === "upload" ? "User Upload" : "Verified Tile"} />
            <DetailBox label="MIME Type" value={sourceImage.mediaType} valueColor="text-cyan-300 font-medium" />
          </div>
        </PipelineStep>

        {/* Stage 2: Intent Classification */}
        <PipelineStep icon={Search} title="Stage 2: Intent Classification & Tool Routing" duration="✓ Routed">
          <div className="text-[11px] text-slate-400 space-y-1 mb-2 font-mono bg-slate-900/60 p-3 rounded-lg border border-slate-800/80">
            <p className="text-slate-200"><strong>Query Intent:</strong> &ldquo;{investigation.query}&rdquo;</p>
            <p>&gt; Tools Activated: Grounding DINO Localization, BLIP Captioning{vqa.length > 0 ? ", RSVQA" : ""}{hasSar ? ", Optical+SAR Baseline" : ""}</p>
          </div>
        </PipelineStep>

        {/* Stage 3: Neural Model Execution */}
        <PipelineStep icon={Cpu} title="Stage 3: Specialist Neural Model Inference" duration="✓ Complete">
          <div className="flex flex-wrap gap-3 mb-2">
            <DetailBox label="Object Localization" value="Grounding DINO (Swin-T)" valueColor="text-emerald-400" />
            <DetailBox label="Scene Captioner" value="BLIP (Salesforce/blip-base)" valueColor="text-cyan-300" />
          </div>
          <p className="text-[11px] text-emerald-400 font-mono flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
            Detections Extracted: {detections.length} bounding candidate(s)
          </p>
        </PipelineStep>

        {/* Stage 4: Evidence Normalization */}
        <PipelineStep icon={Crosshair} title="Stage 4: Spatial Normalization & Evidence Extraction" duration="✓ Normalized">
          <div className="text-[11px] text-slate-400 font-mono bg-slate-900/60 p-3 rounded-lg border border-slate-800/80 space-y-1">
            <p>&gt; Normalized Coordinate Scale: [ymin, xmin, ymax, xmax] normalized (0-1000 range)</p>
            <p>&gt; Total Spatial Evidence Nodes: {detections.length} node(s) verified</p>
          </div>
        </PipelineStep>

        {/* Stage 5: Synthesis */}
        <PipelineStep icon={Sparkles} title="Stage 5: Multimodal Synthesis & Natural Language Report" duration="✓ Synthesized">
          <div className="text-[11px] text-slate-300 font-mono bg-slate-900/60 p-3 rounded-lg border border-slate-800/80 space-y-1.5">
            <div><strong className="text-cyan-300">Executive Finding:</strong> {caption}</div>
            {vqa.length > 0 && (
              <div><strong className="text-cyan-300">VQA Responses:</strong> {vqa.map(v => `${v.question} → ${v.answer}`).join("; ")}</div>
            )}
          </div>
        </PipelineStep>
      </div>
    </div>
  );
}

// ---------------------------------------------
// Node Inspector panel with Real Investigation Data
// ---------------------------------------------
function NodeInspector({
  investigation,
  sourceImage,
}: {
  investigation: CanonicalInvestigationState | null;
  sourceImage: CanonicalSourceImage;
}) {
  const detectionsCount = investigation?.response?.grounding?.detections?.length || 0;
  const normalizedBoxes = useMemo(() => {
    const raw = investigation?.response?.grounding?.detections || [];
    return raw
      .map(d => normalizeBox(d as unknown as Record<string, unknown>))
      .filter((b): b is [number, number, number, number] => b !== null);
  }, [investigation]);

  return (
    <div className="w-full border border-slate-800/90 bg-[#0c1624]/60 backdrop-blur-md rounded-xl overflow-hidden flex flex-col shadow-[0_4px_24px_rgba(0,0,0,0.3)] space-y-4">
      <div className="flex items-center gap-1.5 px-4 py-3 border-b border-slate-800/80 font-mono">
        <Info size={13} className="text-cyan-400" />
        <p className="text-sm font-semibold text-white tracking-tight">Active Investigation Node Inspector</p>
      </div>

      {/* Raster & Sensor Metadata */}
      <div className="px-4 space-y-3">
        <p className="text-[10px] text-slate-400 uppercase tracking-widest font-mono">
          Raster Frame &amp; Input Metadata
        </p>

        <div className="space-y-2 text-[11px] font-mono bg-slate-900/60 p-3 rounded-lg border border-slate-800/80">
          <div className="flex justify-between">
            <span className="text-slate-400">Filename</span>
            <span className="text-cyan-300 font-semibold truncate max-w-[160px]">{sourceImage.filename}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-400">Media Type</span>
            <span className="text-slate-200">{sourceImage.mediaType}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-400">Source Type</span>
            <span className="text-slate-200">{sourceImage.source === "upload" ? "User Upload" : "Demo Catalog"}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-400">Detections</span>
            <span className="text-emerald-400 font-semibold">{detectionsCount} objects</span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-400">Investigation ID</span>
            <span className="text-slate-200 truncate max-w-[140px]">{investigation?.investigation_id || "N/A"}</span>
          </div>
        </div>
      </div>

      {/* Detections preview */}
      {normalizedBoxes.length > 0 && (
        <div className="px-4 pb-4 space-y-2">
          <p className="text-[10px] text-slate-400 uppercase tracking-widest font-mono">
            Normalized Bounding Boxes (0-1000)
          </p>
          <div className="max-h-40 overflow-y-auto space-y-1 bg-slate-950/80 p-2.5 rounded-lg border border-slate-800 text-[10px] font-mono text-cyan-300">
            {normalizedBoxes.map((b, idx) => (
              <div key={idx}>• Target #{idx + 1}: [{b.join(", ")}]</div>
            ))}
          </div>
        </div>
      )}

      {/* Quick link */}
      <div className="px-4 pb-4">
        <Link
          href="/evidence"
          className="w-full flex items-center justify-center gap-1.5 bg-slate-900/80 hover:bg-slate-800 border border-slate-800 hover:border-cyan-500/40 text-slate-300 hover:text-cyan-300 rounded-lg py-2 text-xs font-mono transition-colors"
        >
          <span>Open Evidence Viewer</span>
          <ArrowRight size={12} />
        </Link>
      </div>
    </div>
  );
}

// ---------------------------------------------
// Bottom status bar
// ---------------------------------------------
function StatusBar({
  investigation,
  sourceImage,
}: {
  investigation: CanonicalInvestigationState | null;
  sourceImage: CanonicalSourceImage;
}) {
  return (
    <div className="flex flex-col sm:flex-row items-center justify-between gap-2 py-3 px-4 sm:px-8 border-t border-slate-800/80 text-[10px] font-mono text-slate-400">
      <span className="flex items-center gap-1.5 text-cyan-300">
        <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse" />
        PIPELINE STATUS: {investigation ? "INVESTIGATION SYNCHRONIZED" : "STANDBY"}
      </span>
      <span>SOURCE: {sourceImage.filename} · SPATIAL FRAME: 512x512 PIXEL COORDINATES</span>
    </div>
  );
}

// ---------------------------------------------
// Execution Log Page Main Component
// ---------------------------------------------
export default function ExecutionLogPage() {
  const [investigationState] = useState<CanonicalInvestigationState | null>(() => {
    return getCurrentInvestigation();
  });

  const sourceImage = useMemo<CanonicalSourceImage>(() => {
    if (investigationState?.source_image?.dataUrl) {
      return investigationState.source_image;
    }
    const active = getActiveSourceImage(false);
    if (active) return active;
    return DEFAULT_DEMO_SOURCE;
  }, [investigationState]);

  return (
    <div className="flex flex-col lg:flex-row min-h-screen bg-[#0a1420] text-white font-sans overflow-x-hidden">
      <Sidebar />

      <div className="flex-1 flex flex-col min-w-0">
        <Header investigation={investigationState} sourceImage={sourceImage} />

        <main className="flex-1 p-4 sm:p-6 lg:p-8 grid grid-cols-1 lg:grid-cols-3 gap-5 items-start overflow-y-auto">
          <div className="lg:col-span-2">
            <PipelineSteps investigation={investigationState} sourceImage={sourceImage} />
          </div>
          <div className="lg:col-span-1">
            <NodeInspector investigation={investigationState} sourceImage={sourceImage} />
          </div>
        </main>

        <StatusBar investigation={investigationState} sourceImage={sourceImage} />
      </div>
    </div>
  );
}