"use client";

import { useState, useEffect } from "react";
import {
  Satellite,
  LayoutGrid,
  Search,
  GitCompare,
  Upload,
  FileText,
  Settings,
  Wind,
  CloudFog,
  Waves,
  ScanSearch,
  Download,
  Info,
  Layers,
} from "lucide-react";

import Sidebar from "./Sidebar";

// ---------------------------------------------
// Header
// ---------------------------------------------
function Header() {
  return (
    <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 px-4 sm:px-6 lg:px-8 py-4 border-b border-slate-800/80 bg-[#0d1826]/40 backdrop-blur-md">
      <div>
        <div className="flex items-center gap-2 mb-1.5 font-mono">
          <span className="text-[10px] bg-slate-800/60 text-slate-300 rounded-full px-2 py-0.5 tracking-wide border border-slate-700/60">
            EXECUTION TRACE
          </span>
          <span className="flex items-center gap-1 text-[10px] bg-emerald-500/10 text-emerald-300 rounded-full px-2 py-0.5 border border-emerald-500/20">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
            COMPLETED
          </span>
        </div>
        <p className="text-base font-semibold text-white tracking-tight">
          Query ID: 9482-A-VSD
        </p>
        <p className="text-[11px] text-slate-400 mt-0.5 font-mono">
          04:12.35s total execution time · 4 Pipeline Stages
        </p>
      </div>
      <button className="self-start sm:self-auto flex items-center gap-1.5 bg-slate-900/60 hover:bg-slate-800/80 border border-slate-800 hover:border-cyan-500/30 transition-all text-xs text-slate-300 hover:text-cyan-300 rounded-lg px-3 py-1.5 font-mono active:scale-95 cursor-pointer">
        <Download size={12} />
        <span>Export Log</span>
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
    <div className="bg-slate-900/60 border border-slate-800/80 rounded-lg px-3 py-2 flex-1">
      <p className="text-[9px] text-slate-400 uppercase tracking-widest font-mono">
        {label}
      </p>
      <p
        className={`text-xs font-mono mt-0.5 ${
          valueColor ?? "text-slate-200"
        }`}
      >
        {value}
      </p>
    </div>
  );
}

// ---------------------------------------------
// Pipeline step
// ---------------------------------------------
function PipelineStep({
  icon: Icon,
  title,
  duration,
  children,
}: {
  icon: React.ElementType;
  title: string;
  duration: string;
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
          <span className="text-[10px] text-cyan-400 font-mono bg-cyan-500/10 px-2 py-0.5 rounded border border-cyan-500/20">
            {duration}
          </span>
        </div>
        {children}
      </div>
    </div>
  );
}

// ---------------------------------------------
// Pipeline steps list
// ---------------------------------------------
function PipelineSteps() {
  return (
    <div className="w-full border border-slate-800/90 bg-[#0c1624]/60 backdrop-blur-md rounded-xl p-5 shadow-[0_4px_24px_rgba(0,0,0,0.3)]">
      <p className="text-xs font-semibold text-white tracking-tight uppercase tracking-wider font-mono mb-5 flex items-center gap-2">
        <Layers size={13} className="text-cyan-400" />
        <span>Pipeline Execution Stages</span>
      </p>

      <div className="flex-1">
        <PipelineStep
          icon={Wind}
          title="Atmospheric Correction (Sen2Cor)"
          duration="0.45s"
        >
          <div className="flex flex-wrap sm:flex-nowrap gap-3">
            <DetailBox label="Input Source" value="Sentinel-2 (L2A) T31UFQ" />
            <DetailBox
              label="Model / Routine"
              value="AtmosphericCorr-v2.1"
              valueColor="text-amber-400 font-medium"
            />
          </div>
        </PipelineStep>

        <PipelineStep icon={CloudFog} title="Cloud Masking (s2cloudless)" duration="1.12s">
          <div className="text-[11px] text-slate-400 space-y-1 mb-2 font-mono bg-slate-900/60 p-3 rounded-lg border border-slate-800/80">
            <p>&gt; Threshold set: &lt; 15% opacity</p>
            <p>&gt; Masking applied to 8.4% of total tensor area</p>
            <p>&gt; Confidence score: 0.982 (Nominal)</p>
          </div>
        </PipelineStep>

        <PipelineStep
          icon={Waves}
          title="Identify Water Bodies (NDWI Isolation)"
          duration="2.05s"
        >
          <p className="text-xs text-slate-400 mb-2">
            NDWI index calculation and morphological operations to isolate navigable regions.
          </p>
          <div className="flex flex-wrap sm:flex-nowrap gap-3">
            <DetailBox label="Algorithm" value="Otsu Thresholding (NDWI)" />
            <DetailBox
              label="Output Tensor"
              value="water_mask_v3.npy [1024×1024]"
              valueColor="text-cyan-300 font-medium"
            />
          </div>
        </PipelineStep>

        <PipelineStep
          icon={ScanSearch}
          title="Object Detection (YOLOv8 Aerial-Maritime)"
          duration="0.73s"
        >
          <div className="flex flex-wrap sm:flex-nowrap gap-3 mb-2">
            <DetailBox label="Model" value="yolo-v8-det-marine" />
            <DetailBox label="Weights" value="marine_v3_det_x300.pt" />
          </div>
          <p className="text-[11px] text-emerald-400 font-mono flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
            Detected: 14 bounding boxes (Mean Conf: 91.4%)
          </p>
        </PipelineStep>
      </div>
    </div>
  );
}

// ---------------------------------------------
// Node Inspector panel
// ---------------------------------------------
function NodeInspector({ tensorMetadata }: { tensorMetadata?: any }) {
  return (
    <div className="w-full border border-slate-800/90 bg-[#0c1624]/60 backdrop-blur-md rounded-xl overflow-hidden flex flex-col shadow-[0_4px_24px_rgba(0,0,0,0.3)]">
      <div className="flex items-center gap-1.5 px-4 py-3 border-b border-slate-800/80 font-mono">
        <Info size={13} className="text-cyan-400" />
        <p className="text-sm font-semibold text-white tracking-tight">Node Inspector</p>
      </div>

      {/* Mini map visualization */}
      <div className="relative h-40 bg-[#08121e] border-b border-slate-800/80 overflow-hidden">
        <svg
          className="absolute inset-0 w-full h-full opacity-60"
          viewBox="0 0 300 160"
        >
          <path
            d="M20,20 Q60,10 90,40 T160,50 Q200,60 220,30 T280,60"
            stroke="#06b6d4"
            strokeWidth="1"
            fill="none"
          />
          <path
            d="M10,80 Q80,70 120,100 T240,90 Q270,100 290,80"
            stroke="#06b6d4"
            strokeWidth="1"
            fill="none"
            opacity="0.5"
          />
          <path
            d="M30,130 Q100,120 150,140 T290,120"
            stroke="#06b6d4"
            strokeWidth="1"
            fill="none"
            opacity="0.4"
          />
        </svg>
        <div className="absolute bottom-2 left-2 right-2 flex justify-between text-[9px] font-mono text-slate-400 bg-slate-900/80 px-2 py-0.5 rounded border border-slate-800">
          <span>LON: 43.12984</span>
          <span>LAT: -12.40871</span>
          <span>RES: 10m/px</span>
        </div>
      </div>

      {/* Tensor metadata */}
      <div className="p-4 flex-1">
        <p className="text-[10px] text-slate-400 uppercase tracking-widest mb-3 font-mono">
          Tensor Metadata (Output)
        </p>

        <div className="space-y-2 text-[11px] font-mono bg-slate-900/60 p-3 rounded-lg border border-slate-800/80">
          <div className="flex justify-between">
            <span className="text-slate-400">Shape</span>
            <span className="text-cyan-300 font-semibold">{tensorMetadata?.shape || "(1, 3, 1024, 1024)"}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-400">DType</span>
            <span className="text-slate-200">{tensorMetadata?.dtype || "float32"}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-400">Min / Max Val</span>
            <span className="text-slate-200">{tensorMetadata?.minMaxVal || "-0.984 / 1.000"}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-400">Mean Act.</span>
            <span className="text-slate-200">{tensorMetadata?.meanAct || "0.1425"}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-400">Mem Usage</span>
            <span className="text-emerald-400 font-semibold">{tensorMetadata?.memUsage || "12.0 MB"}</span>
          </div>
        </div>

        <button className="w-full mt-4 bg-slate-900/60 hover:bg-slate-800/80 border border-slate-800 hover:border-cyan-500/40 transition-colors text-xs text-slate-300 hover:text-cyan-300 rounded-lg py-2 font-mono cursor-pointer">
          View Raw JSON Schema
        </button>
      </div>
    </div>
  );
}

// ---------------------------------------------
// Bottom status bar
// ---------------------------------------------
function StatusBar() {
  return (
    <div className="flex flex-col sm:flex-row items-center justify-between gap-2 py-3 px-4 sm:px-8 border-t border-slate-800/80 text-[10px] font-mono text-slate-400">
      <span className="flex items-center gap-1.5 text-cyan-300">
        <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse" />
        SYS: TRACE_LOG_ACTIVE · 100% COGNITIVE CONSISTENCY
      </span>
      <span>LAT: 34.9522° N · LON: 118.2437° W · ELEV: 412M</span>
    </div>
  );
}

// ---------------------------------------------
// Execution Log Page
// ---------------------------------------------
export default function ExecutionLogPage() {
  const [logData, setLogData] = useState<any>(null);

  useEffect(() => {
    fetch("/api/query/q_9482_a/log")
      .then((res) => res.json())
      .then((d) => setLogData(d))
      .catch((err) => console.error(err));
  }, []);

  return (
    <div className="flex flex-col lg:flex-row min-h-screen bg-[#0a1420] text-white font-sans overflow-x-hidden">
      <Sidebar />

      <div className="flex-1 flex flex-col min-w-0">
        <Header />

        <main className="flex-1 p-4 sm:p-6 lg:p-8 grid grid-cols-1 lg:grid-cols-3 gap-5 items-start overflow-y-auto">
          <div className="lg:col-span-2">
            <PipelineSteps />
          </div>
          <div className="lg:col-span-1">
            <NodeInspector tensorMetadata={logData?.tensorMetadata} />
          </div>
        </main>

        <StatusBar />
      </div>
    </div>
  );
}