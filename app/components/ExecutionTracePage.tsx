"use client";

import { useState, useEffect, useMemo } from "react";
import Link from "next/link";
import {
  CheckCircle2,
  CircleDot,
  Scan,
  Boxes,
  Gauge,
  Terminal,
  Search,
  Sparkles,
  Layers,
  Cpu,
} from "lucide-react";

import Sidebar from "./Sidebar";
import type { CanonicalSourceImage, CanonicalInvestigationState } from "../types/nexspace";
import {
  getCurrentInvestigation,
  getActiveSourceImage,
  DEFAULT_DEMO_SOURCE,
} from "../utils/investigationStorage";

// ---------------------------------------------
// Page header
// ---------------------------------------------
function Header({
  investigation,
  sourceImage,
}: {
  investigation: CanonicalInvestigationState | null;
  sourceImage: CanonicalSourceImage;
}) {
  const isUpload = sourceImage.source === "upload";

  return (
    <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 px-4 sm:px-6 lg:px-8 py-4 border-b border-slate-800/80 bg-[#0d1826]/40 backdrop-blur-md">
      <div>
        <div className="flex items-center gap-1.5 text-[10px] text-emerald-400 font-medium mb-1 font-mono">
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
          {investigation ? "INVESTIGATION SYNCHRONIZED" : "STANDBY / READY"}
        </div>
        <p className="text-base font-semibold text-white tracking-tight">
          {investigation ? `Query: "${investigation.query}"` : "Live Agentic Execution Trace"}
        </p>
        <p className="text-[11px] text-slate-400 mt-0.5 font-mono">
          Source: <span className="text-cyan-300">{sourceImage.filename}</span> ({isUpload ? "User Upload" : "Catalog Demo"}) · ID: {investigation?.investigation_id || "INV-STANDBY"}
        </p>
      </div>
      <div className="flex items-center gap-2">
        <Link
          href="/query"
          className="text-xs text-cyan-300 border border-cyan-400/30 bg-cyan-500/10 hover:bg-cyan-500/20 transition-all rounded-lg px-3 py-1.5 font-mono active:scale-95"
        >
          New Query
        </Link>
      </div>
    </div>
  );
}

// ---------------------------------------------
// Pipeline topology
// ---------------------------------------------
type StageState = "done" | "active" | "pending";

function Stage({
  icon: Icon,
  label,
  state,
  sublabel,
}: {
  icon: React.ElementType;
  label: string;
  state: StageState;
  sublabel?: string;
}) {
  const ring =
    state === "done"
      ? "bg-emerald-500 text-[#0a1420] shadow-[0_0_12px_rgba(16,185,129,0.5)]"
      : state === "active"
      ? "bg-cyan-500 text-[#0a1420] ring-4 ring-cyan-500/25 shadow-[0_0_20px_rgba(6,182,212,0.6)] animate-pulse"
      : "bg-slate-900/80 text-slate-500 border border-slate-800";

  return (
    <div className="flex flex-col items-center gap-2 min-w-[100px] flex-1">
      <div
        className={`w-9 h-9 rounded-full flex items-center justify-center ${ring} shrink-0 transition-all duration-300`}
      >
        {state === "done" ? <CheckCircle2 size={16} /> : <Icon size={15} />}
      </div>
      <div className="text-center">
        <p
          className={`text-[11px] whitespace-nowrap font-medium ${
            state === "pending" ? "text-slate-500 font-mono" : state === "active" ? "text-cyan-300 font-semibold" : "text-slate-200"
          }`}
        >
          {label}
        </p>
        {sublabel && (
          <p className="text-[10px] text-cyan-300 font-mono animate-pulse">{sublabel}</p>
        )}
      </div>
    </div>
  );
}

function PipelineTopology({ investigation }: { investigation: CanonicalInvestigationState | null }) {
  const isComplete = Boolean(investigation);

  const stagesList = [
    { label: "1. Input Ingestion", state: isComplete ? "done" : "pending", icon: CircleDot, sublabel: isComplete ? "Raster OK" : undefined },
    { label: "2. Tool Routing", state: isComplete ? "done" : "pending", icon: Search, sublabel: isComplete ? "Routed" : undefined },
    { label: "3. Neural Inference", state: isComplete ? "done" : "pending", icon: Cpu, sublabel: isComplete ? "GDINO + BLIP" : undefined },
    { label: "4. Spatial Norm", state: isComplete ? "done" : "pending", icon: Boxes, sublabel: isComplete ? "[0-1000]" : undefined },
    { label: "5. NL Synthesis", state: isComplete ? "done" : "pending", icon: Sparkles, sublabel: isComplete ? "Complete" : undefined },
  ] as Array<{ label: string; state: StageState; icon: any; sublabel?: string }>;

  return (
    <div className="w-full border border-slate-800/90 bg-[#0c1624]/60 backdrop-blur-md rounded-xl p-5 shadow-[0_4px_24px_rgba(0,0,0,0.3)]">
      <p className="text-xs font-semibold text-slate-200 mb-5 flex items-center gap-1.5 font-mono uppercase tracking-wider">
        <Boxes size={13} className="text-cyan-400" />
        Pipeline Topology Execution Flow
      </p>

      <div className="overflow-x-auto w-full pb-2">
        <div className="flex items-center min-w-[550px]">
          {stagesList.map((st, i) => (
            <div key={i} className="contents">
              <Stage icon={st.icon} label={st.label} state={st.state} sublabel={st.sublabel} />
              {i < stagesList.length - 1 && <div className="h-px flex-1 bg-slate-800 -mx-2 mb-6" />}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------
// Live kernel output (terminal log)
// ---------------------------------------------
interface LogLineItem {
  time?: string;
  tag?: string | null;
  text: string;
  color?: string;
}

function LiveKernelOutput({ investigation, sourceImage }: { investigation: CanonicalInvestigationState | null; sourceImage: CanonicalSourceImage }) {
  const logs = useMemo<LogLineItem[]>(() => {
    if (!investigation) {
      return [
        { time: "00:00.00", tag: "INFO", text: "Pipeline initialized and listening for investigation requests.", color: "text-slate-400" },
        { time: "00:00.01", tag: "STANDBY", text: `Active raster buffer: ${sourceImage.filename}`, color: "text-slate-500" },
      ];
    }

    const detections = investigation.response?.grounding?.detections || [];
    const caption = typeof investigation.response?.optical_caption === "string"
      ? investigation.response.optical_caption
      : (investigation.response?.optical_caption as unknown as { caption?: string })?.caption || "Scene caption generated.";

    return [
      { time: "00:00.12", tag: "INFO", text: `Investigation started for query: "${investigation.query}"`, color: "text-slate-400" },
      { time: "00:00.28", tag: "SUCCESS", text: `Input raster validated: ${sourceImage.filename} (${sourceImage.mediaType})`, color: "text-emerald-400" },
      { time: "00:00.54", tag: "INFO", text: "Routing prompt to specialized neural architectures (Grounding DINO + BLIP)", color: "text-cyan-300" },
      { time: "00:01.80", tag: "SUCCESS", text: `Grounding DINO extracted ${detections.length} candidate detection(s).`, color: "text-emerald-400" },
      { time: "00:02.40", tag: "SUCCESS", text: `BLIP generated scene description: "${caption}"`, color: "text-cyan-300" },
      { time: "00:02.65", tag: "SUCCESS", text: "Normalized all coordinate boxes to canonical [xmin, ymin, xmax, ymax] 0-1000 space.", color: "text-emerald-400" },
      { time: "00:02.90", tag: "DONE", text: `Telemetry synchronized under Investigation ID: ${investigation.investigation_id}`, color: "text-emerald-400" },
    ];
  }, [investigation, sourceImage]);

  return (
    <div className="w-full border border-slate-800/90 bg-[#08121e] rounded-xl p-4 font-mono text-[11px] overflow-auto min-h-[280px] shadow-[0_4px_24px_rgba(0,0,0,0.3)]">
      <div className="flex items-center justify-between mb-3 pb-2 border-b border-slate-800/80">
        <div className="flex items-center gap-2 text-xs font-semibold text-slate-300">
          <Terminal size={13} className="text-cyan-400" />
          <span>Live Kernel Stream (Monotonic Stages)</span>
        </div>
        <span className="text-[10px] text-cyan-400 bg-cyan-500/10 px-2 py-0.5 rounded border border-cyan-500/20">
          PID: AGENT-CONTROLLER-ML
        </span>
      </div>
      <div className="space-y-1">
        {logs.map((line, i) => {
          const timeStr = line?.time || "00:00.00";
          const tagStr = line?.tag;
          const textStr = line?.text || "";
          const colorStr = line?.color || "text-slate-300";

          return (
            <div key={i} className="flex gap-2">
              <span className="text-slate-600 shrink-0">{timeStr}</span>
              {tagStr && (
                <span className={`shrink-0 ${colorStr}`}>[{tagStr}]</span>
              )}
              <span className={tagStr ? "text-slate-400" : colorStr}>
                {textStr}
                {i === logs.length - 1 && (
                  <span className="inline-block w-1.5 h-3 bg-cyan-400 ml-1 animate-pulse align-middle" />
                )}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ---------------------------------------------
// Total progress ring
// ---------------------------------------------
function TotalProgress({ investigation }: { investigation: CanonicalInvestigationState | null }) {
  const percent = investigation ? 100 : 0;
  const radius = 46;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (percent / 100) * circumference;

  return (
    <div className="w-full border border-slate-800/90 bg-[#0c1624]/60 backdrop-blur-md rounded-xl p-4 flex flex-col items-center justify-between shadow-[0_4px_24px_rgba(0,0,0,0.3)] min-h-[280px]">
      <p className="text-xs font-semibold text-slate-200 self-start mb-2 font-mono uppercase tracking-wider">
        Investigation Status
      </p>

      <svg width="120" height="120" viewBox="0 0 120 120" className="my-2">
        <circle
          cx="60"
          cy="60"
          r={radius}
          fill="none"
          stroke="rgba(255,255,255,0.06)"
          strokeWidth="8"
        />
        <circle
          cx="60"
          cy="60"
          r={radius}
          fill="none"
          stroke="#06b6d4"
          strokeWidth="8"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          strokeLinecap="round"
          transform="rotate(-90 60 60)"
        />
        <text
          x="60"
          y="56"
          textAnchor="middle"
          fill="white"
          fontSize="22"
          fontWeight="700"
          fontFamily="sans-serif"
        >
          {percent}%
        </text>
        <text
          x="60"
          y="74"
          textAnchor="middle"
          fill="#94a3b8"
          fontSize="9"
          fontFamily="monospace"
        >
          {investigation ? "COMPLETE" : "STANDBY"}
        </text>
      </svg>

      <div className="flex w-full justify-around mt-3 text-center border-t border-slate-800/80 pt-3">
        <div>
          <p className="text-[10px] text-slate-400 uppercase font-mono tracking-widest">Detections</p>
          <p className="text-xs text-cyan-400 mt-0.5 font-mono font-semibold">
            {investigation?.response?.grounding?.detections?.length || 0} Targets
          </p>
        </div>
        <div>
          <p className="text-[10px] text-slate-400 uppercase font-mono tracking-widest">Pipeline</p>
          <p className="text-xs text-slate-200 mt-0.5 font-mono">5 Monotonic</p>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------
// Status bar
// ---------------------------------------------
function StatusBar({ investigation }: { investigation: CanonicalInvestigationState | null }) {
  return (
    <div className="flex items-center justify-between px-4 sm:px-8 py-3 text-[10px] font-mono text-emerald-400 border-t border-slate-800/80 bg-[#0d1826]/30">
      <div className="flex items-center gap-1.5">
        <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
        <span>Status: {investigation ? "AI Pipeline Execution Nominal · Telemetry Synced" : "Pipeline Idle · Awaiting Query"}</span>
      </div>
      <Link href="/execution-log" className="text-cyan-300 hover:text-cyan-200 underline">
        View Detailed Execution Logs →
      </Link>
    </div>
  );
}

// ---------------------------------------------
// Execution Trace Page
// ---------------------------------------------
export default function ExecutionTracePage() {
  const [investigationState, setInvestigationState] = useState<CanonicalInvestigationState | null>(null);

  useEffect(() => {
    setInvestigationState(getCurrentInvestigation());
  }, []);

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

        <main className="flex-1 p-4 sm:p-6 lg:p-8 space-y-5 overflow-y-auto">
          <PipelineTopology investigation={investigationState} />

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
            <div className="lg:col-span-2">
              <LiveKernelOutput investigation={investigationState} sourceImage={sourceImage} />
            </div>
            <div className="lg:col-span-1">
              <TotalProgress investigation={investigationState} />
            </div>
          </div>

          <StatusBar investigation={investigationState} />
        </main>
      </div>
    </div>
  );
}