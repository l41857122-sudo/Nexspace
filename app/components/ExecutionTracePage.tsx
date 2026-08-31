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
  CheckCircle2,
  CircleDot,
  Scan,
  Boxes,
  Gauge,
  Activity,
  Terminal,
} from "lucide-react";

import Sidebar from "./Sidebar";

// ---------------------------------------------
// Page header
// ---------------------------------------------
function Header({ data }: { data?: any }) {
  return (
    <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 px-4 sm:px-6 lg:px-8 py-4 border-b border-slate-800/80 bg-[#0d1826]/40 backdrop-blur-md">
      <div>
        <div className="flex items-center gap-1.5 text-[10px] text-emerald-400 font-medium mb-1 font-mono">
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
          {data?.status ? `STATUS: ${data.status}` : "ACTIVE OPERATION"}
        </div>
        <p className="text-base font-semibold text-white tracking-tight">
          {data?.operationName || "Geospatial Feature Extraction"}
        </p>
        <p className="text-[11px] text-slate-400 mt-0.5 font-mono">
          Target Region: {data?.targetRegion || "Quadrant 7A, Sector North"}
        </p>
      </div>
      <button className="self-start sm:self-auto text-xs text-red-300 border border-red-400/30 bg-red-500/10 hover:bg-red-500/20 transition-all rounded-lg px-3 py-1.5 font-mono active:scale-95 cursor-pointer">
        Abort Operation
      </button>
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

function PipelineTopology({ stages }: { stages?: any[] }) {
  const defaultStages: Array<{ label: string; state: StageState; icon: any; sublabel?: string }> = [
    { label: "Data Ingestion", state: "done", icon: CircleDot },
    { label: "Radiometric Correction", state: "done", icon: CircleDot },
    { label: "Neural Extraction", state: "active", sublabel: "74%", icon: Scan },
    { label: "Spatial Clustering", state: "pending", icon: Boxes },
    { label: "Confidence Scoring", state: "pending", icon: Gauge }
  ];

  const list: Array<{ label: string; state: StageState; icon: any; sublabel?: string }> = stages && stages.length > 0
    ? stages.map((s, idx) => ({
        label: s.name,
        state: (s.state === "done" || s.state === "active" || s.state === "pending" ? s.state : "pending") as StageState,
        sublabel: s.progressPct ? `${s.progressPct}%` : undefined,
        icon: idx === 2 ? Scan : idx === 3 ? Boxes : idx === 4 ? Gauge : CircleDot
      }))
    : defaultStages;

  return (
    <div className="w-full border border-slate-800/90 bg-[#0c1624]/60 backdrop-blur-md rounded-xl p-5 shadow-[0_4px_24px_rgba(0,0,0,0.3)]">
      <p className="text-xs font-semibold text-slate-200 mb-5 flex items-center gap-1.5 font-mono uppercase tracking-wider">
        <Boxes size={13} className="text-cyan-400" />
        Pipeline Topology Execution Flow
      </p>

      <div className="overflow-x-auto w-full pb-2">
        <div className="flex items-center min-w-[550px]">
          {list.map((st, i) => (
            <div key={i} className="contents">
              <Stage icon={st.icon} label={st.label} state={st.state} sublabel={st.sublabel} />
              {i < list.length - 1 && <div className="h-px flex-1 bg-slate-800 -mx-2 mb-6" />}
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
function LiveKernelOutput({ logData }: { logData?: any }) {
  const initialLogs = logData?.logLines || [
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
    { time: "14:03:09", tag: null, text: "Processing batch 49/64 ...", color: "text-slate-500" },
  ];

  const [logs, setLogs] = useState<any[]>(initialLogs);

  useEffect(() => {
    const extraBatches = [
      { time: "14:03:15", tag: null, text: "Processing batch 54/64 ...", color: "text-slate-500" },
      { time: "14:03:22", tag: null, text: "Processing batch 61/64 ...", color: "text-slate-500" },
      { time: "14:03:28", tag: "SUCCESS", text: "Feature extraction tensor generated. Mean precision: 94.2%", color: "text-emerald-400" },
      { time: "14:03:30", tag: "INFO", text: "Commencing spatial clustering & vector embedding indexing...", color: "text-cyan-400" }
    ];

    let index = 0;
    const timer = setInterval(() => {
      if (index < extraBatches.length) {
        setLogs((prev) => [...prev, extraBatches[index]]);
        index++;
      } else {
        clearInterval(timer);
      }
    }, 1800);

    return () => clearInterval(timer);
  }, []);

  return (
    <div className="w-full border border-slate-800/90 bg-[#08121e] rounded-xl p-4 font-mono text-[11px] overflow-auto min-h-[280px] shadow-[0_4px_24px_rgba(0,0,0,0.3)]">
      <div className="flex items-center justify-between mb-3 pb-2 border-b border-slate-800/80">
        <div className="flex items-center gap-2 text-xs font-semibold text-slate-300">
          <Terminal size={13} className="text-cyan-400" />
          <span>Live Kernel Stream (Active Stream)</span>
        </div>
        <span className="text-[10px] text-cyan-400 bg-cyan-500/10 px-2 py-0.5 rounded border border-cyan-500/20">
          PID: {logData?.pid || "9021-CUDA"}
        </span>
      </div>
      <div className="space-y-1">
        {logs.map((line: any, i: number) => (
          <div key={i} className="flex gap-2">
            <span className="text-slate-600 shrink-0">{line.time}</span>
            {line.tag && (
              <span className={`shrink-0 ${line.color}`}>[{line.tag}]</span>
            )}
            <span className={line.tag ? "text-slate-400" : line.color}>
              {line.text}
              {i === logs.length - 1 && (
                <span className="inline-block w-1.5 h-3 bg-cyan-400 ml-1 animate-pulse align-middle" />
              )}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ---------------------------------------------
// Total progress ring
// ---------------------------------------------
function TotalProgress({ traceData }: { traceData?: any }) {
  const percent = traceData?.totalProgressPct ?? 75;
  const radius = 46;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (percent / 100) * circumference;

  return (
    <div className="w-full border border-slate-800/90 bg-[#0c1624]/60 backdrop-blur-md rounded-xl p-4 flex flex-col items-center justify-between shadow-[0_4px_24px_rgba(0,0,0,0.3)] min-h-[280px]">
      <p className="text-xs font-semibold text-slate-200 self-start mb-2 font-mono uppercase tracking-wider">
        Total Progress
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
          ETA: {traceData?.eta || "02m 34s"}
        </text>
      </svg>

      <div className="flex w-full justify-around mt-3 text-center border-t border-slate-800/80 pt-3">
        <div>
          <p className="text-[10px] text-slate-400 uppercase font-mono tracking-widest">Throughput</p>
          <p className="text-xs text-cyan-400 mt-0.5 font-mono font-semibold">{traceData?.throughput || "4.2 GB/s"}</p>
        </div>
        <div>
          <p className="text-[10px] text-slate-400 uppercase font-mono tracking-widest">Active Nodes</p>
          <p className="text-xs text-slate-200 mt-0.5 font-mono">{traceData?.activeNodes || "2 Compute"}</p>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------
// Status bar
// ---------------------------------------------
function StatusBar() {
  return (
    <div className="flex items-center gap-1.5 px-1 pt-3 text-[10px] font-mono text-emerald-400 border-t border-slate-800/80">
      <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
      <span>Status: AI Pipeline Active · Neural Model Converged</span>
    </div>
  );
}

// ---------------------------------------------
// Execution Trace Page
// ---------------------------------------------
export default function ExecutionTracePage() {
  const [traceData, setTraceData] = useState<any>(null);
  const [logData, setLogData] = useState<any>(null);

  useEffect(() => {
    fetch("/api/query/q_9482_a/trace")
      .then((res) => res.json())
      .then((d) => setTraceData(d))
      .catch((err) => console.error(err));

    fetch("/api/query/q_9482_a/log")
      .then((res) => res.json())
      .then((d) => setLogData(d))
      .catch((err) => console.error(err));
  }, []);

  return (
    <div className="flex flex-col lg:flex-row min-h-screen bg-[#0a1420] text-white font-sans overflow-x-hidden">
      <Sidebar />

      <div className="flex-1 flex flex-col min-w-0">
        <Header data={traceData} />

        <main className="flex-1 p-4 sm:p-6 lg:p-8 space-y-5 overflow-y-auto">
          <PipelineTopology stages={traceData?.stages} />

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
            <div className="lg:col-span-2">
              <LiveKernelOutput logData={logData} />
            </div>
            <div className="lg:col-span-1">
              <TotalProgress traceData={traceData} />
            </div>
          </div>

          <StatusBar />
        </main>
      </div>
    </div>
  );
}