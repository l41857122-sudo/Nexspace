"use client";

import {
  Download,
  Layers,
  ZoomIn,
  ZoomOut,
  Plus,
  CheckCircle2,
  Flag,
  XCircle,
  Activity,
  Crosshair,
} from "lucide-react";

import Sidebar from "./Sidebar";

// ---------------------------------------------
// Top bar
// ---------------------------------------------
function TopBar() {
  return (
    <header className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 px-4 sm:px-6 lg:px-8 py-4 border-b border-slate-800/80 bg-[#0d1826]/40 backdrop-blur-md">
      <div className="flex flex-wrap items-center gap-2 sm:gap-3">
        <p className="text-sm font-semibold text-white tracking-tight">Evidence Viewer</p>
        <span className="text-[10px] font-mono text-cyan-400 bg-cyan-500/10 px-2 py-0.5 rounded-full border border-cyan-500/20">
          ID: SQ-2023-X992
        </span>
        <span className="text-[10px] font-mono text-slate-400 bg-slate-800/60 px-2 py-0.5 rounded-full border border-slate-700/60">
          Sensor: Sentinel-2B
        </span>
      </div>
      <button className="self-start sm:self-auto flex items-center gap-1.5 bg-slate-900/60 hover:bg-slate-800/80 border border-slate-800 hover:border-cyan-500/30 transition-all text-xs text-slate-300 hover:text-cyan-300 rounded-lg px-3 py-1.5 font-mono cursor-pointer">
        <Download size={12} />
        <span>Export Evidence</span>
      </button>
    </header>
  );
}

// ---------------------------------------------
// Image viewer with overlay shapes
// ---------------------------------------------
function EvidenceImage() {
  return (
    <div className="w-full flex-1 flex flex-col bg-[#09121d] relative overflow-hidden rounded-xl border border-slate-800/90 min-h-[480px] shadow-[0_4px_24px_rgba(0,0,0,0.3)]">
      {/* HUD Header Bar */}
      <div className="flex items-center justify-between px-3 sm:px-4 py-2 bg-[#09111c] border-b border-slate-800/80 text-[11px] font-mono text-slate-400 z-20">
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
          <span className="text-slate-200 font-semibold uppercase tracking-wider text-[10px]">
            SPECTRAL TARGET LOCALIZATION
          </span>
        </div>
        <button className="flex items-center gap-1.5 bg-slate-900/80 border border-slate-800 hover:border-cyan-500/40 rounded-md px-2 py-1 text-[10px] text-cyan-300 font-mono transition-colors cursor-pointer">
          <Layers size={11} />
          <span>AI Context Layer Active</span>
        </button>
      </div>

      {/* Simulated thermal/satellite base */}
      <div className="flex-1 relative bg-gradient-to-br from-[#0e1d2e] via-[#09121d] to-[#060b12] overflow-hidden p-4 flex flex-col justify-between">
        {/* Background Grid */}
        <div
          className="absolute inset-0 opacity-10 pointer-events-none"
          style={{
            backgroundImage:
              "linear-gradient(rgba(56,189,248,0.2) 1px, transparent 1px), linear-gradient(90deg, rgba(56,189,248,0.2) 1px, transparent 1px)",
            backgroundSize: "32px 32px",
          }}
        />

        {/* Heat dots scattered around */}
        {[
          "top-[15%] left-[20%]",
          "top-[25%] left-[35%]",
          "top-[40%] left-[15%]",
          "top-[55%] left-[30%]",
          "top-[70%] left-[45%]",
          "top-[30%] right-[20%]",
          "top-[60%] right-[15%]",
          "top-[80%] left-[55%]",
        ].map((pos, i) => (
          <span
            key={i}
            className={`absolute w-2 h-2 rounded-full bg-red-500/70 shadow-[0_0_10px_3px_rgba(239,68,68,0.5)] ${pos}`}
          />
        ))}

        {/* Detected polygon (triangle) */}
        <svg
          className="absolute top-[20%] left-[45%] w-28 h-28 text-emerald-400"
          viewBox="0 0 100 100"
        >
          <polygon
            points="50,10 90,70 10,80"
            fill="rgba(52,211,153,0.12)"
            stroke="currentColor"
            strokeWidth="2"
            strokeDasharray="4 2"
          />
        </svg>

        {/* Selected region (dashed square) */}
        <div className="absolute top-[42%] left-[35%] w-28 h-28 border-2 border-dashed border-cyan-400 rounded-lg flex items-center justify-center shadow-[0_0_15px_rgba(6,182,212,0.3)]">
          <Crosshair size={18} className="text-cyan-400 animate-pulse" />
        </div>

        {/* Zoom controls */}
        <div className="absolute bottom-3 right-3 flex items-center gap-1 bg-slate-900/80 p-0.5 rounded-lg border border-slate-800 z-20">
          <button className="p-1 rounded hover:bg-cyan-500/10 text-slate-400 hover:text-cyan-300 transition-colors cursor-pointer">
            <ZoomIn size={13} />
          </button>
          <button className="p-1 rounded hover:bg-cyan-500/10 text-slate-400 hover:text-cyan-300 transition-colors cursor-pointer">
            <ZoomOut size={13} />
          </button>
        </div>

        {/* Coordinates */}
        <div className="relative z-10 text-[10px] font-mono text-slate-400 bg-slate-900/80 px-2.5 py-1 rounded-md border border-slate-800 self-start">
          LAT: 34.9522° N · LON: 118.2437° W · ELEV: 412M
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------
// Spectral signature bar
// ---------------------------------------------
function SpectralBar({
  band,
  value,
  color,
}: {
  band: string;
  value: number;
  color: string;
}) {
  return (
    <div className="mb-3">
      <div className="flex items-center justify-between mb-1">
        <span className="text-xs text-slate-300 font-mono">{band}</span>
        <span className="text-xs text-cyan-400 font-mono font-semibold">
          {value.toFixed(3)}
        </span>
      </div>
      <div className="h-1.5 rounded-full bg-slate-800 overflow-hidden">
        <div
          className={`h-full rounded-full ${color}`}
          style={{ width: `${value * 100}%` }}
        />
      </div>
    </div>
  );
}

// ---------------------------------------------
// Right panel: Spectral signature + verification
// ---------------------------------------------
function EvidencePanel() {
  return (
    <div className="w-full border border-slate-800/90 bg-[#0c1624]/60 backdrop-blur-md rounded-xl p-4 sm:p-5 flex flex-col justify-between shadow-[0_4px_24px_rgba(0,0,0,0.3)] min-h-[480px]">
      <div>
        <p className="text-sm font-semibold text-white tracking-tight mb-4 font-mono">
          Spectral Signature Profile
        </p>

        <div className="space-y-1 bg-slate-900/60 border border-slate-800/80 rounded-xl p-3 mb-4">
          <SpectralBar band="B04 (Red 665nm)" value={0.142} color="bg-red-400" />
          <SpectralBar band="B08 (NIR 842nm)" value={0.875} color="bg-cyan-400" />
          <SpectralBar band="B11 (SWIR 1610nm)" value={0.329} color="bg-amber-400" />
        </div>
      </div>

      <div>
        <p className="text-[10px] font-semibold text-slate-400 mb-3 uppercase tracking-widest font-mono">
          Verification Action
        </p>

        <button className="w-full flex items-center justify-center gap-2 bg-emerald-500 hover:bg-emerald-400 active:scale-95 transition-all text-[#0a1420] text-xs font-bold rounded-lg py-2.5 mb-2 cursor-pointer shadow-[0_0_12px_rgba(52,211,153,0.3)]">
          <CheckCircle2 size={14} />
          <span>Confirm Verification</span>
        </button>

        <div className="flex gap-2">
          <button className="flex-1 flex items-center justify-center gap-1.5 bg-slate-900/60 hover:bg-slate-800/80 border border-slate-800 hover:border-amber-500/40 transition-all text-xs text-slate-300 hover:text-amber-300 rounded-lg py-2 cursor-pointer">
            <Flag size={12} className="text-amber-400" />
            <span>Flag</span>
          </button>
          <button className="flex-1 flex items-center justify-center gap-1.5 bg-slate-900/60 hover:bg-slate-800/80 border border-slate-800 hover:border-red-500/40 transition-all text-xs text-slate-300 hover:text-red-300 rounded-lg py-2 cursor-pointer">
            <XCircle size={12} className="text-red-400" />
            <span>Reject</span>
          </button>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------
// Evidence Viewer Page
// ---------------------------------------------
export default function EvidenceViewerPage() {
  return (
    <div className="flex flex-col lg:flex-row min-h-screen bg-[#0a1420] text-white font-sans overflow-x-hidden">
      <Sidebar />

      <div className="flex-1 flex flex-col min-w-0">
        <TopBar />

        <main className="flex-1 p-4 sm:p-6 lg:p-8 grid grid-cols-1 lg:grid-cols-3 gap-5 items-start overflow-y-auto">
          <div className="lg:col-span-2">
            <EvidenceImage />
          </div>
          <div className="lg:col-span-1">
            <EvidencePanel />
          </div>
        </main>
      </div>
    </div>
  );
}