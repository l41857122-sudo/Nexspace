"use client";

import { useState, useEffect } from "react";
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
  Info,
  ShieldCheck
} from "lucide-react";

import Sidebar from "./Sidebar";

// ---------------------------------------------
// Top bar
// ---------------------------------------------
function TopBar({ evidence }: { evidence?: any }) {
  return (
    <header className="flex flex-col gap-3 px-4 sm:px-6 lg:px-8 py-4 border-b border-slate-800/80 bg-[#0d1826]/40 backdrop-blur-md">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div className="flex flex-wrap items-center gap-2 sm:gap-3">
          <p className="text-sm font-semibold text-white tracking-tight">Evidence Viewer &amp; Verification Audit</p>
          <span className="text-[10px] font-mono text-cyan-400 bg-cyan-500/10 px-2 py-0.5 rounded-full border border-cyan-500/20">
            Target ID: {evidence?.evidenceId || "SQ-2023-X992"}
          </span>
          <span className="text-[10px] font-mono text-slate-400 bg-slate-800/60 px-2 py-0.5 rounded-full border border-slate-700/60">
            Sensor: {evidence?.sensor || "Sentinel-2B (MSI 10m)"}
          </span>
        </div>
        <button
          onClick={() => window.open("/api/reports/SQ-REP-2023-11A/pdf", "_blank")}
          className="self-start sm:self-auto flex items-center gap-1.5 bg-slate-900/60 hover:bg-slate-800/80 border border-slate-800 hover:border-cyan-500/30 transition-all text-xs text-slate-300 hover:text-cyan-300 rounded-lg px-3 py-1.5 font-mono cursor-pointer"
        >
          <Download size={12} />
          <span>Export Evidence PDF</span>
        </button>
      </div>

      {/* Clear Purpose Banner */}
      <div className="flex items-center gap-2 bg-cyan-500/10 border border-cyan-500/25 rounded-lg px-3 py-2 text-[11px] text-cyan-300 font-mono">
        <Info size={14} className="text-cyan-400 shrink-0" />
        <span>
          <strong>Purpose:</strong> High-resolution sub-meter satellite target localization &amp; multi-spectral verification audit. Allows human analysts to inspect target satellite cutouts, analyze Red/NIR/SWIR reflectance curves, and issue official verifications (Confirm / Flag / Reject) prior to intelligence reporting.
        </span>
      </div>
    </header>
  );
}

// ---------------------------------------------
// Image viewer with real satellite crop & interactive zoom
// ---------------------------------------------
function EvidenceImage({ evidence }: { evidence?: any }) {
  const [zoom, setZoom] = useState(1);

  return (
    <div className="w-full flex-1 flex flex-col bg-[#09121d] relative overflow-hidden rounded-xl border border-slate-800/90 min-h-[480px] shadow-[0_4px_24px_rgba(0,0,0,0.3)]">
      {/* HUD Header Bar */}
      <div className="flex items-center justify-between px-3 sm:px-4 py-2 bg-[#09111c] border-b border-slate-800/80 text-[11px] font-mono text-slate-400 z-20">
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
          <span className="text-slate-200 font-semibold uppercase tracking-wider text-[10px]">
            SPECTRAL TARGET LOCALIZATION (SUB-METER CROP)
          </span>
        </div>
        <div className="flex items-center gap-2">
          <button className="flex items-center gap-1.5 bg-slate-900/80 border border-slate-800 rounded-md px-2 py-1 text-[10px] text-cyan-300 font-mono">
            <Layers size={11} />
            <span>Target: Panamax Container Vessel</span>
          </button>
        </div>
      </div>

      {/* Main High-Res Satellite Viewport */}
      <div className="flex-1 relative bg-[#09121d] overflow-hidden p-4 flex flex-col justify-between">
        {/* Real Satellite Imagery Cutout Background */}
        <div
          className="absolute inset-0 bg-cover bg-center transition-transform duration-300 ease-out contrast-125 brightness-95"
          style={{
            backgroundImage:
              "url('https://images.unsplash.com/photo-1541185933-ef5d8ed016c2?q=80&w=1600&auto=format&fit=crop')",
            transform: `scale(${zoom})`,
          }}
        />
        <div className="absolute inset-0 bg-gradient-to-t from-[#09121d]/80 via-transparent to-[#09121d]/60 pointer-events-none" />

        {/* Background Coordinate Grid Overlay */}
        <div
          className="absolute inset-0 opacity-15 pointer-events-none z-10"
          style={{
            backgroundImage:
              "linear-gradient(rgba(56,189,248,0.3) 1px, transparent 1px), linear-gradient(90deg, rgba(56,189,248,0.3) 1px, transparent 1px)",
            backgroundSize: "32px 32px",
          }}
        />

        {/* Bounding Box Bounding Target */}
        <div className="absolute top-[32%] left-[30%] w-48 h-36 border-2 border-cyan-400 bg-cyan-500/15 rounded-lg flex flex-col justify-between p-2 shadow-[0_0_20px_rgba(6,182,212,0.5)] z-15 pointer-events-none">
          <div className="flex justify-between items-center">
            <span className="text-[9px] font-mono font-bold bg-slate-900/90 text-cyan-300 px-1.5 py-0.5 rounded border border-cyan-500/30">
              TARGET BBOX #992
            </span>
            <span className="text-[9px] font-mono text-emerald-400 bg-emerald-500/20 px-1.5 py-0.5 rounded font-semibold">
              94% CONF
            </span>
          </div>
          <div className="self-center">
            <Crosshair size={24} className="text-cyan-400 animate-pulse" />
          </div>
          <span className="text-[8px] font-mono text-slate-300 bg-slate-900/90 px-1.5 py-0.5 rounded self-start">
            294m × 32m · Panamax Class
          </span>
        </div>

        {/* Interactive Zoom Controls */}
        <div className="absolute bottom-3 right-3 flex items-center gap-1 bg-slate-900/90 p-1 rounded-lg border border-slate-800 z-20 font-mono text-[10px]">
          <button
            onClick={() => setZoom((z) => Math.min(Number((z + 0.25).toFixed(2)), 2.5))}
            className="p-1.5 rounded hover:bg-cyan-500/10 text-slate-300 hover:text-cyan-300 transition-colors cursor-pointer"
            title="Zoom In"
          >
            <ZoomIn size={14} />
          </button>
          <span className="text-cyan-400 font-semibold px-1">{zoom.toFixed(2)}×</span>
          <button
            onClick={() => setZoom((z) => Math.max(Number((z - 0.25).toFixed(2)), 0.8))}
            className="p-1.5 rounded hover:bg-cyan-500/10 text-slate-300 hover:text-cyan-300 transition-colors cursor-pointer"
            title="Zoom Out"
          >
            <ZoomOut size={14} />
          </button>
        </div>

        {/* Coordinates */}
        <div className="relative z-20 text-[10px] font-mono text-slate-300 bg-slate-900/90 px-3 py-1 rounded-md border border-slate-800 self-start shadow">
          {evidence?.coordinates || "LAT: 34.9522° N · LON: 118.2437° W · ELEV: 412M MSL"}
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
function EvidencePanel({ evidence }: { evidence?: any }) {
  const [status, setStatus] = useState<string>("VERIFIED CONFIRMED");

  const handleAction = async (action: string) => {
    try {
      const res = await fetch("/api/entities/entity-1/evidence", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action })
      });
      if (res.ok) setStatus(action.toUpperCase());
    } catch (e) {
      console.error(e);
    }
  };

  const bands = evidence?.spectralBands || [
    { band: "B04 (Red 665nm)", value: 0.142, color: "bg-red-400" },
    { band: "B08 (NIR 842nm)", value: 0.875, color: "bg-cyan-400" },
    { band: "B11 (SWIR 1610nm)", value: 0.329, color: "bg-amber-400" }
  ];

  return (
    <div className="w-full border border-slate-800/90 bg-[#0c1624]/60 backdrop-blur-md rounded-xl p-4 sm:p-5 flex flex-col justify-between shadow-[0_4px_24px_rgba(0,0,0,0.3)] min-h-[480px]">
      <div>
        <div className="flex items-center justify-between mb-4">
          <p className="text-sm font-semibold text-white tracking-tight font-mono">
            Spectral Signature Profile
          </p>
          <span className="text-[10px] font-mono px-2.5 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-bold">
            {status}
          </span>
        </div>

        <div className="space-y-1 bg-slate-900/60 border border-slate-800/80 rounded-xl p-3 mb-4">
          {bands.map((b: any, i: number) => (
            <SpectralBar key={i} band={b.band} value={b.value} color={b.color} />
          ))}
        </div>

        <div className="bg-slate-900/40 border border-slate-800 p-3 rounded-lg text-[11px] font-mono text-slate-300 space-y-1">
          <div className="flex justify-between">
            <span className="text-slate-400">Classification:</span>
            <span className="text-cyan-300">Vessel_Panamax_01</span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-400">Spectral Match:</span>
            <span className="text-emerald-400">98.4% (Nominal)</span>
          </div>
        </div>
      </div>

      <div>
        <p className="text-[10px] font-semibold text-slate-400 mb-3 uppercase tracking-widest font-mono">
          Verification Action &amp; Sign-off
        </p>

        <button
          onClick={() => handleAction("confirm")}
          className="w-full flex items-center justify-center gap-2 bg-emerald-500 hover:bg-emerald-400 active:scale-95 transition-all text-[#0a1420] text-xs font-bold rounded-lg py-2.5 mb-2 cursor-pointer shadow-[0_0_12px_rgba(52,211,153,0.3)]"
        >
          <CheckCircle2 size={14} />
          <span>Confirm Verification</span>
        </button>

        <div className="flex gap-2">
          <button
            onClick={() => handleAction("flag")}
            className="flex-1 flex items-center justify-center gap-1.5 bg-slate-900/60 hover:bg-slate-800/80 border border-slate-800 hover:border-amber-500/40 transition-all text-xs text-slate-300 hover:text-amber-300 rounded-lg py-2 cursor-pointer"
          >
            <Flag size={12} className="text-amber-400" />
            <span>Flag</span>
          </button>
          <button
            onClick={() => handleAction("reject")}
            className="flex-1 flex items-center justify-center gap-1.5 bg-slate-900/60 hover:bg-slate-800/80 border border-slate-800 hover:border-red-500/40 transition-all text-xs text-slate-300 hover:text-red-300 rounded-lg py-2 cursor-pointer"
          >
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
  const [evidence, setEvidence] = useState<any>(null);

  useEffect(() => {
    fetch("/api/entities/entity-1/evidence")
      .then((res) => res.json())
      .then((data) => setEvidence(data))
      .catch((err) => console.error(err));
  }, []);

  return (
    <div className="flex flex-col lg:flex-row min-h-screen bg-[#0a1420] text-white font-sans overflow-x-hidden">
      <Sidebar />

      <div className="flex-1 flex flex-col min-w-0">
        <TopBar evidence={evidence} />

        <main className="flex-1 p-4 sm:p-6 lg:p-8 grid grid-cols-1 lg:grid-cols-3 gap-5 items-start overflow-y-auto">
          <div className="lg:col-span-2">
            <EvidenceImage evidence={evidence} />
          </div>
          <div className="lg:col-span-1">
            <EvidencePanel evidence={evidence} />
          </div>
        </main>
      </div>
    </div>
  );
}