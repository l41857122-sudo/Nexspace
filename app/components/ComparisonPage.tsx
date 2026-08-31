"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Bell,
  UserCircle,
  Sparkles,
  MapPin,
  Download,
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Sliders,
  Layers,
  Info,
  CheckCircle2,
  Loader2,
  Scan,
  Crosshair,
  Activity,
  Maximize2,
  Eye,
} from "lucide-react";
import Sidebar from "./Sidebar";

// ---------------------------------------------
// Top bar
// ---------------------------------------------
function TopBar() {
  return (
    <header className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 px-4 sm:px-6 lg:px-8 py-4 border-b border-slate-800/80 bg-[#0d1826]/40 backdrop-blur-md">
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-2">
          <div className="p-1.5 rounded-lg bg-cyan-500/10 border border-cyan-500/20 text-cyan-400">
            <Scan size={15} />
          </div>
          <div>
            <h1 className="text-sm font-semibold text-white tracking-tight">
              Temporal Analysis
            </h1>
            <p className="text-[10px] text-slate-400 font-mono">
              Bi-temporal multi-spectral comparison · ID: B492-XT-P
            </p>
          </div>
        </div>
        <span className="text-[10px] font-mono text-cyan-400 bg-cyan-500/10 px-2 py-0.5 rounded-full border border-cyan-500/20 hidden md:inline-block">
          DELTA Δ 24.8%
        </span>
      </div>

      <div className="flex items-center gap-2 self-end sm:self-auto">
        <div className="flex items-center gap-1.5 text-[11px] font-mono text-emerald-400 bg-emerald-500/10 border border-emerald-500/20 px-2.5 py-1 rounded-lg">
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
          <span>Coregistered (RMS &lt; 0.12px)</span>
        </div>
        <button
          className="p-2 rounded-lg bg-slate-900/60 border border-slate-800 text-slate-400 hover:text-cyan-300 hover:border-cyan-500/30 transition-all active:scale-95"
          aria-label="Notifications"
        >
          <Bell size={14} />
        </button>
        <button
          className="p-1 rounded-lg hover:bg-white/5 transition-colors"
          aria-label="User Profile"
        >
          <UserCircle size={22} className="text-slate-400 hover:text-slate-200" />
        </button>
      </div>
    </header>
  );
}

// ---------------------------------------------
// Before / After image viewer with interactive zoom & overlays
// ---------------------------------------------
function ImageViewer({
  opacity,
  deltaHeatmap,
  ndviOverlay,
  activeAnomaly,
  hoveredAnomaly,
  setActiveAnomaly,
  zoom,
  setZoom,
}: {
  opacity: number;
  deltaHeatmap: boolean;
  ndviOverlay: boolean;
  activeAnomaly: string | null;
  hoveredAnomaly: string | null;
  setActiveAnomaly: (v: string | null) => void;
  zoom: number;
  setZoom: (z: number | ((prev: number) => number)) => void;
}) {
  return (
    <div className="w-full flex flex-col bg-[#0b1624] relative overflow-hidden rounded-xl border border-slate-800/90 shadow-[0_4px_30px_rgba(0,0,0,0.5)]">
      {/* HUD Header Bar */}
      <div className="flex items-center justify-between px-3 sm:px-4 py-2 bg-[#09111c] border-b border-slate-800/80 text-[11px] font-mono text-slate-400 z-20">
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
          <span className="text-slate-200 font-semibold uppercase tracking-wider text-[10px]">
            DUAL ORBITAL PASS SYNCHRONIZED
          </span>
        </div>

        {/* Zoom Controls */}
        <div className="flex items-center gap-1 bg-slate-900/80 border border-slate-800 rounded-lg p-0.5 shadow-inner">
          <button
            onClick={() => setZoom((z) => Math.min(Number((z + 0.25).toFixed(2)), 2.5))}
            aria-label="Zoom in"
            className="p-1 hover:bg-cyan-500/10 hover:text-cyan-300 rounded text-slate-400 transition-colors"
            title="Zoom In"
          >
            <ZoomIn size={13} />
          </button>
          <span className="text-[10px] font-mono text-cyan-400 px-1.5 font-semibold min-w-[36px] text-center">
            {zoom.toFixed(2)}×
          </span>
          <button
            onClick={() => setZoom((z) => Math.max(Number((z - 0.25).toFixed(2)), 0.75))}
            aria-label="Zoom out"
            className="p-1 hover:bg-cyan-500/10 hover:text-cyan-300 rounded text-slate-400 transition-colors"
            title="Zoom Out"
          >
            <ZoomOut size={13} />
          </button>
          <button
            onClick={() => setZoom(1)}
            aria-label="Reset zoom"
            className="p-1 hover:bg-cyan-500/10 hover:text-cyan-300 rounded text-slate-500 hover:text-slate-200 border-l border-slate-800 ml-0.5 transition-colors"
            title="Reset to 1.0×"
          >
            <RotateCcw size={11} />
          </button>
        </div>
      </div>

      {/* Main Split Screen Area */}
      <div className="relative flex-1 overflow-hidden min-h-[360px] sm:min-h-[440px]">
        {/* Technical Coordinate Grid Pattern */}
        <div
          className="absolute inset-0 opacity-[0.07] pointer-events-none z-10"
          style={{
            backgroundImage:
              "linear-gradient(rgba(56,189,248,0.4) 1px, transparent 1px), linear-gradient(90deg, rgba(56,189,248,0.4) 1px, transparent 1px)",
            backgroundSize: "32px 32px",
          }}
        />

        {/* Scan-line Sweep Animation */}
        <div className="absolute inset-0 overflow-hidden pointer-events-none z-10">
          <div
            className="w-full h-[2px] bg-gradient-to-r from-transparent via-cyan-400/50 to-transparent animate-scan"
            style={{ animationDuration: "5s" }}
          />
        </div>

        {/* Scalable Container */}
        <div
          className="w-full h-full grid grid-cols-1 sm:grid-cols-2 min-h-[360px] sm:min-h-[440px] transition-transform duration-300 origin-center"
          style={{ transform: `scale(${zoom})` }}
        >
          {/* Historical Baseline (Left Panel) */}
          <div className="relative border-b sm:border-b-0 sm:border-r border-slate-800/80 bg-[#091522] flex flex-col justify-between p-4 overflow-hidden group">
            {/* Real Satellite Imagery Background */}
            <div
              className="absolute inset-0 bg-cover bg-center opacity-70 mix-blend-luminosity contrast-125 brightness-90 transition-all duration-300"
              style={{
                backgroundImage:
                  "url('https://images.unsplash.com/photo-1541185933-ef5d8ed016c2?q=80&w=1200&auto=format&fit=crop')",
              }}
            />
            <div className="absolute inset-0 bg-gradient-to-t from-[#091522] via-transparent to-[#091522]/80 pointer-events-none" />

            {/* Corner Bracket Marks */}
            <div className="absolute top-2 left-2 w-3 h-3 border-t-2 border-l-2 border-slate-400/60 z-20" />
            <div className="absolute bottom-2 left-2 w-3 h-3 border-b-2 border-l-2 border-slate-400/60 z-20" />

            {/* Timestamp Badge */}
            <div className="relative z-20 flex items-center justify-between">
              <div className="flex items-center gap-1.5 bg-slate-900/85 backdrop-blur-md rounded-lg px-2.5 py-1 text-[10px] text-slate-300 font-mono border border-slate-700/60 shadow">
                <span className="w-1.5 h-1.5 rounded-full bg-slate-400" />
                <span>T0: Oct 12, 2023 · 02:11Z</span>
              </div>
              <span className="text-[9px] font-mono text-slate-300 bg-slate-900/80 px-2 py-0.5 rounded border border-slate-800">
                S2B_MSI_L2A
              </span>
            </div>

            {/* Center Reticle */}
            <div className="relative z-20 flex flex-col items-center justify-center my-auto py-8">
              <div className="w-16 h-16 rounded-full border border-dashed border-slate-300/40 backdrop-blur-xs flex items-center justify-center mb-2 shadow-lg">
                <Crosshair size={24} className="text-slate-200" />
              </div>
              <span className="text-[11px] font-mono text-slate-200 uppercase tracking-widest bg-[#09111c]/90 px-3 py-1 rounded-full border border-slate-700 shadow">
                Historical Baseline
              </span>
              <span className="text-[9px] font-mono text-slate-300 mt-1 bg-slate-900/80 px-2 py-0.5 rounded">
                Ref: EO-LANDSAT-8921
              </span>
            </div>

            {/* Bottom Telemetry */}
            <div className="relative z-20 flex items-center justify-between text-[9px] font-mono text-slate-300 bg-slate-900/80 px-2 py-1 rounded border border-slate-800">
              <span>CLOUD: 1.2%</span>
              <span>AZ: 142.8°</span>
            </div>
          </div>

          {/* Current Recon Telemetry (Right Panel) */}
          <div className="relative bg-[#081726] flex flex-col justify-between p-4 overflow-hidden group">
            {/* Real Satellite Imagery Background */}
            <div
              className="absolute inset-0 bg-cover bg-center opacity-85 contrast-125 brightness-95 transition-all duration-300"
              style={{
                backgroundImage:
                  "url('https://images.unsplash.com/photo-1506703719100-a0f3a48c0f86?q=80&w=1200&auto=format&fit=crop')",
              }}
            />
            <div className="absolute inset-0 bg-gradient-to-t from-[#081726] via-transparent to-[#081726]/80 pointer-events-none" />

            {/* Corner Bracket Marks */}
            <div className="absolute top-2 right-2 w-3 h-3 border-t-2 border-r-2 border-cyan-400/80 z-20" />
            <div className="absolute bottom-2 right-2 w-3 h-3 border-b-2 border-r-2 border-cyan-400/80 z-20" />

            {/* Dynamic Delta Heatmap Overlay */}
            {deltaHeatmap && (
              <div
                className="absolute inset-0 bg-gradient-to-tr from-amber-500/30 via-red-500/25 to-cyan-500/20 mix-blend-color-dodge pointer-events-none transition-opacity duration-300 z-10"
                style={{ opacity: opacity / 100 }}
              />
            )}

            {/* Dynamic NDVI color wash */}
            {ndviOverlay && (
              <div
                className="absolute inset-0 bg-emerald-500/30 mix-blend-color-dodge pointer-events-none transition-opacity duration-300 z-10"
                style={{ opacity: opacity / 100 }}
              />
            )}

            {/* Anomaly Bounding Boxes */}
            <div className="absolute inset-0 pointer-events-none z-15">
              {/* Anomaly #01 Vegetation Loss */}
              <div
                onClick={() => setActiveAnomaly("veg")}
                className={`absolute top-[25%] left-[30%] w-32 h-24 border-2 ${
                  activeAnomaly === "veg" || hoveredAnomaly === "veg"
                    ? "border-amber-400 bg-amber-500/20 shadow-[0_0_20px_rgba(245,158,11,0.6)]"
                    : "border-amber-400/70 bg-amber-500/10"
                } rounded-lg pointer-events-auto cursor-pointer transition-all duration-200 flex flex-col justify-between p-1.5`}
              >
                <span className="text-[9px] font-mono font-bold text-amber-300 bg-slate-900/90 px-1.5 py-0.5 rounded self-start">
                  ANOMALY #01 (VEG)
                </span>
                <span className="text-[8px] font-mono text-emerald-300 bg-slate-900/90 px-1 py-0.5 rounded self-end">
                  Δ 4.2 km²
                </span>
              </div>

              {/* Anomaly #02 New Structure */}
              <div
                onClick={() => setActiveAnomaly("struct")}
                className={`absolute top-[55%] left-[55%] w-28 h-20 border-2 ${
                  activeAnomaly === "struct" || hoveredAnomaly === "struct"
                    ? "border-red-400 bg-red-500/20 shadow-[0_0_20px_rgba(239,68,68,0.6)]"
                    : "border-red-400/70 bg-red-500/10"
                } rounded-lg pointer-events-auto cursor-pointer transition-all duration-200 flex flex-col justify-between p-1.5`}
              >
                <span className="text-[9px] font-mono font-bold text-red-300 bg-slate-900/90 px-1.5 py-0.5 rounded self-start">
                  ANOMALY #02 (STR)
                </span>
                <span className="text-[8px] font-mono text-cyan-300 bg-slate-900/90 px-1 py-0.5 rounded self-end">
                  0.8 km²
                </span>
              </div>
            </div>

            {/* Timestamp Badge */}
            <div className="relative z-20 flex items-center justify-between">
              <div className="flex items-center gap-1.5 bg-slate-900/85 backdrop-blur-md rounded-lg px-2.5 py-1 text-[10px] text-cyan-300 font-mono border border-cyan-500/30 shadow">
                <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse shadow-[0_0_6px_rgba(6,182,212,0.8)]" />
                <span>T1: Oct 14, 2024 · 02:12Z</span>
              </div>
              <span className="text-[9px] font-mono text-cyan-400 bg-cyan-500/10 px-2 py-0.5 rounded border border-cyan-500/20">
                SAR_COSMO_X
              </span>
            </div>

            {/* Center Label & Reticle */}
            <div className="relative z-20 flex flex-col items-center justify-center my-auto py-8">
              <div className="w-16 h-16 rounded-full border border-dashed border-cyan-400/60 backdrop-blur-xs flex items-center justify-center mb-2 shadow-lg">
                <Crosshair size={24} className="text-cyan-300 animate-pulse" />
              </div>
              <span className="text-[11px] font-mono text-cyan-200 uppercase tracking-widest bg-[#09111c]/90 px-3 py-1 rounded-full border border-cyan-500/40 shadow-[0_0_12px_rgba(6,182,212,0.3)]">
                Current Recon Telemetry
              </span>
              <span className="text-[9px] font-mono text-cyan-400/60 mt-1">
                Active Sentinel Surveillance
              </span>
            </div>

            {/* Anomaly 1: Vegetation Loss (Amber/Yellow) */}
            {deltaHeatmap && (
              <div
                onClick={() => setActiveAnomaly(activeAnomaly === "veg" ? null : "veg")}
                className={`absolute w-32 h-24 border-2 rounded-lg bottom-10 right-6 flex flex-col justify-between p-1.5 transition-all duration-300 cursor-pointer z-30 ${
                  activeAnomaly === "veg" || hoveredAnomaly === "veg"
                    ? "border-amber-300 bg-amber-400/25 shadow-[0_0_35px_rgba(252,211,77,0.7)] scale-105"
                    : "border-amber-400/80 bg-amber-400/15 shadow-[0_0_15px_rgba(251,191,36,0.3)] hover:border-amber-300 hover:scale-[1.02]"
                }`}
                style={{
                  opacity: Math.max(opacity / 100, 0.35),
                }}
              >
                <div className="flex items-center justify-between">
                  <span className="text-[8px] font-mono font-bold text-amber-300 bg-black/80 px-1 py-0.5 rounded border border-amber-400/40">
                    ANOMALY #01
                  </span>
                  <span className="w-1.5 h-1.5 rounded-full bg-amber-400 animate-ping" />
                </div>
                <div className="flex items-center justify-between text-[9px] font-mono font-semibold text-amber-200 bg-black/80 px-1.5 py-0.5 rounded border border-amber-400/30">
                  <span>Δ VEG (96%)</span>
                  <span>4.2 km²</span>
                </div>
              </div>
            )}

            {/* Anomaly 2: New Structure (Red) */}
            {deltaHeatmap && (
              <div
                onClick={() => setActiveAnomaly(activeAnomaly === "struct" ? null : "struct")}
                className={`absolute w-24 h-20 border-2 rounded-lg top-14 right-14 flex flex-col justify-between p-1.5 transition-all duration-300 cursor-pointer z-30 ${
                  activeAnomaly === "struct" || hoveredAnomaly === "struct"
                    ? "border-red-400 bg-red-500/25 shadow-[0_0_35px_rgba(248,113,113,0.8)] scale-105"
                    : "border-red-500/80 bg-red-500/15 shadow-[0_0_15px_rgba(239,68,68,0.3)] hover:border-red-400 hover:scale-[1.02]"
                }`}
                style={{
                  opacity: Math.max(opacity / 100, 0.35),
                }}
              >
                <div className="flex items-center justify-between">
                  <span className="text-[8px] font-mono font-bold text-red-300 bg-black/80 px-1 py-0.5 rounded border border-red-400/40">
                    ANOMALY #02
                  </span>
                  <span className="w-1.5 h-1.5 rounded-full bg-red-400 animate-ping" />
                </div>
                <div className="flex items-center justify-between text-[9px] font-mono font-semibold text-red-200 bg-black/80 px-1 py-0.5 rounded border border-red-400/30">
                  <span>STR (87%)</span>
                  <span>0.8 km²</span>
                </div>
              </div>
            )}

            {/* Bottom Telemetry */}
            <div className="relative z-20 flex items-center justify-between text-[9px] font-mono text-cyan-400/70">
              <span>SAR POL: VV+VH</span>
              <span>GSD: 0.5m</span>
            </div>
          </div>
        </div>

        {/* Center Split Divider Bar */}
        <div className="absolute top-0 bottom-0 left-1/2 -translate-x-1/2 w-0.5 bg-cyan-400/40 hidden sm:flex flex-col items-center justify-center pointer-events-none z-20 shadow-[0_0_8px_rgba(6,182,212,0.6)]">
          <div className="w-6 h-6 rounded-full bg-[#0a1420] border-2 border-cyan-400 flex items-center justify-center shadow-[0_0_12px_rgba(6,182,212,0.5)]">
            <span className="text-[8px] font-mono font-bold text-cyan-300">Δ</span>
          </div>
        </div>
      </div>

      {/* Bottom Telemetry Strip */}
      <div className="px-4 py-2.5 bg-[#09111c] border-t border-slate-800/80 flex flex-wrap items-center justify-between gap-2 text-[10px] font-mono text-slate-400 z-20">
        <div className="flex items-center gap-3">
          <span>COORDINATES: LAT 34.9522° N, LON 118.2437° W</span>
          <span className="text-slate-700 hidden md:inline">|</span>
          <span className="text-slate-500 hidden md:inline">ELEV: +412M MSL</span>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-cyan-400 font-semibold">
            MAG: {(14.2 * zoom).toFixed(1)}×
          </span>
          <span className="text-slate-700">|</span>
          <span>SENSOR: SAR-B INTERFEROMETRY</span>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------
// Interactive Analysis side panel with Tabs
// ---------------------------------------------
function AnalysisPanel({
  opacity,
  setOpacity,
  deltaHeatmap,
  setDeltaHeatmap,
  ndviOverlay,
  setNdviOverlay,
  activeAnomaly,
  hoveredAnomaly,
  setActiveAnomaly,
  setHoveredAnomaly,
}: {
  opacity: number;
  setOpacity: (v: number) => void;
  deltaHeatmap: boolean;
  setDeltaHeatmap: (v: boolean | ((prev: boolean) => boolean)) => void;
  ndviOverlay: boolean;
  setNdviOverlay: (v: boolean | ((prev: boolean) => boolean)) => void;
  activeAnomaly: string | null;
  hoveredAnomaly: string | null;
  setActiveAnomaly: (v: string | null) => void;
  setHoveredAnomaly: (v: string | null) => void;
}) {
  const [activeTab, setActiveTab] = useState<"Analysis" | "Controls" | "Metadata">("Analysis");
  const [exporting, setExporting] = useState(false);
  const [exported, setExported] = useState(false);

  const tabs: ("Analysis" | "Controls" | "Metadata")[] = ["Analysis", "Controls", "Metadata"];

  const handleExport = async () => {
    if (exporting) return;
    setExporting(true);
    try {
      window.open("/api/comparisons/comp_b492_xt_p/export?format=pdf", "_blank");
      setExported(true);
      setTimeout(() => setExported(false), 3000);
    } catch (e) {
      console.error(e);
    } finally {
      setExporting(false);
    }
  };

  return (
    <div className="w-full bg-[#0f1b29]/95 border border-slate-800/90 rounded-xl p-4 sm:p-5 flex flex-col justify-between shadow-[0_4px_30px_rgba(0,0,0,0.5)] min-h-[480px]">
      <div>
        {/* Animated Tabs Bar */}
        <div className="relative flex items-center gap-2 border-b border-slate-800/80 pb-2 mb-4">
          {tabs.map((tab) => {
            const isSelected = activeTab === tab;
            return (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`relative px-3 py-1.5 text-xs font-medium transition-colors duration-200 cursor-pointer rounded-lg ${
                  isSelected ? "text-cyan-300 font-semibold" : "text-slate-400 hover:text-slate-200"
                }`}
              >
                {tab}
                {isSelected && (
                  <motion.div
                    layoutId="comparison-active-tab"
                    className="absolute bottom-[-9px] left-0 right-0 h-[2px] bg-gradient-to-r from-cyan-400 to-sky-400 shadow-[0_0_8px_rgba(6,182,212,0.8)]"
                    transition={{ type: "spring", stiffness: 380, damping: 30 }}
                  />
                )}
              </button>
            );
          })}
        </div>

        {/* Tab 1: Analysis */}
        {activeTab === "Analysis" && (
          <div className="space-y-4">
            {/* Model Status Card */}
            <div className="flex items-center justify-between p-3 rounded-lg bg-slate-900/60 border border-slate-800/80">
              <div className="flex items-center gap-2">
                <div className="p-1.5 rounded-md bg-cyan-500/10 border border-cyan-500/20 text-cyan-400">
                  <Sparkles size={13} />
                </div>
                <div>
                  <p className="text-xs font-semibold text-white tracking-tight">
                    Change Detection Model
                  </p>
                  <p className="text-[10px] text-slate-400 font-mono">
                    DeepRes-U-Net Temporal v2.4
                  </p>
                </div>
              </div>
              <span className="text-[10px] font-mono font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 rounded-full px-2 py-0.5">
                ACTIVE
              </span>
            </div>

            {/* Detected Anomalies Section */}
            <div>
              <div className="flex items-center justify-between mb-2">
                <p className="text-[10px] text-slate-400 uppercase tracking-widest font-mono">
                  Detected Anomalies (2)
                </p>
                <span className="text-[10px] text-cyan-400 font-mono">
                  Confidence &gt; 85%
                </span>
              </div>

              <div className="space-y-2.5">
                {/* Anomaly 1: Vegetation Loss */}
                <div
                  onMouseEnter={() => setHoveredAnomaly("veg")}
                  onMouseLeave={() => setHoveredAnomaly(null)}
                  className={`p-3 rounded-xl border transition-all duration-200 ${
                    activeAnomaly === "veg" || hoveredAnomaly === "veg"
                      ? "bg-amber-500/10 border-amber-400/50 shadow-[0_0_15px_rgba(251,191,36,0.15)]"
                      : "bg-slate-900/50 border-slate-800/80 hover:border-slate-700 hover:bg-slate-900/80"
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <span className="w-2 h-2 rounded-full bg-amber-400 shadow-[0_0_8px_rgba(251,191,36,0.8)]" />
                      <p className="text-xs font-semibold text-slate-100">
                        Vegetation Loss
                      </p>
                    </div>
                    <span className="text-[10px] font-mono font-bold text-amber-300 bg-amber-500/15 border border-amber-500/30 px-2 py-0.5 rounded-md">
                      96% CONF
                    </span>
                  </div>
                  <p className="text-[11px] text-slate-400 mt-1 ml-4 font-mono">
                    Area: 4.2 km² · Sector 12 · Deforestation signature
                  </p>
                  <div className="flex items-center justify-between mt-2.5 pt-2 border-t border-slate-800/60 ml-4">
                    <span className="text-[10px] font-mono text-slate-400">
                      LAT: 34.951° LON: −118.241°
                    </span>
                    <button
                      onClick={() => setActiveAnomaly(activeAnomaly === "veg" ? null : "veg")}
                      className="flex items-center gap-1 text-[11px] font-semibold text-cyan-400 hover:text-cyan-300 cursor-pointer font-mono hover:scale-105 active:scale-95 transition-transform"
                    >
                      <MapPin size={11} />
                      <span>{activeAnomaly === "veg" ? "Target Centered" : "Locate Target"}</span>
                    </button>
                  </div>
                </div>

                {/* Anomaly 2: New Structure */}
                <div
                  onMouseEnter={() => setHoveredAnomaly("struct")}
                  onMouseLeave={() => setHoveredAnomaly(null)}
                  className={`p-3 rounded-xl border transition-all duration-200 ${
                    activeAnomaly === "struct" || hoveredAnomaly === "struct"
                      ? "bg-red-500/10 border-red-400/50 shadow-[0_0_15px_rgba(239,68,68,0.15)]"
                      : "bg-slate-900/50 border-slate-800/80 hover:border-slate-700 hover:bg-slate-900/80"
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <span className="w-2 h-2 rounded-full bg-red-400 shadow-[0_0_8px_rgba(239,68,68,0.8)]" />
                      <p className="text-xs font-semibold text-slate-100">
                        New Structure
                      </p>
                    </div>
                    <span className="text-[10px] font-mono font-bold text-red-300 bg-red-500/15 border border-red-500/30 px-2 py-0.5 rounded-md">
                      87% CONF
                    </span>
                  </div>
                  <p className="text-[11px] text-slate-400 mt-1 ml-4 font-mono">
                    Type: Industrial Compound · Area: 0.8 km²
                  </p>
                  <div className="flex items-center justify-between mt-2.5 pt-2 border-t border-slate-800/60 ml-4">
                    <span className="text-[10px] font-mono text-slate-400">
                      LAT: 34.958° LON: −118.239°
                    </span>
                    <button
                      onClick={() => setActiveAnomaly(activeAnomaly === "struct" ? null : "struct")}
                      className="flex items-center gap-1 text-[11px] font-semibold text-cyan-400 hover:text-cyan-300 cursor-pointer font-mono hover:scale-105 active:scale-95 transition-transform"
                    >
                      <MapPin size={11} />
                      <span>{activeAnomaly === "struct" ? "Target Centered" : "Locate Target"}</span>
                    </button>
                  </div>
                </div>
              </div>
            </div>

            {/* Visual Overlays Toggles */}
            <div className="pt-2 border-t border-slate-800/80">
              <p className="text-[10px] text-slate-400 uppercase tracking-widest mb-2 font-mono">
                Visual Overlays
              </p>

              <div className="space-y-1.5">
                <button
                  onClick={() => setDeltaHeatmap((v) => !v)}
                  className="w-full flex items-center justify-between p-2 rounded-lg bg-slate-900/40 hover:bg-slate-900/80 border border-slate-800/60 hover:border-slate-700 transition-all cursor-pointer"
                >
                  <div className="flex items-center gap-2">
                    <Layers size={13} className="text-amber-400" />
                    <span className="text-xs text-slate-200 font-medium">
                      Delta Heatmap
                    </span>
                  </div>
                  <span
                    className={`w-9 h-5 rounded-full relative transition-colors duration-200 ${
                      deltaHeatmap ? "bg-cyan-500 shadow-[0_0_8px_rgba(6,182,212,0.4)]" : "bg-slate-800"
                    }`}
                  >
                    <span
                      className={`absolute top-0.5 w-4 h-4 rounded-full bg-white shadow transition-all duration-200 ${
                        deltaHeatmap ? "left-[18px]" : "left-0.5"
                      }`}
                    />
                  </span>
                </button>

                <button
                  onClick={() => setNdviOverlay((v) => !v)}
                  className="w-full flex items-center justify-between p-2 rounded-lg bg-slate-900/40 hover:bg-slate-900/80 border border-slate-800/60 hover:border-slate-700 transition-all cursor-pointer"
                >
                  <div className="flex items-center gap-2">
                    <Activity size={13} className="text-emerald-400" />
                    <span className="text-xs text-slate-200 font-medium">
                      Spectral Index (NDVI)
                    </span>
                  </div>
                  <span
                    className={`w-9 h-5 rounded-full relative transition-colors duration-200 ${
                      ndviOverlay ? "bg-cyan-500 shadow-[0_0_8px_rgba(6,182,212,0.4)]" : "bg-slate-800"
                    }`}
                  >
                    <span
                      className={`absolute top-0.5 w-4 h-4 rounded-full bg-white shadow transition-all duration-200 ${
                        ndviOverlay ? "left-[18px]" : "left-0.5"
                      }`}
                    />
                  </span>
                </button>
              </div>
            </div>

            {/* Live Opacity Slider */}
            <div className="pt-2">
              <div className="flex items-center justify-between mb-1.5">
                <span className="text-[10px] text-slate-400 uppercase tracking-widest font-mono">
                  Overlay Opacity
                </span>
                <span className="text-xs text-cyan-400 font-mono font-bold">
                  {opacity}%
                </span>
              </div>
              <div className="relative">
                <div className="h-1.5 w-full bg-slate-800 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-cyan-500 to-sky-400 rounded-full transition-all duration-150"
                    style={{ width: `${opacity}%` }}
                  />
                </div>
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={opacity}
                  onChange={(e) => setOpacity(Number(e.target.value))}
                  className="absolute inset-0 w-full opacity-0 cursor-pointer h-4 -top-1"
                />
              </div>
            </div>
          </div>
        )}

        {/* Tab 2: Controls */}
        {activeTab === "Controls" && (
          <div className="space-y-4">
            <p className="text-xs text-slate-400">Sensor Display Enhancements</p>
            <div className="p-3 rounded-lg bg-slate-900/60 border border-slate-800/80 space-y-3">
              <div>
                <div className="flex justify-between text-[11px] text-slate-300 font-mono mb-1.5">
                  <span>Contrast Boost</span>
                  <span className="text-cyan-400 font-bold">+12%</span>
                </div>
                <input
                  type="range"
                  defaultValue={62}
                  className="w-full accent-cyan-400 cursor-pointer h-1.5 bg-slate-800 rounded-lg"
                />
              </div>
              <div>
                <div className="flex justify-between text-[11px] text-slate-300 font-mono mb-1.5">
                  <span>Gamma Calibration</span>
                  <span className="text-cyan-400 font-bold">1.05</span>
                </div>
                <input
                  type="range"
                  defaultValue={50}
                  className="w-full accent-cyan-400 cursor-pointer h-1.5 bg-slate-800 rounded-lg"
                />
              </div>
              <div>
                <div className="flex justify-between text-[11px] text-slate-300 font-mono mb-1.5">
                  <span>Kernel Sharpening</span>
                  <span className="text-cyan-400 font-bold">3×3 Conv</span>
                </div>
                <input
                  type="range"
                  defaultValue={30}
                  className="w-full accent-cyan-400 cursor-pointer h-1.5 bg-slate-800 rounded-lg"
                />
              </div>
            </div>
          </div>
        )}

        {/* Tab 3: Metadata */}
        {activeTab === "Metadata" && (
          <div className="space-y-2.5 font-mono text-xs">
            <div className="bg-slate-900/60 p-3 rounded-lg border border-slate-800/80">
              <span className="text-slate-500 block text-[10px] uppercase">BASELINE SENSOR</span>
              <span className="text-slate-200 font-medium">Sentinel-2B MSI (10m GSD)</span>
            </div>
            <div className="bg-slate-900/60 p-3 rounded-lg border border-slate-800/80">
              <span className="text-slate-500 block text-[10px] uppercase">RECON SENSOR</span>
              <span className="text-slate-200 font-medium">SAR-X Band Cosmo-SkyMed (0.5m)</span>
            </div>
            <div className="bg-slate-900/60 p-3 rounded-lg border border-slate-800/80">
              <span className="text-slate-500 block text-[10px] uppercase">SOLAR AZIMUTH / ELEV</span>
              <span className="text-cyan-300 font-semibold">142.84° / +56.2°</span>
            </div>
            <div className="bg-slate-900/60 p-3 rounded-lg border border-slate-800/80">
              <span className="text-slate-500 block text-[10px] uppercase">CORE-G ALGORITHM</span>
              <span className="text-emerald-400 font-semibold">ORB-Phase-Correlation v3.1</span>
            </div>
          </div>
        )}
      </div>

      {/* Main Export CTA Button */}
      <button
        onClick={handleExport}
        disabled={exporting}
        className="mt-6 flex items-center justify-center gap-2 bg-gradient-to-r from-cyan-500 to-sky-400 hover:from-cyan-400 hover:to-sky-300 active:scale-95 disabled:opacity-75 transition-all text-[#0a1420] text-xs font-bold rounded-xl py-3 shadow-[0_0_15px_rgba(6,182,212,0.35)] hover:shadow-[0_0_22px_rgba(6,182,212,0.6)] cursor-pointer"
      >
        {exporting ? (
          <>
            <Loader2 size={14} className="animate-spin text-[#0a1420]" />
            <span>Compiling Report & GeoJSON...</span>
          </>
        ) : exported ? (
          <>
            <CheckCircle2 size={14} className="text-[#0a1420]" />
            <span>Assessment Exported!</span>
          </>
        ) : (
          <>
            <Download size={14} className="text-[#0a1420]" />
            <span>Export Temporal Assessment</span>
          </>
        )}
      </button>
    </div>
  );
}

// ---------------------------------------------
// Comparison Page
// ---------------------------------------------
export default function ComparisonPage() {
  const [opacity, setOpacity] = useState(75);
  const [deltaHeatmap, setDeltaHeatmap] = useState(true);
  const [ndviOverlay, setNdviOverlay] = useState(false);
  const [activeAnomaly, setActiveAnomaly] = useState<string | null>(null);
  const [hoveredAnomaly, setHoveredAnomaly] = useState<string | null>(null);
  const [zoom, setZoom] = useState(1);

  return (
    <div className="flex flex-col lg:flex-row min-h-screen bg-[#0a1420] text-white font-sans overflow-x-hidden">
      <Sidebar />

      <div className="flex-1 flex flex-col min-w-0">
        <TopBar />

        <main className="flex-1 p-4 sm:p-6 lg:p-8 grid grid-cols-1 lg:grid-cols-3 gap-5 items-start overflow-y-auto">
          <div className="lg:col-span-2">
            <ImageViewer
              opacity={opacity}
              deltaHeatmap={deltaHeatmap}
              ndviOverlay={ndviOverlay}
              activeAnomaly={activeAnomaly}
              hoveredAnomaly={hoveredAnomaly}
              setActiveAnomaly={setActiveAnomaly}
              zoom={zoom}
              setZoom={setZoom}
            />
          </div>
          <div className="lg:col-span-1">
            <AnalysisPanel
              opacity={opacity}
              setOpacity={setOpacity}
              deltaHeatmap={deltaHeatmap}
              setDeltaHeatmap={setDeltaHeatmap}
              ndviOverlay={ndviOverlay}
              setNdviOverlay={setNdviOverlay}
              activeAnomaly={activeAnomaly}
              hoveredAnomaly={hoveredAnomaly}
              setActiveAnomaly={setActiveAnomaly}
              setHoveredAnomaly={setHoveredAnomaly}
            />
          </div>
        </main>
      </div>
    </div>
  );
}