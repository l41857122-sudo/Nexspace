"use client";

import React, { useState, useEffect, useRef } from "react";
import { ArrowRight, Layers, Eye, ShieldCheck, CheckCircle2, Crosshair, MapPin } from "lucide-react";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";

interface CategoryPill {
  name: string;
  color: string;
  bgColor: string;
  borderColor: string;
}

const CATEGORIES: CategoryPill[] = [
  {
    name: "Buildings",
    color: "#00f0ff",
    bgColor: "rgba(0, 240, 255, 0.15)",
    borderColor: "rgba(0, 240, 255, 0.6)",
  },
  {
    name: "Roads",
    color: "#10b981",
    bgColor: "rgba(16, 185, 129, 0.15)",
    borderColor: "rgba(16, 185, 129, 0.6)",
  },
  {
    name: "Vegetation",
    color: "#06b6d4",
    bgColor: "rgba(6, 182, 212, 0.15)",
    borderColor: "rgba(6, 182, 212, 0.6)",
  },
  {
    name: "Water",
    color: "#3b82f6",
    bgColor: "rgba(59, 130, 246, 0.15)",
    borderColor: "rgba(59, 130, 246, 0.6)",
  },
  {
    name: "Infrastructure",
    color: "#f59e0b",
    bgColor: "rgba(245, 158, 11, 0.15)",
    borderColor: "rgba(245, 158, 11, 0.6)",
  },
];

const STEPS = [
  { id: 0, label: "01 RAW SENSOR", desc: "Unprocessed multi-spectral reflectance" },
  { id: 1, label: "02 SHARPEN", desc: "Atmospheric correction & noise filtering" },
  { id: 2, label: "03 GRID SYNC", desc: "Geodetic CRS & analytical coordinate grid" },
  { id: 3, label: "04 ROADS", desc: "Arterial road geometry vectors resolved" },
  { id: 4, label: "05 BUILDINGS", desc: "Building polygons & boundary extraction" },
  { id: 5, label: "06 SPECTRAL LAYERS", desc: "Vegetation, water & infrastructure masks" },
  { id: 6, label: "07 TARGET LOCK", desc: "Spatial feature isolation & focus lock" },
  { id: 7, label: "08 VERIFIED", desc: "Calibrated geometric evidence sealed" },
];

export default function PixelsToIntelligenceScene() {
  const containerRef = useRef<HTMLDivElement>(null);
  const [scrollProgress, setScrollProgress] = useState(0);
  const [manualStep, setManualStep] = useState<number | null>(null);

  useEffect(() => {
    gsap.registerPlugin(ScrollTrigger);

    const el = containerRef.current;
    if (!el) return;

    // Check prefers-reduced-motion or mobile
    const isMobile = window.innerWidth < 1024;
    const isReduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    if (isReduced || isMobile) {
      // Don't pin on mobile/reduced-motion; let users scroll naturally and click steps
      return;
    }

    const st = ScrollTrigger.create({
      trigger: el,
      start: "top top",
      end: "+=160%",
      pin: true,
      scrub: 0.5,
      onUpdate: (self) => {
        setScrollProgress(self.progress);
        if (manualStep !== null) {
          setManualStep(null);
        }
      },
    });

    return () => {
      st.kill();
    };
  }, [manualStep]);

  // Determine active step index: manual override or scrollProgress (0 to 1 -> 0 to 7)
  const effectiveStep =
    manualStep !== null ? manualStep : Math.min(7, Math.floor(scrollProgress * 8));

  // Determine which 3-stage preview card is active (0: Raw, 1: Spatial Analysis, 2: Verified Geometry)
  const activeSummaryStage = effectiveStep <= 1 ? 0 : effectiveStep <= 5 ? 1 : 2;

  // Visual interpolation values
  // Step 1: blur decreases from 8px to 0px, grayscale from 100% to 0%
  const blurVal = Math.max(0, 8 - effectiveStep * 4);
  const grayscaleVal = effectiveStep === 0 ? 100 : effectiveStep === 1 ? 40 : 0;
  const contrastVal = effectiveStep === 0 ? 120 : 105;

  return (
    <section
      ref={containerRef}
      id="pixels"
      className="relative min-h-screen w-full bg-[#010308] text-slate-100 py-20 px-6 sm:px-10 lg:px-14 flex flex-col justify-center overflow-hidden border-t border-cyan-950/30 select-none"
    >
      {/* Background Volumetric Glow */}
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[900px] h-[600px] bg-cyan-600/5 rounded-full blur-[180px] pointer-events-none" />

      <div className="max-w-[1440px] mx-auto w-full relative z-10 flex flex-col justify-between">
        {/* Section Header Row matching Reference Panel 04 */}
        <div className="flex items-center justify-between pb-3 border-b border-slate-800/60 mb-8">
          <div className="flex items-center gap-3">
            <span className="text-xs font-mono font-bold text-[#00f0ff] tracking-[0.25em] uppercase">
              04
            </span>
            <span className="text-slate-600 font-mono text-xs">/</span>
            <span className="text-xs font-mono tracking-[0.2em] text-slate-400 uppercase">
              0:16 — 0:22 PIXELS BECOME INTELLIGENCE
            </span>
          </div>

          <div className="hidden sm:flex items-center gap-2 font-mono text-xs text-[#00f0ff]">
            <span className="w-2 h-2 rounded-full bg-[#00f0ff] animate-pulse" />
            <span className="tracking-widest uppercase">
              STEP {effectiveStep + 1} OF 8: {STEPS[effectiveStep].label.split(" ").slice(1).join(" ")}
            </span>
          </div>
        </div>

        {/* Section Title */}
        <div className="text-center max-w-3xl mx-auto mb-8">
          <h2 className="text-3xl sm:text-5xl lg:text-6xl font-extrabold tracking-tight text-white mb-3 uppercase font-sans">
            PIXELS BECOME <span className="text-[#00f0ff]">INTELLIGENCE</span>
          </h2>
          <p className="text-slate-400 text-xs sm:text-sm leading-relaxed font-light">
            Continuous transformation from raw electromagnetic sensor recordings into verified
            geospatial structure.
          </p>
        </div>

        {/* Interactive 8-Step Timeline Scrubber */}
        <div className="w-full max-w-5xl mx-auto mb-8 px-2">
          {/* Progress bar line */}
          <div className="relative h-1 bg-slate-800/80 rounded-full mb-3 overflow-hidden">
            <div
              className="absolute top-0 left-0 h-full bg-gradient-to-r from-cyan-500 to-[#00f0ff] transition-all duration-300 shadow-[0_0_12px_#00f0ff]"
              style={{ width: `${((effectiveStep + 1) / STEPS.length) * 100}%` }}
            />
          </div>

          {/* Step Pill Badges */}
          <div className="grid grid-cols-4 sm:grid-cols-8 gap-1 sm:gap-2">
            {STEPS.map((step) => {
              const isPast = step.id < effectiveStep;
              const isCurrent = step.id === effectiveStep;

              return (
                <button
                  key={step.id}
                  onClick={() => setManualStep(step.id)}
                  className={`px-2 py-1.5 rounded-lg border text-[10px] font-mono transition-all text-center cursor-pointer ${
                    isCurrent
                      ? "bg-cyan-950/80 border-[#00f0ff] text-[#00f0ff] shadow-[0_0_12px_rgba(0,240,255,0.4)] font-bold scale-105"
                      : isPast
                      ? "bg-[#040d1a] border-cyan-800/60 text-slate-300 hover:border-cyan-500/50"
                      : "bg-[#030914] border-slate-800/80 text-slate-500 hover:border-slate-700"
                  }`}
                >
                  <div className="truncate">{step.label}</div>
                </button>
              );
            })}
          </div>
        </div>

        {/* Signature Central Analytical Inspection Viewport (Product Film Scroll Moment) */}
        <div className="relative w-full max-w-5xl mx-auto aspect-[16/9] max-h-[500px] rounded-2xl overflow-hidden border border-cyan-500/40 bg-[#020712] shadow-[0_0_50px_rgba(0,0,0,0.9)] mb-8">
          {/* Layer 1: Base Satellite Image with dynamic blur, grayscale, contrast */}
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src="/imagery/satellite_urban_surface.jpg"
            alt="Sensor Observation"
            className="w-full h-full object-cover transition-all duration-500"
            style={{
              filter: `blur(${blurVal}px) grayscale(${grayscaleVal}%) contrast(${contrastVal}%)`,
              transform: effectiveStep >= 6 ? "scale(1.06)" : "scale(1)",
              transformOrigin: "35% 35%",
            }}
          />

          {/* Sensor scanline noise overlay on Step 0 */}
          {effectiveStep === 0 && (
            <div
              className="absolute inset-0 opacity-25 pointer-events-none"
              style={{
                backgroundImage:
                  "repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(255,255,255,0.12) 2px, rgba(255,255,255,0.12) 4px)",
              }}
            />
          )}

          {/* Layer 2: Analytical Geodetic Grid (Step 2+) */}
          <div
            className={`absolute inset-0 pointer-events-none transition-opacity duration-500 ${
              effectiveStep >= 2 ? "opacity-30" : "opacity-0"
            }`}
            style={{
              backgroundImage:
                "linear-gradient(to right, rgba(0,240,255,0.4) 1px, transparent 1px), linear-gradient(to bottom, rgba(0,240,255,0.4) 1px, transparent 1px)",
              backgroundSize: "60px 60px",
            }}
          />

          {/* SVG Vector Overlays: Roads, Buildings, Spectral Polygons, Target Lock */}
          <svg className="absolute inset-0 w-full h-full pointer-events-none" viewBox="0 0 100 100">
            {/* Step 3+: Roads Geometry (Emerald `#10b981`) */}
            {effectiveStep >= 3 && (
              <g className="transition-opacity duration-500 opacity-100">
                <path
                  d="M0,75 Q30,68 60,78 T100,72"
                  fill="none"
                  stroke="#10b981"
                  strokeWidth="2.2"
                  strokeLinecap="round"
                  className="drop-shadow-[0_0_6px_#10b981]"
                />
                <path
                  d="M45,20 L45,100"
                  fill="none"
                  stroke="#10b981"
                  strokeWidth="1.6"
                  strokeDasharray="2 1"
                  className="drop-shadow-[0_0_4px_#10b981]"
                />
              </g>
            )}

            {/* Step 4+: Building Polygons (Cyan `#00f0ff`) */}
            {effectiveStep >= 4 && (
              <g className="transition-opacity duration-500 opacity-100">
                <polygon
                  points="22,25 42,25 42,46 22,46"
                  fill="rgba(0,240,255,0.22)"
                  stroke="#00f0ff"
                  strokeWidth="0.9"
                  className="drop-shadow-[0_0_8px_rgba(0,240,255,0.6)]"
                />
                <polygon
                  points="50,22 80,22 80,48 50,48"
                  fill="rgba(0,240,255,0.2)"
                  stroke="#00f0ff"
                  strokeWidth="0.9"
                />
                <polygon
                  points="54,54 84,54 84,72 54,72"
                  fill="rgba(0,240,255,0.18)"
                  stroke="#00f0ff"
                  strokeWidth="0.8"
                />
              </g>
            )}

            {/* Step 5+: Spectral Classification Polygons (Vegetation & Water) */}
            {effectiveStep >= 5 && (
              <g className="transition-opacity duration-500 opacity-100">
                {/* Vegetation (Mint `#06b6d4`) */}
                <polygon
                  points="10,54 38,50 36,70 8,70"
                  fill="rgba(6,182,212,0.28)"
                  stroke="#06b6d4"
                  strokeWidth="0.8"
                />
                {/* Water (Blue `#3b82f6`) */}
                <polygon
                  points="65,78 95,78 95,95 65,95"
                  fill="rgba(59,130,246,0.3)"
                  stroke="#3b82f6"
                  strokeWidth="0.8"
                />
              </g>
            )}

            {/* Step 6+: Target Isolation & Acquisition Crosshairs */}
            {effectiveStep >= 6 && (
              <g className="transition-all duration-500">
                {/* Target Bounding Reticle over Building Complex */}
                <rect
                  x="20"
                  y="23"
                  width="24"
                  height="25"
                  fill="rgba(0,240,255,0.25)"
                  stroke="#00f0ff"
                  strokeWidth="1.2"
                  className="animate-pulse"
                />
                {/* Center crosshair dot */}
                <circle cx="32" cy="35.5" r="1" fill="#00f0ff" className="drop-shadow-[0_0_6px_#00f0ff]" />

                {/* Step 7: Geometric Morph Connector Line from Target to Telemetry Panel (Requirement 11) */}
                {effectiveStep >= 7 && (
                  <path
                    d="M 44 35.5 L 62 35.5 L 68 62"
                    fill="none"
                    stroke="#00f0ff"
                    strokeWidth="0.8"
                    strokeDasharray="2 1.5"
                    className="animate-pulse drop-shadow-[0_0_8px_#00f0ff]"
                  />
                )}
              </g>
            )}
          </svg>

          {/* Step 6+: Floating Target Acquisition Label */}
          {effectiveStep >= 6 && (
            <div className="absolute top-[18%] left-[22%] -translate-y-full px-3 py-1 rounded-lg bg-[#020712]/95 border border-[#00f0ff] shadow-[0_0_20px_rgba(0,240,255,0.8)] backdrop-blur-md flex items-center gap-2 animate-in fade-in zoom-in-95 duration-300">
              <Crosshair className="w-3.5 h-3.5 text-[#00f0ff] animate-spin" />
              <div className="font-mono text-[10px]">
                <span className="text-[#00f0ff] font-bold">TARGET LOCK: </span>
                <span className="text-white">BLDG-COMPLEX-04</span>
              </div>
            </div>
          )}

          {/* Step 7: Verified Geometry Telemetry Card */}
          {effectiveStep >= 7 && (
            <div className="absolute right-4 bottom-4 max-w-xs p-4 rounded-xl bg-[#030914]/95 border border-cyan-400 shadow-[0_0_30px_rgba(0,240,255,0.4)] backdrop-blur-xl animate-in slide-in-from-right-8 duration-500">
              <div className="flex items-center gap-2 mb-2 pb-1.5 border-b border-cyan-500/30">
                <ShieldCheck className="w-4 h-4 text-[#00f0ff]" />
                <span className="text-[11px] font-mono font-bold text-[#00f0ff] uppercase tracking-wider">
                  VERIFIED GEOMETRY
                </span>
              </div>

              <div className="space-y-1.5 font-mono text-[10.5px]">
                <div className="flex justify-between">
                  <span className="text-slate-400">Footprint:</span>
                  <span className="text-white font-semibold">4,820 m²</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Confidence:</span>
                  <span className="text-[#00f0ff] font-bold">96.8% (Calibrated)</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Sensor:</span>
                  <span className="text-slate-200">Sentinel-2 L2A</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">CRS:</span>
                  <span className="text-slate-200">EPSG:4326 (WGS-84)</span>
                </div>
              </div>
            </div>
          )}

          {/* Top Left Viewport Telemetry Tag */}
          <div className="absolute top-3 left-3 px-2.5 py-1 rounded bg-[#020712]/80 border border-slate-800 text-[10px] font-mono text-cyan-400 backdrop-blur-md flex items-center gap-2">
            <span className="w-1.5 h-1.5 rounded-full bg-[#00f0ff] animate-ping" />
            <span>SENSOR: MULTI-SPECTRAL 10M</span>
          </div>

          {/* Top Right Floating Classification Legend (Step 5+) */}
          {effectiveStep >= 5 && (
            <div className="absolute top-3 right-3 hidden sm:flex flex-col gap-1 z-10 scale-90 origin-top-right">
              {CATEGORIES.map((cat) => (
                <span
                  key={cat.name}
                  className="px-2 py-0.5 rounded text-[9px] font-mono font-bold uppercase tracking-wider backdrop-blur-md"
                  style={{
                    color: cat.color,
                    backgroundColor: cat.bgColor,
                    border: `1px solid ${cat.borderColor}`,
                  }}
                >
                  {cat.name}
                </span>
              ))}
            </div>
          )}
        </div>

        {/* 3 Progressive Analytical Summary Stages matching Reference Panel 04 */}
        <div className="grid grid-cols-1 md:grid-cols-11 gap-3 items-center">
          {/* Stage 1: RAW IMAGERY */}
          <div
            onClick={() => setManualStep(0)}
            className={`md:col-span-3 rounded-2xl p-3.5 bg-[#040c18] border transition-all duration-300 cursor-pointer ${
              activeSummaryStage === 0
                ? "border-cyan-400 shadow-[0_0_24px_rgba(0,240,255,0.25)] bg-[#051426]"
                : "border-slate-800/80 hover:border-slate-700 opacity-60 hover:opacity-100"
            }`}
          >
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-mono font-bold tracking-wider text-[#00f0ff] uppercase">
                RAW IMAGERY
              </span>
              <span className="text-[10px] font-mono text-slate-500">STAGE 01</span>
            </div>
            <p className="text-xs font-mono text-slate-400">Satellite pixels capture the world.</p>
          </div>

          {/* Arrow 1 */}
          <div className="md:col-span-1 flex justify-center py-1 md:py-0">
            <div className="w-7 h-7 rounded-full bg-slate-900 border border-slate-800 flex items-center justify-center text-cyan-400 shadow-md">
              <ArrowRight className="w-3.5 h-3.5 rotate-90 md:rotate-0" />
            </div>
          </div>

          {/* Stage 2: SPATIAL ANALYSIS */}
          <div
            onClick={() => setManualStep(3)}
            className={`md:col-span-3 rounded-2xl p-3.5 bg-[#040c18] border transition-all duration-300 cursor-pointer ${
              activeSummaryStage === 1
                ? "border-cyan-400 shadow-[0_0_24px_rgba(0,240,255,0.25)] bg-[#051426]"
                : "border-slate-800/80 hover:border-slate-700 opacity-60 hover:opacity-100"
            }`}
          >
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-mono font-bold tracking-wider text-[#00f0ff] uppercase">
                SPATIAL ANALYSIS
              </span>
              <span className="text-[10px] font-mono text-slate-500">STAGE 02</span>
            </div>
            <p className="text-xs font-mono text-slate-400">AI detects and understands patterns.</p>
          </div>

          {/* Arrow 2 */}
          <div className="md:col-span-1 flex justify-center py-1 md:py-0">
            <div className="w-7 h-7 rounded-full bg-slate-900 border border-slate-800 flex items-center justify-center text-cyan-400 shadow-md">
              <ArrowRight className="w-3.5 h-3.5 rotate-90 md:rotate-0" />
            </div>
          </div>

          {/* Stage 3: VERIFIED GEOMETRY */}
          <div
            onClick={() => setManualStep(7)}
            className={`md:col-span-3 rounded-2xl p-3.5 bg-[#040c18] border transition-all duration-300 cursor-pointer ${
              activeSummaryStage === 2
                ? "border-cyan-400 shadow-[0_0_24px_rgba(0,240,255,0.25)] bg-[#051426]"
                : "border-slate-800/80 hover:border-slate-700 opacity-60 hover:opacity-100"
            }`}
          >
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-mono font-bold tracking-wider text-[#00f0ff] uppercase">
                VERIFIED GEOMETRY
              </span>
              <span className="text-[10px] font-mono text-slate-500">STAGE 03</span>
            </div>
            <p className="text-xs font-mono text-slate-400">Calibrated vector structures resolved.</p>
          </div>
        </div>

        {/* Bottom Callout matching Reference Panel 04 */}
        <div className="text-center pt-6">
          <p className="text-sm sm:text-base font-sans text-slate-300">
            From pixels to{" "}
            <span className="font-bold text-[#00f0ff] drop-shadow-[0_0_20px_rgba(0,240,255,0.5)]">
              meaningful, actionable intelligence.
            </span>
          </p>
        </div>
      </div>
    </section>
  );
}
