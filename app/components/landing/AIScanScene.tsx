"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import { ArrowRight, Sparkles } from "lucide-react";

interface AIScanSceneProps {
  progress?: number;
  isReducedMotion?: boolean;
}

interface TargetItem {
  id: string;
  tag: string;
  label: string;
  coords: string;
  tier: string;
  type: "building" | "vehicle" | "road" | "region";
  xPct: number;
  yPct: number;
  wPx: number;
  hPx: number;
}

const TARGETS: TargetItem[] = [
  {
    id: "target-1",
    tag: "01",
    label: "01 BUILDING",
    coords: "26.8481° N, 80.9421° E",
    tier: "High Confidence",
    type: "building",
    xPct: 22,
    yPct: 26,
    wPx: 140,
    hPx: 95,
  },
  {
    id: "target-2",
    tag: "02",
    label: "02 VEHICLE",
    coords: "26.8465° N, 80.9455° E",
    tier: "High Confidence",
    type: "vehicle",
    xPct: 52,
    yPct: 48,
    wPx: 48,
    hPx: 28,
  },
  {
    id: "target-3",
    tag: "03",
    label: "03 ROAD",
    coords: "26.8440° N, 80.9400° E",
    tier: "Verified Linear",
    type: "road",
    xPct: 34,
    yPct: 68,
    wPx: 180,
    hPx: 32,
  },
  {
    id: "target-4",
    tag: "04",
    label: "04 WATER REGION",
    coords: "26.8510° N, 80.9520° E",
    tier: "Continuous Polygon",
    type: "region",
    xPct: 68,
    yPct: 16,
    wPx: 150,
    hPx: 110,
  },
];

export default function AIScanScene({ isReducedMotion = false }: AIScanSceneProps) {
  const [activeStep, setActiveStep] = useState(isReducedMotion ? 4 : 1);
  const [hoveredTarget, setHoveredTarget] = useState<string | null>(null);

  // Sequential target locking animation: One by one progression
  useEffect(() => {
    if (isReducedMotion) return;
    const interval = setInterval(() => {
      setActiveStep((prev) => (prev >= 4 ? 1 : prev + 1));
    }, 2800);
    return () => clearInterval(interval);
  }, [isReducedMotion]);

  // Image zoom factor progressing into analytical view
  const zoomScale = isReducedMotion ? 1.06 : 1.0 + activeStep * 0.035;

  return (
    <section
      id="analysis"
      className="relative min-h-screen w-full bg-[#02060d] text-slate-100 py-24 px-4 sm:px-6 lg:px-12 flex flex-col justify-center items-center overflow-hidden border-t border-cyan-950/40"
    >
      {/* Background Ambience */}
      <div className="absolute top-1/4 left-1/2 -translate-x-1/2 w-[700px] h-[700px] bg-cyan-600/5 rounded-full blur-[140px] pointer-events-none" />

      {/* Header */}
      <div className="max-w-4xl text-center mb-12 relative z-10">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-cyan-950/60 border border-cyan-500/30 text-cyan-300 font-mono text-xs tracking-wider mb-4 shadow-sm">
          <Sparkles className="w-3.5 h-3.5 text-cyan-400 animate-pulse" />
          <span>STAGE 04 // SPATIAL ANALYSIS &amp; TARGET LOCKING</span>
        </div>
        <h2 className="text-3xl sm:text-5xl font-extrabold tracking-tight text-white mb-4">
          Pixels Become <span className="text-cyan-400">Classified Geometry</span>
        </h2>
        <p className="text-slate-400 text-sm sm:text-base max-w-2xl mx-auto leading-relaxed">
          Camera smoothly zooms into the region of interest. Targets lock sequentially with
          mathematical spatial bounding, land-cover polygons, and linear transportation corridors.
        </p>
      </div>

      {/* Main Interactive Analytical Canvas */}
      <div className="relative w-full max-w-5xl aspect-[16/10] rounded-xl overflow-hidden border border-cyan-500/30 shadow-[0_0_50px_rgba(0,212,255,0.1)] bg-slate-950">
        {/* Real Satellite Base Layer with Smooth Analytical Zoom */}
        <div
          className="absolute inset-0 w-full h-full transition-transform duration-1000 ease-out"
          style={{ transform: `scale(${zoomScale})` }}
        >
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src="/imagery/satellite_infrastructure_roads.jpg"
            alt="Real Satellite Observation Target Area"
            className="w-full h-full object-cover filter brightness-[0.88] contrast-[1.08]"
          />
        </div>

        {/* Dynamic Subtle Scanning Laser Leading Edge */}
        <div
          className="absolute top-0 bottom-0 pointer-events-none transition-all duration-1000 ease-in-out"
          style={{
            left: `${activeStep === 1 ? 28 : activeStep === 2 ? 55 : activeStep === 3 ? 42 : 78}%`,
            width: "100px",
            transform: "translateX(-50%)",
            background:
              "linear-gradient(90deg, transparent 0%, rgba(0, 240, 255, 0.05) 50%, rgba(0, 240, 255, 0.35) 96%, #ffffff 100%)",
            boxShadow: "0 0 25px rgba(0,240,255,0.3)",
          }}
        >
          <div className="absolute right-0 top-0 bottom-0 w-[1.5px] bg-cyan-300 shadow-[0_0_12px_#00f0ff]" />
        </div>

        {/* SVG Geometry Layer (Bounding Boxes, Polygons, Linear Features) */}
        <svg
          className="absolute inset-0 w-full h-full pointer-events-none transition-transform duration-1000 ease-out"
          viewBox="0 0 1000 625"
          preserveAspectRatio="none"
          style={{ transform: `scale(${zoomScale})` }}
        >
          <defs>
            <filter id="cyan-glow" x="-20%" y="-20%" width="140%" height="140%">
              <feGaussianBlur stdDeviation="3" result="blur" />
              <feComposite in="SourceGraphic" in2="blur" operator="over" />
            </filter>
          </defs>

          {/* 1. DISCRETE OBJECT: Building 01 (Locks at Step >= 1) */}
          <g className={`transition-opacity duration-700 ${activeStep >= 1 ? "opacity-100" : "opacity-0"}`}>
            <rect
              x="220"
              y="180"
              width="140"
              height="95"
              fill={hoveredTarget === "target-1" ? "rgba(0, 240, 255, 0.22)" : "rgba(0, 240, 255, 0.08)"}
              stroke="#00f0ff"
              strokeWidth={hoveredTarget === "target-1" ? "2.5" : "2"}
              filter="url(#cyan-glow)"
              strokeDasharray={activeStep === 1 ? "6 4" : "none"}
              className="transition-all duration-300"
            />
          </g>

          {/* 2. DISCRETE OBJECT: Vehicle 02 (Locks at Step >= 2) */}
          <g className={`transition-opacity duration-700 ${activeStep >= 2 ? "opacity-100" : "opacity-0"}`}>
            <rect
              x="520"
              y="320"
              width="48"
              height="28"
              fill={hoveredTarget === "target-2" ? "rgba(16, 185, 129, 0.3)" : "rgba(16, 185, 129, 0.12)"}
              stroke="#10b981"
              strokeWidth={hoveredTarget === "target-2" ? "2.5" : "1.8"}
              className="transition-all duration-300"
            />
          </g>

          {/* 3. LINEAR FEATURE: Arterial Road Corridor (Traces at Step >= 3) */}
          <g className={`transition-opacity duration-700 ${activeStep >= 3 ? "opacity-100" : "opacity-0"}`}>
            <path
              d="M 50,440 Q 320,380 620,410 T 960,370"
              fill="none"
              stroke="#38bdf8"
              strokeWidth={hoveredTarget === "target-3" ? "5" : "3.5"}
              strokeDasharray="12 6"
              filter="url(#cyan-glow)"
              className="transition-all duration-300"
            />
          </g>

          {/* 4. REGION FEATURE: Reservoir / Water Body Polygon (Locks at Step >= 4) */}
          <g className={`transition-opacity duration-700 ${activeStep >= 4 ? "opacity-100" : "opacity-0"}`}>
            <polygon
              points="660,120 840,110 920,240 850,340 710,290 640,200"
              fill={hoveredTarget === "target-4" ? "rgba(6, 182, 212, 0.3)" : "rgba(6, 182, 212, 0.14)"}
              stroke="#22d3ee"
              strokeWidth={hoveredTarget === "target-4" ? "3" : "2"}
              strokeDasharray="8 4"
              filter="url(#cyan-glow)"
              className="transition-all duration-300"
            />
          </g>
        </svg>

        {/* HTML Labels & Telemetry Badges (With Expandable Hover Badge, Zero Layout Shift) */}
        <div className="absolute inset-0 pointer-events-none p-4 font-mono select-none">
          {TARGETS.map((target, idx) => {
            const stepNum = idx + 1;
            const isUnlocked = activeStep >= stepNum;
            const isHovered = hoveredTarget === target.id;

            return (
              <div
                key={target.id}
                onMouseEnter={() => setHoveredTarget(target.id)}
                onMouseLeave={() => setHoveredTarget(null)}
                className={`absolute pointer-events-auto cursor-pointer transition-all duration-500 ease-out ${isUnlocked ? "opacity-100 translate-y-0" : "opacity-0 translate-y-2 pointer-events-none"
                  }`}
                style={{
                  left: `${target.xPct}%`,
                  top: `${target.yPct}%`,
                }}
              >
                {/* Compact Label Tag (Expands cleanly on hover without layout shift) */}
                <div
                  className={`relative px-2.5 py-1 rounded-md bg-slate-950/95 border transition-all duration-300 shadow-lg flex flex-col gap-0.5 whitespace-nowrap ${isHovered
                    ? "border-cyan-400 shadow-[0_0_20px_rgba(0,240,255,0.4)] scale-105 z-30"
                    : "border-cyan-500/50 hover:border-cyan-400 z-10"
                    }`}
                >
                  <div className="flex items-center gap-1.5">
                    <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-ping" />
                    <span className="text-[10px] font-bold text-cyan-300 tracking-wider">
                      {target.label}
                    </span>
                  </div>

                  {/* Expanded Hover Detail (Coordinates & Confidence) */}
                  <div
                    className={`overflow-hidden transition-all duration-300 text-[8.5px] font-mono ${isHovered ? "max-h-12 opacity-100 mt-0.5 pt-0.5 border-t border-cyan-500/30" : "max-h-0 opacity-0"
                      }`}
                  >
                    <div className="text-slate-300">{target.coords}</div>
                    <div className="text-emerald-400 font-semibold">{target.tier}</div>
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        {/* Live Status Bar */}
        <div className="absolute bottom-3 left-3 right-3 bg-slate-950/90 backdrop-blur-md border border-cyan-500/30 rounded-lg px-4 py-2.5 flex flex-wrap items-center justify-between gap-3 text-xs font-mono">
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-cyan-400 animate-ping" />
            <span className="text-cyan-300 font-bold">
              {activeStep === 1
                ? "LOCKING TARGET 01 // BUILDING..."
                : activeStep === 2
                  ? "LOCKING TARGET 02 // VEHICLE..."
                  : activeStep === 3
                    ? "TRACING TARGET 03 // ROAD CORRIDOR..."
                    : "RESOLVING TARGET 04 // WATER REGION..."}
            </span>
          </div>
          <div className="flex items-center gap-4 text-slate-400 text-[11px]">
            <span>LOCK STEP: <strong className="text-white">{activeStep} / 4</strong></span>
            <span>ZOOM: <strong className="text-white">{zoomScale.toFixed(2)}x</strong></span>
            <span>TARGETS: <strong className="text-white">4 RESOLVED</strong></span>
          </div>
        </div>
      </div>

      {/* CTA Footer */}
      <div className="mt-8 flex flex-col sm:flex-row items-center gap-4 z-10">
        <Link
          href="/query"
          className="inline-flex items-center gap-2 px-6 py-3 rounded-lg bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-semibold text-sm transition-all shadow-[0_0_25px_rgba(0,240,255,0.3)] hover:scale-105"
        >
          <span>Run Analysis in Investigation Workspace</span>
          <ArrowRight className="w-4 h-4" />
        </Link>
      </div>
    </section>
  );
}
