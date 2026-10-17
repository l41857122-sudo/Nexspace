"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import { Search, ZoomIn, ArrowRight } from "lucide-react";

interface ScaleLevel {
  id: string;
  label: string;
  extent: string;
  zoom: number;
  originX: number; // percentage
  originY: number;
  description: string;
  targetCount: string;
  resolution: string;
}

const SCALE_LEVELS: ScaleLevel[] = [
  {
    id: "full",
    label: "FULL SCENE",
    extent: "10 KM AOI",
    zoom: 1.0,
    originX: 50,
    originY: 50,
    description: "Synoptic satellite pass over maritime entrance & regional transportation corridors.",
    targetCount: "28 Candidates",
    resolution: "10m Multispectral",
  },
  {
    id: "block",
    label: "CITY BLOCK",
    extent: "2 KM ZONE",
    zoom: 2.2,
    originX: 42,
    originY: 38,
    description: "Maritime terminal basin, breakwaters, and arterial logistics corridors resolved.",
    targetCount: "12 Primary Zones",
    resolution: "2.5m Panchromatic",
  },
  {
    id: "cluster",
    label: "FACILITY CLUSTER",
    extent: "500 M BERTH",
    zoom: 4.6,
    originX: 35,
    originY: 30,
    description: "Active container berths, gantry crane tracks, and vessel mooring alignment.",
    targetCount: "6 Bounded Objects",
    resolution: "0.5m Calibrated GSD",
  },
  {
    id: "target",
    label: "INDIVIDUAL TARGET",
    extent: "50 M OBJECT",
    zoom: 8.0,
    originX: 34,
    originY: 28,
    description: "Discrete commercial cargo vessel: 182m length, heading 142°, verified evidence.",
    targetCount: "TARGET 01 VERIFIED",
    resolution: "Sub-meter Precision",
  },
];

export default function QueryScene() {
  const [scaleIdx, setScaleIdx] = useState(0);
  const currentScale = SCALE_LEVELS[scaleIdx];

  // Auto-cycle through spatial zoom scales if user doesn't manually click
  useEffect(() => {
    const interval = setInterval(() => {
      setScaleIdx((prev) => (prev + 1) % SCALE_LEVELS.length);
    }, 4000);
    return () => clearInterval(interval);
  }, []);

  return (
    <section
      id="query"
      className="relative min-h-screen w-full bg-[#030814] text-slate-100 py-24 px-4 sm:px-6 lg:px-12 flex flex-col justify-center items-center overflow-hidden border-t border-cyan-950/40"
    >
      {/* Deep Space Atmosphere Glow */}
      <div className="absolute -top-40 -left-40 w-96 h-96 bg-blue-600/10 rounded-full blur-[160px] pointer-events-none" />
      <div className="absolute -bottom-40 -right-40 w-96 h-96 bg-cyan-600/10 rounded-full blur-[160px] pointer-events-none" />

      {/* Header */}
      <div className="max-w-4xl text-center mb-10 relative z-10">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-cyan-950/70 border border-cyan-500/30 text-cyan-300 font-mono text-xs tracking-wider mb-4 shadow-sm">
          <ZoomIn className="w-3.5 h-3.5 text-cyan-400" />
          <span>STAGE 05 // MULTI-SCALE SPATIAL RESOLUTION</span>
        </div>
        <h2 className="text-3xl sm:text-5xl font-extrabold tracking-tight text-white mb-4">
          Zooming Through <span className="text-cyan-400">Spatial Scales</span>
        </h2>
        <p className="text-slate-400 text-sm sm:text-base max-w-2xl mx-auto leading-relaxed">
          From a 10-kilometer synoptic satellite pass down to a 50-meter individual vessel. NexSpace
          seamlessly deconstructs natural language queries into verified spatial evidence.
        </p>
      </div>

      {/* Main Interactive Multi-Scale Explorer */}
      <div className="w-full max-w-6xl grid grid-cols-1 lg:grid-cols-12 gap-6 items-center relative z-10">
        {/* Left: Dynamic Multi-Scale Satellite Canvas (~65%) */}
        <div className="lg:col-span-8 relative aspect-[16/10] rounded-2xl overflow-hidden border border-cyan-500/40 shadow-[0_0_60px_rgba(0,212,255,0.1)] bg-slate-950">
          {/* Real Satellite Base Layer with Smooth Transform Zoom to Selected Scale */}
          <div
            className="absolute inset-0 w-full h-full transition-transform duration-700 ease-out"
            style={{
              transform: `scale(${currentScale.zoom})`,
              transformOrigin: `${currentScale.originX}% ${currentScale.originY}%`,
            }}
          >
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src="/imagery/satellite_mumbai_port.jpg"
              alt="Real Satellite Observation of Port Infrastructure"
              className="w-full h-full object-cover filter brightness-[0.92] contrast-[1.08]"
            />

            {/* Target Reticle at Vessel Position */}
            <div className="absolute left-[34%] top-[28%] w-12 h-12 -translate-x-1/2 -translate-y-1/2 border-2 border-cyan-400 rounded-sm shadow-[0_0_20px_#00f0ff] pointer-events-none">
              <span className="w-2 h-2 rounded-full bg-cyan-400 absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 animate-ping" />
              <div className="absolute -top-5 -left-1 px-1.5 py-0.5 rounded bg-slate-950/90 border border-cyan-400 text-[8px] font-mono text-cyan-300 font-bold whitespace-nowrap">
                TARGET 01
              </div>
            </div>
          </div>

          {/* Coordinate Reticle Overlay */}
          <div
            className="absolute inset-0 opacity-15 pointer-events-none"
            style={{
              backgroundImage:
                "linear-gradient(rgba(0, 240, 255, 0.4) 1px, transparent 1px), linear-gradient(90deg, rgba(0, 240, 255, 0.4) 1px, transparent 1px)",
              backgroundSize: "60px 60px",
            }}
          />

          {/* Scale Indicator Telemetry HUD */}
          <div className="absolute top-3 left-3 bg-slate-950/90 backdrop-blur-md border border-cyan-500/30 rounded-lg px-3.5 py-2 font-mono text-xs flex items-center gap-3">
            <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
            <span className="text-white font-bold">{currentScale.label}</span>
            <span className="text-cyan-400 font-bold">[{currentScale.extent}]</span>
            <span className="text-slate-400 hidden sm:inline">• {currentScale.resolution}</span>
          </div>

          <div className="absolute bottom-3 left-3 right-3 bg-slate-950/90 backdrop-blur-md border border-cyan-500/30 rounded-lg px-4 py-2 font-mono text-xs text-slate-300 flex items-center justify-between">
            <span className="text-cyan-300 font-medium">{currentScale.description}</span>
            <span className="text-white font-bold bg-cyan-950/80 px-2 py-0.5 rounded border border-cyan-400/40 text-[10.5px]">
              {currentScale.targetCount}
            </span>
          </div>
        </div>

        {/* Right: Scale Selector Controls & Query Deconstruction (~35%) */}
        <div className="lg:col-span-4 flex flex-col gap-3.5">
          {/* Natural Language Prompt Pill */}
          <div className="p-3.5 rounded-xl bg-slate-950/90 border border-cyan-500/30 shadow-md">
            <div className="flex items-center gap-2 text-cyan-400 font-mono text-[10px] uppercase mb-1">
              <Search className="w-3 h-3" />
              <span>NATURAL LANGUAGE QUERY</span>
            </div>
            <div className="text-white font-sans text-sm font-semibold tracking-tight">
              &ldquo;Locate commercial ships near Mumbai port docks&rdquo;
            </div>
          </div>

          {/* Interactive Scale Stepper Buttons */}
          <div className="space-y-2">
            <div className="text-slate-400 font-mono text-[10px] tracking-wider uppercase px-1">
              SELECT SPATIAL SCALE:
            </div>
            {SCALE_LEVELS.map((level, idx) => {
              const isSel = idx === scaleIdx;
              return (
                <button
                  key={level.id}
                  onClick={() => setScaleIdx(idx)}
                  className={`w-full text-left p-3 rounded-xl border transition-all duration-300 cursor-pointer flex items-center justify-between ${isSel
                    ? "bg-cyan-950/70 border-cyan-400 text-white shadow-[0_0_20px_rgba(0,240,255,0.25)] translate-x-1"
                    : "bg-slate-950/50 border-slate-800/80 hover:border-cyan-500/40 text-slate-400 hover:text-slate-200"
                    }`}
                >
                  <div className="flex items-center gap-3">
                    <span
                      className={`w-6 h-6 rounded-lg font-mono text-xs font-bold flex items-center justify-center ${isSel ? "bg-cyan-400 text-slate-950" : "bg-slate-900 text-slate-400"
                        }`}
                    >
                      {idx + 1}
                    </span>
                    <div>
                      <div className="font-mono text-xs font-bold tracking-wide">{level.label}</div>
                      <div className="text-[10px] text-slate-400">{level.extent}</div>
                    </div>
                  </div>
                  <span
                    className={`text-[10px] font-mono font-bold px-2 py-0.5 rounded border ${isSel
                      ? "text-cyan-300 border-cyan-500/40 bg-cyan-950/50"
                      : "text-slate-500 border-slate-800 bg-slate-900/40"
                      }`}
                  >
                    {level.zoom}x
                  </span>
                </button>
              );
            })}
          </div>

          {/* Execution Action */}
          <Link
            href="/query"
            className="w-full inline-flex items-center justify-center gap-2 p-3 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold text-xs transition-all shadow-[0_0_20px_rgba(0,240,255,0.25)] hover:scale-[1.02] mt-1 cursor-pointer"
          >
            <span>Investigate Multi-Scale Imagery</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </Link>
        </div>
      </div>
    </section>
  );
}
