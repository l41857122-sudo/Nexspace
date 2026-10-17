"use client";

import React, { useState } from "react";
import Link from "next/link";
import { MapPin, ArrowRight, Layers, Map as MapIcon, Satellite } from "lucide-react";

export default function MapRevealScene() {
  const [viewMode, setViewMode] = useState<"satellite" | "hybrid" | "vector">("hybrid");

  return (
    <section
      id="map"
      className="relative min-h-screen w-full bg-[#020610] text-slate-100 py-24 px-4 sm:px-6 lg:px-12 flex flex-col justify-center items-center overflow-hidden border-t border-cyan-950/40"
    >
      {/* Background Ambience */}
      <div className="absolute top-1/3 right-1/4 w-[600px] h-[600px] bg-blue-600/10 rounded-full blur-[160px] pointer-events-none" />

      {/* Header */}
      <div className="max-w-4xl text-center mb-10 relative z-10">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-cyan-950/70 border border-cyan-500/30 text-cyan-300 font-mono text-xs tracking-wider mb-4 shadow-sm">
          <MapIcon className="w-3.5 h-3.5 text-cyan-400" />
          <span>STAGE 07 // GEOGRAPHIC REVOLUTION &amp; MORPH</span>
        </div>
        <h2 className="text-3xl sm:text-5xl font-extrabold tracking-tight text-white mb-4">
          Pixels Become <span className="text-cyan-400">Places.</span>
        </h2>
        <p className="text-slate-400 text-sm sm:text-base max-w-2xl mx-auto leading-relaxed">
          Transition from raw satellite pixels to real-world geography. Morph from sensor footprint
          envelopes to calibrated WGS-84 vector cartography and Google Maps coordinates.
        </p>
      </div>

      {/* Main Map & Intelligence Card Layout */}
      <div className="w-full max-w-6xl grid grid-cols-1 lg:grid-cols-12 gap-8 items-center relative z-10">
        {/* Left: Morphing Geographic Map Canvas (~60%) */}
        <div className="lg:col-span-7 relative aspect-[16/10] rounded-2xl overflow-hidden border border-cyan-500/40 shadow-[0_0_60px_rgba(0,212,255,0.12)] bg-[#071322]">
          {/* Zoomable Camera Canvas (Zooms into detected location as footprint resolves) */}
          <div
            className="absolute inset-0 w-full h-full transition-transform duration-700 ease-out"
            style={{
              transform: viewMode === "vector" ? "scale(1.08)" : viewMode === "hybrid" ? "scale(1.04)" : "scale(1.0)",
              transformOrigin: "54% 42%",
            }}
          >
            {/* Layer 1: Real Satellite Footprint Base */}
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src="/imagery/satellite_mumbai_port.jpg"
              alt="Real Satellite Footprint"
              className={`absolute inset-0 w-full h-full object-cover transition-opacity duration-700 ${viewMode === "vector" ? "opacity-0" : viewMode === "hybrid" ? "opacity-60" : "opacity-100"
                }`}
            />

            {/* Layer 2: Vector Cartography Overlay (Geographic streets, waterway contours, coastline) */}
            <div
              className={`absolute inset-0 transition-opacity duration-700 ${viewMode === "satellite" ? "opacity-0" : "opacity-100"
                }`}
            >
              {/* Geodetic Grid Lines */}
              <div
                className="absolute inset-0 opacity-40"
                style={{
                  backgroundImage:
                    "linear-gradient(rgba(0, 212, 255, 0.35) 1px, transparent 1px), linear-gradient(90deg, rgba(0, 212, 255, 0.35) 1px, transparent 1px)",
                  backgroundSize: "60px 60px",
                }}
              />

              {/* Coastline & Maritime Channels Vector Overlay */}
              <svg
                className="absolute inset-0 w-full h-full opacity-80 pointer-events-none"
                viewBox="0 0 800 500"
                preserveAspectRatio="none"
              >
                {/* Landmass Polygon */}
                <path
                  d="M 0,0 L 450,0 C 470,120 420,240 520,330 C 580,390 530,500 560,500 L 0,500 Z"
                  fill={viewMode === "vector" ? "#0a1d36" : "rgba(10, 29, 54, 0.45)"}
                  stroke="#00f0ff"
                  strokeWidth="1.5"
                />

                {/* Georeferenced Scene Footprint Bounding Envelope */}
                <rect
                  x="280"
                  y="110"
                  width="340"
                  height="240"
                  fill="rgba(0, 240, 255, 0.05)"
                  stroke="#00f0ff"
                  strokeWidth="2"
                  strokeDasharray="8 5"
                />

                {/* Footprint Corner Reticles */}
                <circle cx="280" cy="110" r="3" fill="#00f0ff" />
                <circle cx="620" cy="110" r="3" fill="#00f0ff" />
                <circle cx="280" cy="350" r="3" fill="#00f0ff" />
                <circle cx="620" cy="350" r="3" fill="#00f0ff" />
              </svg>
            </div>

            {/* Location Marker & Radiating Real-World Pulse */}
            <div className="absolute left-[54%] top-[42%] -translate-x-1/2 -translate-y-1/2 flex flex-col items-center z-20 pointer-events-none">
              <div className="relative">
                <span className="w-10 h-10 rounded-full bg-cyan-400/40 absolute -top-2 -left-2 animate-ping" />
                <div className="w-6 h-6 rounded-full bg-cyan-500 border-2 border-white shadow-[0_0_20px_#00f0ff] flex items-center justify-center">
                  <MapPin className="w-3.5 h-3.5 text-slate-950" />
                </div>
              </div>
              <div className="mt-2 px-2.5 py-1 rounded bg-slate-950/95 border border-cyan-400 text-xs font-mono font-bold text-cyan-300 shadow-2xl whitespace-nowrap flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
                <span>MUMBAI PORT COMPLEX // 18.9499° N, 72.9515° E</span>
              </div>
            </div>
          </div>

          {/* View Mode Morph Selector Buttons */}
          <div className="absolute top-3 right-3 flex items-center gap-1 bg-slate-950/90 backdrop-blur-md p-1 rounded-lg border border-cyan-500/30 z-20 font-mono text-[10px]">
            <button
              onClick={() => setViewMode("satellite")}
              className={`px-2 py-1 rounded flex items-center gap-1 transition-colors cursor-pointer ${viewMode === "satellite" ? "bg-cyan-500 text-slate-950 font-bold" : "text-slate-400 hover:text-white"
                }`}
            >
              <Satellite className="w-3 h-3" />
              <span>Satellite</span>
            </button>
            <button
              onClick={() => setViewMode("hybrid")}
              className={`px-2 py-1 rounded flex items-center gap-1 transition-colors cursor-pointer ${viewMode === "hybrid" ? "bg-cyan-500 text-slate-950 font-bold" : "text-slate-400 hover:text-white"
                }`}
            >
              <Layers className="w-3 h-3" />
              <span>Footprint</span>
            </button>
            <button
              onClick={() => setViewMode("vector")}
              className={`px-2 py-1 rounded flex items-center gap-1 transition-colors cursor-pointer ${viewMode === "vector" ? "bg-cyan-500 text-slate-950 font-bold" : "text-slate-400 hover:text-white"
                }`}
            >
              <MapIcon className="w-3 h-3" />
              <span>Map</span>
            </button>
          </div>

          {/* Map Telemetry Bar */}
          <div className="absolute bottom-3 left-3 right-3 flex items-center justify-between text-[10.5px] font-mono text-cyan-300 bg-slate-950/90 backdrop-blur-md px-3 py-1.5 rounded-lg border border-cyan-500/30 z-20">
            <span>CRS: EPSG:4326 (WGS 84)</span>
            <span>FOOTPRINT: 2.4 KM²</span>
            <span>ELEVATION: 12M MSL</span>
          </div>
        </div>

        {/* Right: Elegant Geographic Briefing Card (~40%) */}
        <div className="lg:col-span-5 bg-slate-950/90 backdrop-blur-xl border border-cyan-500/40 rounded-2xl p-6 sm:p-8 shadow-[0_0_40px_rgba(0,212,255,0.1)] flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between border-b border-cyan-500/20 pb-4 mb-5">
              <div>
                <span className="text-[10px] font-mono text-cyan-400 font-bold tracking-widest uppercase">
                  GEOREFERENCED FOOTPRINT
                </span>
                <h3 className="text-xl font-bold text-white tracking-tight mt-0.5">
                  Real-World Anchor
                </h3>
              </div>
              <span className="w-2.5 h-2.5 rounded-full bg-emerald-400 shadow-[0_0_10px_#34d399]" />
            </div>

            <div className="space-y-4 font-mono text-xs">
              <div>
                <span className="text-slate-400 block text-[10px] uppercase">Query Prompt</span>
                <span className="text-white text-sm font-sans font-medium">
                  &ldquo;Locate ships near the port&rdquo;
                </span>
              </div>

              <div>
                <span className="text-slate-400 block text-[10px] uppercase">Spatial Ground Truth</span>
                <span className="text-cyan-300 text-base font-bold">
                  6 Commercial Vessels Mapped to Docks
                </span>
              </div>

              <div className="grid grid-cols-2 gap-3 pt-2">
                <div className="p-2.5 rounded-lg bg-slate-900/80 border border-slate-800">
                  <span className="text-slate-400 block text-[9px] uppercase">Geographic Location</span>
                  <span className="text-slate-200 font-bold">Mumbai Port, India</span>
                </div>
                <div className="p-2.5 rounded-lg bg-slate-900/80 border border-slate-800">
                  <span className="text-slate-400 block text-[9px] uppercase">Coordinate System</span>
                  <span className="text-slate-200 font-bold">EPSG:4326 WGS 84</span>
                </div>
              </div>

              <div className="p-2.5 rounded-lg bg-slate-900/80 border border-slate-800">
                <span className="text-slate-400 block text-[9px] uppercase">Satellite Ingestion Source</span>
                <span className="text-slate-200 font-bold">Sentinel-2 / Cartosat-2S Calibrated Optical Pass</span>
              </div>
            </div>
          </div>

          <div className="mt-8 pt-4 border-t border-cyan-500/20 flex items-center gap-3">
            <Link
              href="/query"
              className="w-full inline-flex items-center justify-center gap-2 px-5 py-3 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold text-sm transition-all shadow-[0_0_20px_rgba(0,240,255,0.3)] hover:scale-[1.02]"
            >
              <span>Explore in Investigation Workspace</span>
              <ArrowRight className="w-4 h-4" />
            </Link>
          </div>
        </div>
      </div>
    </section>
  );
}
