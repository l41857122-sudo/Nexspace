"use client";

import React, { useState, useEffect, useRef } from "react";
import Link from "next/link";
import {
  Search,
  Layers,
  Edit3,
  Crosshair,
  Sliders,
  Plus,
  Minus,
  ArrowRight,
  Maximize2,
  CheckCircle2,
  Calendar,
  Cloud,
  Compass,
} from "lucide-react";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";

interface LocationPreset {
  id: string;
  name: string;
  query: string;
  coords: string;
  satellite: string;
  date: string;
  resolution: string;
  cloudCover: string;
  area: string;
  image: string;
  pinX: number; // percentage
  pinY: number;
}

const PRESETS: LocationPreset[] = [
  {
    id: "lucknow",
    name: "Indira Nagar, Lucknow",
    query: "Indira Nagar, Lucknow",
    coords: "26.8742° N, 80.9917° E",
    satellite: "Sentinel-2 (L2A)",
    date: "12 Mar 2024",
    resolution: "10 m / pixel",
    cloudCover: "2.4%",
    area: "12.5 km²",
    image: "/imagery/satellite_urban_surface.jpg",
    pinX: 52,
    pinY: 44,
  },
  {
    id: "mumbai",
    name: "Mumbai Port",
    query: "Mumbai Port, India",
    coords: "18.9482° N, 72.8441° E",
    satellite: "Sentinel-2 (L2A)",
    date: "08 Apr 2024",
    resolution: "10 m / pixel",
    cloudCover: "1.1%",
    area: "18.2 km²",
    image: "/imagery/satellite_mumbai_port.jpg",
    pinX: 48,
    pinY: 50,
  },
  {
    id: "amazon",
    name: "Amazon Rainforest",
    query: "Amazon Rainforest Basin",
    coords: "3.4653° S, 62.2159° W",
    satellite: "Sentinel-2 (L2A)",
    date: "28 Feb 2024",
    resolution: "10 m / pixel",
    cloudCover: "3.8%",
    area: "45.0 km²",
    image: "/imagery/satellite_vegetation_canopy.jpg",
    pinX: 54,
    pinY: 42,
  },
  {
    id: "tokyo",
    name: "Tokyo Bay",
    query: "Tokyo Bay, Japan",
    coords: "35.6190° N, 139.7798° E",
    satellite: "Sentinel-2 (L2A)",
    date: "19 Mar 2024",
    resolution: "10 m / pixel",
    cloudCover: "0.8%",
    area: "24.6 km²",
    image: "/imagery/satellite_infrastructure_roads.jpg",
    pinX: 50,
    pinY: 46,
  },
  {
    id: "newyork",
    name: "New York Harbor",
    query: "New York Harbor, USA",
    coords: "40.6892° N, 74.0445° W",
    satellite: "Sentinel-2 (L2A)",
    date: "15 Apr 2024",
    resolution: "10 m / pixel",
    cloudCover: "1.9%",
    area: "16.8 km²",
    image: "/imagery/satellite_coastline_water.jpg",
    pinX: 46,
    pinY: 48,
  },
];

export default function LiveInvestigationScene() {
  const containerRef = useRef<HTMLDivElement>(null);
  const [selectedPreset, setSelectedPreset] = useState<LocationPreset>(PRESETS[0]);
  const [inputVal, setInputVal] = useState("Indira Nagar, Lucknow");
  const [zoomLevel, setZoomLevel] = useState(1);
  const [activeTool, setActiveTool] = useState("target");
  const [scrollProgress, setScrollProgress] = useState(0);

  useEffect(() => {
    gsap.registerPlugin(ScrollTrigger);

    const el = containerRef.current;
    if (!el) return;

    const isMobile = window.innerWidth < 1024;
    const isReduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    if (isReduced || isMobile) {
      setScrollProgress(1);
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
      },
    });

    return () => {
      st.kill();
    };
  }, []);

  const handleSelectPreset = (preset: LocationPreset) => {
    setInputVal("");
    let i = 0;
    const interval = setInterval(() => {
      if (i < preset.query.length) {
        setInputVal(preset.query.slice(0, i + 1));
        i++;
      } else {
        clearInterval(interval);
        setSelectedPreset(preset);
      }
    }, 28);
  };

  // Progression phases driven by scroll (Requirements 8 & 9):
  // p >= 0.0: STAGE A (Search field)
  // p >= 0.18: STAGE B (Location Resolved)
  // p >= 0.35: STAGE C/D (Map enters from depth, satellite resolves)
  // p >= 0.50: STAGE E (AOI polygon draws)
  // p >= 0.65: STAGE F (Location marker appears)
  // p >= 0.78: STAGE G (Metadata panel emerges)
  // p >= 0.88: STAGE H (Cinematic Map Zoom toward AOI 1.0 -> 1.55x)
  const isSearchActive = true;
  const isLocationResolved = scrollProgress >= 0.16;
  const isMapEntered = scrollProgress >= 0.30;
  const isAOIDrawn = scrollProgress >= 0.48;
  const isMarkerActive = scrollProgress >= 0.62;
  const isMetadataActive = scrollProgress >= 0.76;

  // Dynamic zoom multiplier: if user zoomed manually or scroll cinematic zoom kicks in
  const scrollZoom =
    scrollProgress > 0.85 ? 1.0 + (scrollProgress - 0.85) * 3.6 : 1.0;
  const effectiveZoom = zoomLevel * scrollZoom;

  return (
    <section
      ref={containerRef}
      id="investigation"
      data-spatial="true"
      className="relative min-h-screen w-full bg-[#010308] text-slate-100 py-20 px-6 sm:px-10 lg:px-14 flex flex-col justify-center overflow-hidden border-t border-cyan-950/30 select-none"
    >
      {/* Background Subtle Radial Lighting */}
      <div className="absolute top-1/3 left-1/3 w-[800px] h-[800px] bg-cyan-600/5 rounded-full blur-[180px] pointer-events-none" />

      <div className="max-w-[1440px] mx-auto w-full relative z-10 flex flex-col justify-between">
        {/* Section Header Row matching Reference Panel 03 */}
        <div className="flex items-center justify-between pb-3 border-b border-slate-800/60 mb-6">
          <div className="flex items-center gap-3">
            <span className="text-xs font-mono font-bold text-[#00f0ff] tracking-[0.25em] uppercase">
              03
            </span>
            <span className="text-slate-600 font-mono text-xs">/</span>
            <span className="text-xs font-mono tracking-[0.2em] text-slate-400 uppercase">
              0:08 — 0:16 LIVE INVESTIGATION
            </span>
          </div>

          <div className="hidden sm:flex items-center gap-2 font-mono text-xs text-[#00f0ff]">
            <span className="w-2 h-2 rounded-full bg-[#00f0ff] animate-ping" />
            <span className="tracking-widest uppercase">
              {isLocationResolved ? "AOI GEOMETRY LOCKED" : "SYNCHRONIZING ORBITAL SENSORS"}
            </span>
          </div>
        </div>

        {/* Section Title & Subcopy */}
        <div className="max-w-3xl mb-6">
          <h2 className="text-3xl sm:text-5xl lg:text-6xl font-extrabold tracking-tight text-white mb-2 leading-tight">
            Zooming Through <br className="hidden sm:block" />
            <span className="text-white">Spatial Scales</span>
          </h2>
          <p className="text-slate-400 text-xs sm:text-sm leading-relaxed font-light">
            Enter any location and explore real satellite imagery, AI analysis and geographical context
            in seconds.
          </p>
        </div>

        {/* Search Bar & Quick Switcher Tags (STAGE A & B) */}
        <div className="max-w-2xl mb-6">
          <form
            onSubmit={(e) => {
              e.preventDefault();
              window.location.href = `/query?prompt=${encodeURIComponent(inputVal)}`;
            }}
            className="flex items-center gap-2 p-1.5 rounded-2xl bg-[#040c18] border border-cyan-500/40 shadow-[0_0_30px_rgba(0,0,0,0.8)] backdrop-blur-xl focus-within:border-cyan-400 transition-all mb-3"
          >
            <input
              type="text"
              value={inputVal}
              onChange={(e) => setInputVal(e.target.value)}
              className="flex-1 bg-transparent px-4 py-1.5 text-sm text-white focus:outline-none font-mono placeholder-slate-500"
              placeholder="Enter location to investigate..."
            />
            <button
              type="submit"
              data-cursor="interactive"
              className="w-10 h-10 rounded-xl bg-[#00f0ff] hover:bg-[#38f5ff] text-[#020712] flex items-center justify-center transition-all shadow-[0_0_16px_rgba(0,240,255,0.6)] cursor-pointer hover:scale-105 active:scale-95 shrink-0"
              aria-label="Search location"
            >
              <Search className="w-4 h-4 stroke-[2.2]" />
            </button>
          </form>

          {/* Quick Try Tags */}
          <div className="flex flex-wrap items-center gap-2 text-xs font-mono text-slate-400">
            <span className="text-slate-500 text-[11px]">Try:</span>
            {PRESETS.map((preset) => (
              <button
                key={preset.id}
                type="button"
                data-cursor="interactive"
                onClick={() => handleSelectPreset(preset)}
                className={`px-2.5 py-0.5 rounded-lg border text-[10.5px] transition-all cursor-pointer ${
                  selectedPreset.id === preset.id
                    ? "bg-cyan-950/70 border-cyan-400/80 text-[#00f0ff] font-bold"
                    : "bg-[#061222]/80 border-slate-800 text-slate-300 hover:border-slate-700 hover:text-white"
                }`}
              >
                {preset.name}
              </button>
            ))}
          </div>
        </div>

        {/* Live Spatial Investigation Interface (STAGE C -> H Pinned Sequence & Map Zoom Transition) */}
        <div
          className={`relative rounded-2xl overflow-hidden border border-cyan-500/35 bg-[#030a16] shadow-[0_0_50px_rgba(0,0,0,0.9)] aspect-[16/9] max-h-[540px] flex items-stretch transition-all duration-700 ${
            isMapEntered ? "opacity-100 scale-100" : "opacity-40 scale-95"
          }`}
        >
          {/* Left Tool Rail */}
          <div className="w-12 sm:w-14 bg-[#020813]/90 border-r border-slate-800/80 p-2 flex flex-col justify-between items-center z-20 backdrop-blur-md">
            <div className="flex flex-col gap-2.5">
              {[
                { id: "layers", icon: Layers, label: "Layers" },
                { id: "draw", icon: Edit3, label: "Draw AOI" },
                { id: "target", icon: Crosshair, label: "Target" },
                { id: "filters", icon: Sliders, label: "Filter" },
              ].map((tool) => {
                const ToolIcon = tool.icon;
                const isActive = activeTool === tool.id;
                return (
                  <button
                    key={tool.id}
                    onClick={() => setActiveTool(tool.id)}
                    title={tool.label}
                    data-cursor="interactive"
                    className={`w-8 h-8 rounded-lg flex items-center justify-center transition-all cursor-pointer ${
                      isActive
                        ? "bg-[#00f0ff]/20 text-[#00f0ff] border border-cyan-400/60 shadow-[0_0_12px_rgba(0,240,255,0.3)]"
                        : "text-slate-400 hover:text-white hover:bg-slate-800/60"
                    }`}
                  >
                    <ToolIcon className="w-3.5 h-3.5" />
                  </button>
                );
              })}
            </div>

            {/* Zoom Controls */}
            <div className="flex flex-col gap-1.5 pt-2 border-t border-slate-800 w-full items-center">
              <button
                onClick={() => setZoomLevel((prev) => Math.min(prev + 0.25, 2.0))}
                data-cursor="interactive"
                className="w-7 h-7 rounded-lg bg-slate-900 border border-slate-800 hover:border-slate-700 text-slate-300 hover:text-white flex items-center justify-center transition-all"
                aria-label="Zoom in"
              >
                <Plus className="w-3 h-3" />
              </button>
              <button
                onClick={() => setZoomLevel((prev) => Math.max(prev - 0.25, 0.75))}
                data-cursor="interactive"
                className="w-7 h-7 rounded-lg bg-slate-900 border border-slate-800 hover:border-slate-700 text-slate-300 hover:text-white flex items-center justify-center transition-all"
                aria-label="Zoom out"
              >
                <Minus className="w-3 h-3" />
              </button>
            </div>
          </div>

          {/* Centered Map Canvas Viewport with High-Res Satellite Capture */}
          <div className="relative flex-1 overflow-hidden">
            {/* Satellite Image Background with cinematic zoom scale */}
            <div
              className="absolute inset-0 w-full h-full transition-all duration-700 ease-out"
              style={{
                transform: `scale(${effectiveZoom})`,
                transformOrigin: `${selectedPreset.pinX}% ${selectedPreset.pinY}%`,
              }}
            >
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={selectedPreset.image}
                alt={selectedPreset.name}
                className="w-full h-full object-cover transition-all duration-700"
                style={{
                  filter: isMapEntered ? "contrast(110%) brightness(95%)" : "blur(4px) brightness(60%)",
                }}
              />

              {/* Cartographic Coordinate Overlay Grid */}
              <div
                className="absolute inset-0 opacity-20 pointer-events-none"
                style={{
                  backgroundImage:
                    "linear-gradient(to right, rgba(0,240,255,0.18) 1px, transparent 1px), linear-gradient(to bottom, rgba(0,240,255,0.18) 1px, transparent 1px)",
                  backgroundSize: "80px 80px",
                }}
              />

              {/* Illuminated Cyan AOI Bounding Polygon (STAGE E) */}
              <div
                onClick={() => setZoomLevel((prev) => (prev > 1.1 ? 1 : 1.4))}
                data-cursor="interactive"
                title="Click to focus AOI"
                className={`absolute border-2 border-[#00f0ff] bg-[#00f0ff]/15 hover:bg-[#00f0ff]/25 shadow-[0_0_30px_rgba(0,240,255,0.45)] hover:shadow-[0_0_45px_rgba(0,240,255,0.85)] transition-all duration-700 cursor-pointer group ${
                  isAOIDrawn ? "opacity-100 scale-100" : "opacity-0 scale-90"
                }`}
                style={{
                  left: `${selectedPreset.pinX - 16}%`,
                  top: `${selectedPreset.pinY - 14}%`,
                  width: "32%",
                  height: "36%",
                }}
              >
                {/* Corner crosshairs */}
                <div className="absolute -top-1.5 -left-1.5 w-3 h-3 border-t-2 border-l-2 border-white group-hover:scale-125 transition-transform" />
                <div className="absolute -top-1.5 -right-1.5 w-3 h-3 border-t-2 border-r-2 border-white group-hover:scale-125 transition-transform" />
                <div className="absolute -bottom-1.5 -left-1.5 w-3 h-3 border-b-2 border-l-2 border-white group-hover:scale-125 transition-transform" />
                <div className="absolute -bottom-1.5 -right-1.5 w-3 h-3 border-b-2 border-r-2 border-white group-hover:scale-125 transition-transform" />

                {/* Subtle Hover Tooltip inside AOI */}
                <div className="absolute bottom-2 left-1/2 -translate-x-1/2 opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none px-2 py-0.5 rounded bg-[#020712]/95 border border-[#00f0ff] text-[9px] font-mono text-[#00f0ff] uppercase whitespace-nowrap shadow-[0_0_12px_#00f0ff]">
                  AOI // CLICK TO {zoomLevel > 1.1 ? "RESET" : "FOCUS"}
                </div>
              </div>

              {/* Pin Badge on Map (STAGE F) */}
              <div
                className={`absolute -translate-x-1/2 -translate-y-full flex flex-col items-center pointer-events-none transition-all duration-700 ${
                  isMarkerActive ? "opacity-100 translate-y-0" : "opacity-0 -translate-y-4"
                }`}
                style={{
                  left: `${selectedPreset.pinX}%`,
                  top: `${selectedPreset.pinY}%`,
                }}
              >
                <div className="px-3.5 py-1.5 rounded-lg bg-[#020712]/95 border border-[#00f0ff] shadow-[0_0_16px_rgba(0,240,255,0.7)] backdrop-blur-md flex items-center gap-2 mb-1">
                  <span className="w-2 h-2 rounded-full bg-[#00f0ff] animate-ping" />
                  <span className="text-xs font-mono font-bold text-white tracking-wide">
                    {selectedPreset.name}
                  </span>
                </div>
                <div className="w-2 h-2 rotate-45 bg-[#00f0ff] shadow-[0_0_8px_#00f0ff]" />
              </div>
            </div>

            {/* Top Right Floating Coordinate Readout */}
            <div className="absolute top-3 left-3 z-10 px-3 py-1 rounded-lg bg-[#020712]/85 border border-slate-800 text-[10px] font-mono text-cyan-400 backdrop-blur-md">
              <span>WGS-84: </span>
              <span className="text-white font-bold">{selectedPreset.coords}</span>
            </div>
          </div>

          {/* Right Floating "Location Resolved" Telemetry Panel (STAGE G) */}
          <div
            className={`hidden lg:flex w-72 bg-[#040c18]/95 border-l border-slate-800/80 p-5 flex-col justify-between z-20 backdrop-blur-xl transition-all duration-500 ${
              isMetadataActive ? "opacity-100 translate-x-0" : "opacity-0 translate-x-8"
            }`}
          >
            <div>
              <div className="flex items-center gap-2 mb-1 text-[10px] font-mono text-[#00f0ff] uppercase tracking-wider font-semibold">
                <CheckCircle2 className="w-3.5 h-3.5" />
                <span>Location Resolved</span>
              </div>
              <h3 className="text-base font-bold font-mono text-white uppercase tracking-wide mb-1">
                {selectedPreset.name}
              </h3>
              <p className="text-[11px] font-mono text-slate-400 mb-4">{selectedPreset.coords}</p>

              {/* Satellite Thumbnail Crop */}
              <div className="relative aspect-[16/9] rounded-xl overflow-hidden border border-cyan-500/30 mb-4">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={selectedPreset.image}
                  alt={selectedPreset.name}
                  className="w-full h-full object-cover"
                />
                <div className="absolute inset-0 bg-gradient-to-t from-[#020712] via-transparent to-transparent opacity-80" />
                <div className="absolute bottom-2 left-2 text-[9.5px] font-mono text-[#00f0ff]">
                  SENSOR FOOTPRINT
                </div>
              </div>

              {/* Telemetry Metadata Table */}
              <div className="space-y-2 font-mono text-[11px]">
                <div className="flex items-center justify-between pb-1.5 border-b border-slate-800/70">
                  <span className="text-slate-400">Satellite</span>
                  <span className="text-white font-semibold">{selectedPreset.satellite}</span>
                </div>
                <div className="flex items-center justify-between pb-1.5 border-b border-slate-800/70">
                  <span className="text-slate-400">Date</span>
                  <span className="text-white">{selectedPreset.date}</span>
                </div>
                <div className="flex items-center justify-between pb-1.5 border-b border-slate-800/70">
                  <span className="text-slate-400">Resolution</span>
                  <span className="text-cyan-300 font-semibold">{selectedPreset.resolution}</span>
                </div>
                <div className="flex items-center justify-between pb-1.5 border-b border-slate-800/70">
                  <span className="text-slate-400">Cloud Cover</span>
                  <span className="text-emerald-400 font-semibold">{selectedPreset.cloudCover}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-slate-400">Area</span>
                  <span className="text-white font-semibold">{selectedPreset.area}</span>
                </div>
              </div>
            </div>

            {/* View on Map CTA Button */}
            <div className="pt-4">
              <Link
                href={`/query?prompt=${encodeURIComponent(selectedPreset.query)}`}
                data-cursor="interactive"
                className="w-full py-2.5 px-4 rounded-xl bg-gradient-to-r from-cyan-500 to-[#00f0ff] hover:from-cyan-400 hover:to-[#38f5ff] text-[#020712] font-bold text-xs transition-all shadow-[0_0_20px_rgba(0,240,255,0.5)] flex items-center justify-center gap-2 group cursor-pointer"
              >
                <span>View on Map</span>
                <ArrowRight className="w-3.5 h-3.5 group-hover:translate-x-1 transition-transform" />
              </Link>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
