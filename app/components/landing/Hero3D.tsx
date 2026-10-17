"use client";

import React, { useRef, useState, useEffect } from "react";
import Link from "next/link";
import { ArrowRight, Gauge, Mic, TrendingUp, ChevronRight } from "lucide-react";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import EarthScene from "./EarthScene";
import SatelliteScene from "./SatelliteScene";
import SatelliteImageryReveal from "./SatelliteImageryReveal";
import HeroHeadlineSequence from "./HeroHeadlineSequence";
import { getCameraTimelineState } from "../../lib/landing/cameraController";

export default function Hero3D() {
  const containerRef = useRef<HTMLDivElement>(null);
  const [progress, setProgress] = useState(0);
  const [globeMode, setGlobeMode] = useState<"particle" | "realistic">("realistic");
  const [isDraggingGlobe, setIsDraggingGlobe] = useState(false);
  const [queryText, setQueryText] = useState("Indira Nagar, Lucknow");
  const [isReducedMotion, setIsReducedMotion] = useState(() => {
    if (typeof window !== "undefined" && window.matchMedia) {
      return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    }
    return false;
  });
  const [headlinePhase, setHeadlinePhase] = useState(1);

  const [mousePos, setMousePos] = useState({ x: 0, y: 0 });
  const [investigateBtnPos, setInvestigateBtnPos] = useState({ x: 0, y: 0 });

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (isReducedMotion) return;
    const { innerWidth, innerHeight } = window;
    const x = (e.clientX / innerWidth - 0.5) * 2;
    const y = (e.clientY / innerHeight - 0.5) * 2;
    setMousePos({ x, y });
  };

  const handleBtnMouseMove = (e: React.MouseEvent<HTMLButtonElement>) => {
    if (isReducedMotion) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const centerX = rect.left + rect.width / 2;
    const centerY = rect.top + rect.height / 2;
    const maxShift = 5;
    const deltaX = Math.max(-maxShift, Math.min(maxShift, (e.clientX - centerX) * 0.14));
    const deltaY = Math.max(-maxShift, Math.min(maxShift, (e.clientY - centerY) * 0.14));
    setInvestigateBtnPos({ x: deltaX, y: deltaY });
  };

  const handleBtnMouseLeave = () => {
    setInvestigateBtnPos({ x: 0, y: 0 });
  };

  useEffect(() => {
    const mediaQuery = window.matchMedia("(prefers-reduced-motion: reduce)");
    const handleChange = (e: MediaQueryListEvent) => setIsReducedMotion(e.matches);
    mediaQuery.addEventListener("change", handleChange);

    gsap.registerPlugin(ScrollTrigger);

    const el = containerRef.current;
    if (!el) return;

    const st = ScrollTrigger.create({
      trigger: el,
      start: "top top",
      end: "+=180%",
      pin: true,
      scrub: 0.4,
      onUpdate: (self) => {
        setProgress(self.progress);
      },
    });

    return () => {
      st.kill();
      mediaQuery.removeEventListener("change", handleChange);
    };
  }, []);

  const cameraState = getCameraTimelineState(progress);

  return (
    <div
      ref={containerRef}
      id="hero"
      onMouseMove={handleMouseMove}
      className="relative w-full h-screen bg-[#010308] text-slate-100 overflow-hidden select-none"
    >
      {/* 1. LAYER 1: Deep Space Background with Subtle Sparse Stars (0.1x Parallax) */}
      <div
        className="absolute inset-0 pointer-events-none opacity-40 z-0 transition-transform duration-200 ease-out"
        style={{
          transform: `translate3d(${mousePos.x * 12 * 0.1}px, ${mousePos.y * 12 * 0.1}px, 0)`,
          backgroundImage:
            "radial-gradient(1px 1px at 14% 22%, #ffffff 100%, transparent), radial-gradient(1px 1px at 48% 68%, #00f0ff 100%, transparent), radial-gradient(1.5px 1.5px at 80% 16%, #ffffff 100%, transparent), radial-gradient(1px 1px at 64% 48%, #00d4ff 100%, transparent)",
          backgroundSize: "360px 360px",
        }}
      />

      {/* 2. LAYER 2: Real-time 3D Earth & Satellite WebGL Canvas (OGL) */}
      <EarthScene
        scrollProgress={progress}
        earthScale={cameraState.earthScale}
        earthAlpha={cameraState.earthAlpha}
        satelliteAlpha={cameraState.satelliteAlpha}
        camZ={cameraState.camZ}
        camX={cameraState.camX + (isReducedMotion ? 0 : mousePos.x * 0.10 * 0.2)}
        camY={cameraState.camY + (isReducedMotion ? 0 : -mousePos.y * 0.10 * 0.2)}
        mouseX={mousePos.x}
        mouseY={mousePos.y}
        isReducedMotion={isReducedMotion}
        globeMode={globeMode}
        onDragStateChange={setIsDraggingGlobe}
      />

      {/* 3. LAYER 3: Orbital Trajectory Line */}
      <SatelliteScene
        scrollProgress={cameraState.satelliteOrbitProgress}
        opacity={cameraState.satelliteAlpha}
        isReducedMotion={isReducedMotion}
      />

      {/* 4. LAYER 4: Continuous Satellite Imagery Reveal Transition (Camera Push-in) */}
      <SatelliteImageryReveal
        progress={progress}
        opacity={cameraState.imageryAlpha}
        scale={cameraState.imageryScale}
        gridAlpha={cameraState.gridAlpha}
        scanBeamProgress={cameraState.scanBeamProgress}
        analysisReadyAlpha={cameraState.analysisReadyAlpha}
        isReducedMotion={isReducedMotion}
      />

      {/* 5. Top-Right Reference Tag: A CLEARER SAFER PLANET */}
      <div
        className="absolute top-24 right-8 lg:right-14 z-30 hidden sm:flex items-center gap-2 pointer-events-none transition-transform duration-200"
        style={{
          transform: `translate3d(${mousePos.x * 4 * 0.15}px, ${mousePos.y * 4 * 0.15}px, 0)`,
        }}
      >
        <span className="w-1.5 h-1.5 rounded-full bg-[#00f0ff] shadow-[0_0_8px_#00f0ff] animate-pulse" />
        <span className="text-[11px] font-mono tracking-[0.25em] text-slate-400 uppercase">
          A CLEARER SAFER PLANET
        </span>
      </div>

      {/* 6. HERO CONTENT — Reference Panel 01 Composition with Parallax & Line Reveals */}
      <div
        className="relative z-30 h-full max-w-[1440px] mx-auto px-6 sm:px-10 lg:px-14 flex flex-col justify-center pointer-events-none transition-all duration-300 pt-16 lg:pt-8"
        style={{
          opacity: cameraState.heroTextAlpha,
          transform: `translate3d(${mousePos.x * 6 * 0.2}px, ${-progress * 120 + mousePos.y * 6 * 0.2}px, 0)`,
        }}
      >
        <div className="max-w-2xl pointer-events-auto">
          {/* Eyebrow matching Panel 01 & responding to sequence phase */}
          <div className="text-[11px] sm:text-xs font-mono tracking-[0.22em] text-[#00f0ff] uppercase mb-4 select-none font-semibold flex items-center gap-2.5 transition-all duration-300">
            <span className="w-2 h-2 rounded-full bg-[#00f0ff] shadow-[0_0_10px_#00f0ff] animate-pulse" />
            <span className="transition-all duration-300">
              {headlinePhase === 2
                ? "STAGE 01 // QUERY RESOLUTION // LAT 26.85° N, LON 80.95° E"
                : headlinePhase === 3
                ? "STAGE 02 // MULTI-SENSOR DETECTION & GROUNDING"
                : headlinePhase === 4
                ? "STAGE 03 // EVIDENCE VERIFICATION // EPSG:4326 CONFIRMED"
                : "EARTH INTELLIGENCE. REAL IMPACT."}
            </span>
          </div>

          {/* Cinematic Animated Headline Sequence */}
          <HeroHeadlineSequence
            onPhaseChange={setHeadlinePhase}
            isReducedMotion={isReducedMotion}
          />

          {/* Secondary Process Indicator (Dynamically Highlighted & Compact) */}
          <div className="flex items-center gap-2 sm:gap-2.5 font-mono text-[11px] sm:text-xs tracking-[0.2em] mb-3 select-none">
            <span
              className={`transition-all duration-500 ${
                headlinePhase === 2
                  ? "text-[#00f0ff] font-bold drop-shadow-[0_0_10px_#00f0ff]"
                  : "text-slate-500 font-medium"
              }`}
            >
              ASK
            </span>
            <span className="text-slate-600 font-light select-none">→</span>
            <span
              className={`transition-all duration-500 ${
                headlinePhase === 3
                  ? "text-cyan-200 font-bold drop-shadow-[0_0_10px_rgba(0,240,255,0.6)]"
                  : "text-slate-500 font-medium"
              }`}
            >
              ANALYZE
            </span>
            <span className="text-slate-600 font-light select-none">→</span>
            <span
              className={`transition-all duration-500 ${
                headlinePhase === 4
                  ? "text-[#00f0ff] font-bold drop-shadow-[0_0_10px_#00f0ff]"
                  : "text-slate-500 font-medium"
              }`}
            >
              VERIFY
            </span>
          </div>

          {/* Supporting Copy matching Panel 01 with tightened spacing */}
          <p
            className="text-slate-300 text-sm sm:text-base mb-6 font-light max-w-lg leading-relaxed transition-opacity duration-300"
            style={{ opacity: Math.max(0, 1 - progress * 2.5) }}
          >
            From pixels to a more informed planet.
          </p>

          {/* 3 Telemetry Stats Badges matching Panel 01 with Layered Parallax */}
          <div
            className="grid grid-cols-3 gap-3 max-w-lg mb-8 transition-transform duration-300"
            style={{
              transform: `translate3d(${mousePos.x * 8 * 0.15}px, ${mousePos.y * 8 * 0.15}px, 0)`,
            }}
          >
            {/* Stat 1: 100% GLOBAL COVERAGE */}
            <div className="p-3 sm:p-3.5 rounded-xl bg-[#061222]/80 border border-slate-800/80 backdrop-blur-md hover:border-cyan-500/40 hover:-translate-y-1 transition-all duration-300">
              <div className="text-xl sm:text-2xl font-extrabold text-[#00f0ff] font-mono leading-none mb-1">
                100%
              </div>
              <div className="text-[9px] sm:text-[10px] text-slate-400 uppercase tracking-wider font-mono">
                GLOBAL COVERAGE
              </div>
            </div>

            {/* Stat 2: MULTI-SENSOR SATELLITE SOURCES */}
            <div className="p-3 sm:p-3.5 rounded-xl bg-[#061222]/80 border border-slate-800/80 backdrop-blur-md hover:border-cyan-500/40 hover:-translate-y-1 transition-all duration-300">
              <div className="text-xs sm:text-sm font-bold text-white font-mono leading-tight mb-1">
                MULTI-SENSOR
              </div>
              <div className="text-[9px] sm:text-[10px] text-slate-400 uppercase tracking-wider font-mono">
                SATELLITE SOURCES
              </div>
            </div>

            {/* Stat 3: REAL-TIME AI ANALYSIS */}
            <div className="p-3 sm:p-3.5 rounded-xl bg-[#061222]/80 border border-slate-800/80 backdrop-blur-md hover:border-cyan-500/40 hover:-translate-y-1 transition-all duration-300">
              <div className="text-xs sm:text-sm font-bold text-[#00f0ff] font-mono leading-tight mb-1">
                REAL-TIME
              </div>
              <div className="text-[9px] sm:text-[10px] text-slate-400 uppercase tracking-wider font-mono">
                AI ANALYSIS
              </div>
            </div>
          </div>

          {/* Interactive Fast Query / Investigation Launch Bar */}
          <div className="max-w-xl">
            <form
              onSubmit={(e) => {
                e.preventDefault();
                window.location.href = `/query?prompt=${encodeURIComponent(queryText)}`;
              }}
              className="flex items-center gap-2 p-1.5 rounded-2xl bg-[#040d1a]/85 border border-cyan-500/35 backdrop-blur-xl shadow-[0_0_30px_rgba(0,0,0,0.85)] hover:border-cyan-400/60 transition-all"
            >
              <div className="flex-1 flex items-center gap-2 px-3 py-1.5">
                <input
                  type="text"
                  value={queryText}
                  onChange={(e) => setQueryText(e.target.value)}
                  placeholder="Enter location or investigation query..."
                  className="w-full bg-transparent text-xs sm:text-sm text-slate-100 placeholder-slate-400 focus:outline-none font-mono"
                />
              </div>

              <button
                type="submit"
                data-cursor="interactive"
                onMouseMove={handleBtnMouseMove}
                onMouseLeave={handleBtnMouseLeave}
                style={{
                  transform: `translate3d(${investigateBtnPos.x}px, ${investigateBtnPos.y}px, 0)`,
                }}
                className="px-5 sm:px-6 py-2.5 sm:py-3 rounded-xl bg-[#00f0ff] hover:bg-[#38f5ff] text-[#020712] font-bold text-xs sm:text-sm transition-all duration-150 shadow-[0_0_24px_rgba(0,240,255,0.65)] hover:shadow-[0_0_32px_rgba(0,240,255,0.85)] hover:scale-105 active:scale-95 cursor-pointer whitespace-nowrap flex items-center gap-1.5 group"
              >
                <span>Investigate</span>
                <ArrowRight size={14} className="group-hover:translate-x-1 transition-transform" />
              </button>
            </form>
          </div>
        </div>
      </div>

      {/* 7. Bottom Navigation Pills (matching approved mockup bottom right) */}
      <div className="absolute bottom-6 right-8 lg:right-14 z-30 hidden sm:flex items-center gap-3">
        <Link
          href="/query"
          className="text-slate-300 hover:text-white font-mono text-xs px-4 py-2 rounded-xl bg-[#061222]/80 border border-slate-800 backdrop-blur-md"
        >
          Log In
        </Link>
        <Link
          href="/query"
          className="bg-[#00f0ff] hover:bg-[#38f5ff] text-black font-bold text-xs px-5 py-2 rounded-xl shadow-[0_0_20px_rgba(0,240,255,0.6)]"
        >
          Get Started
        </Link>
      </div>
    </div>
  );
}
