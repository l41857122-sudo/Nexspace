"use client";

import React, { useState, useEffect, useRef } from "react";
import Link from "next/link";
import {
  ShieldCheck,
  MapPin,
  CheckCircle2,
  ArrowRight,
  ArrowDown,
  Layers,
  Cpu,
  Globe2,
} from "lucide-react";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";

export default function EvidenceScene() {
  const containerRef = useRef<HTMLDivElement>(null);
  const [scrollProgress, setScrollProgress] = useState(0);
  const [isTargetHovered, setIsTargetHovered] = useState(false);

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
      end: "+=140%",
      pin: true,
      scrub: 0.4,
      onUpdate: (self) => {
        setScrollProgress(self.progress);
      },
    });

    return () => {
      st.kill();
    };
  }, []);

  // Storytelling stages based on scroll:
  // 1. Target detected (p >= 0.0)
  // 2. Geographic context (p >= 0.20)
  // 3. Evidence & Pipeline chain (p >= 0.40)
  // 4. Confidence meter (p >= 0.60)
  // 5. Source (p >= 0.80)
  // 6. Verified Intelligence Seal climax (p >= 0.90)
  const isTargetVisible = true;
  const isContextVisible = scrollProgress >= 0.2;
  const isEvidenceVisible = scrollProgress >= 0.4;
  const isConfidenceVisible = scrollProgress >= 0.6;
  const isSourceVisible = scrollProgress >= 0.8;
  const isClimaxAchieved = scrollProgress >= 0.9;

  // Animated confidence calculation (0% -> 92.4%)
  const confidenceScore = Math.min(
    92.4,
    Math.max(12, Math.round(((scrollProgress - 0.5) / 0.4) * 92.4 * 10) / 10)
  );

  return (
    <section
      ref={containerRef}
      id="evidence"
      data-spatial="true"
      className="relative min-h-screen w-full bg-[#010308] text-slate-100 py-24 px-6 sm:px-10 lg:px-14 flex flex-col justify-center overflow-hidden border-t border-cyan-950/30 select-none"
    >
      {/* Ambient Glow */}
      <div
        className={`absolute top-1/2 left-1/4 -translate-y-1/2 w-[700px] h-[700px] bg-cyan-600/5 rounded-full blur-[180px] pointer-events-none transition-all duration-700 ${
          isClimaxAchieved ? "opacity-100 scale-125 bg-cyan-500/10" : "opacity-40"
        }`}
      />

      <div className="max-w-[1440px] mx-auto w-full relative z-10 flex flex-col justify-between">
        {/* Section Header Row matching Reference Panel 07 */}
        <div className="flex items-center justify-between pb-3 border-b border-slate-800/60 mb-8">
          <div className="flex items-center gap-3">
            <span className="text-xs font-mono font-bold text-[#00f0ff] tracking-[0.25em] uppercase">
              07
            </span>
            <span className="text-slate-600 font-mono text-xs">/</span>
            <span className="text-xs font-mono tracking-[0.2em] text-slate-400 uppercase">
              0:35 — 0:41 EVIDENCE / VERIFICATION
            </span>
          </div>

          <div className="hidden sm:flex items-center gap-2 font-mono text-xs text-[#00f0ff]">
            <span
              className={`w-2 h-2 rounded-full ${
                isClimaxAchieved ? "bg-emerald-400 shadow-[0_0_10px_#10b981]" : "bg-[#00f0ff] animate-ping"
              }`}
            />
            <span className="tracking-widest uppercase">
              {isClimaxAchieved ? "VERIFICATION COMPLETE" : "EVIDENCE CORRELATION IN PROGRESS"}
            </span>
          </div>
        </div>

        {/* Section Title with Masked Line Reveal */}
        <div className="max-w-3xl mb-10">
          <h2 className="text-3xl sm:text-5xl lg:text-6xl font-extrabold tracking-tight text-white mb-3 uppercase font-sans">
            <span className="block overflow-hidden">
              <span className="block">EVIDENCE &amp;</span>
            </span>
            <span className="block overflow-hidden">
              <span className="block text-[#00f0ff] drop-shadow-[0_0_35px_rgba(0,240,255,0.45)]">
                VERIFICATION
              </span>
            </span>
          </h2>
          <p className="text-slate-400 text-xs sm:text-base leading-relaxed font-light">
            NexSpace doesn’t simply output answers. Every finding is anchored to verifiable geodetic
            bounds, sensor telemetry, and calibrated confidence models.
          </p>
        </div>

        {/* Main Verification Dashboard Layout matching Reference Panel 07 */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 lg:gap-8 items-center mb-8">
          {/* Left: Satellite Capture with Interactive Detected Target Bounding Box (Requirement 22) */}
          <div className="lg:col-span-6 relative aspect-[16/11] rounded-2xl overflow-hidden border border-cyan-500/40 bg-[#030914] shadow-[0_0_50px_rgba(0,0,0,0.8)] group">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src="/imagery/satellite_urban_surface.jpg"
              alt="Verified Target Imagery"
              className="w-full h-full object-cover transition-transform duration-700 group-hover:scale-105"
            />

            {/* Coordinate Grid Overlay */}
            <div
              className="absolute inset-0 opacity-20 pointer-events-none"
              style={{
                backgroundImage:
                  "linear-gradient(to right, rgba(0,240,255,0.2) 1px, transparent 1px), linear-gradient(to bottom, rgba(0,240,255,0.2) 1px, transparent 1px)",
                backgroundSize: "60px 60px",
              }}
            />

            {/* Target Detected Glowing Cyan Bounding Box with Hover Interaction */}
            <div
              onMouseEnter={() => setIsTargetHovered(true)}
              onMouseLeave={() => setIsTargetHovered(false)}
              data-cursor="interactive"
              className={`absolute border-2 transition-all duration-300 cursor-pointer ${
                isTargetHovered
                  ? "border-white bg-[#00f0ff]/30 shadow-[0_0_45px_rgba(0,240,255,0.85)] scale-102"
                  : "border-[#00f0ff] bg-[#00f0ff]/15 shadow-[0_0_30px_rgba(0,240,255,0.5)]"
              } ${isTargetVisible ? "opacity-100 scale-100" : "opacity-0 scale-95"}`}
              style={{
                left: "28%",
                top: "32%",
                width: "44%",
                height: "40%",
              }}
            >
              {/* Corner crosshairs */}
              <div className="absolute -top-1 -left-1 w-2.5 h-2.5 border-t-2 border-l-2 border-white" />
              <div className="absolute -top-1 -right-1 w-2.5 h-2.5 border-t-2 border-r-2 border-white" />
              <div className="absolute -bottom-1 -left-1 w-2.5 h-2.5 border-b-2 border-l-2 border-white" />
              <div className="absolute -bottom-1 -right-1 w-2.5 h-2.5 border-b-2 border-r-2 border-white" />

              {/* Pulsing Target Dot */}
              <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-3 h-3 rounded-full bg-[#00f0ff] shadow-[0_0_12px_#00f0ff]">
                <div className="absolute inset-0 rounded-full bg-[#00f0ff] animate-ping opacity-75" />
              </div>

              {/* Interactive Target Information Connector Line (Requirement 22) */}
              {isTargetHovered && (
                <div className="absolute -top-6 -right-6 pointer-events-none animate-in fade-in zoom-in-95 duration-200">
                  <div className="px-2 py-0.5 rounded bg-[#020712]/95 border border-[#00f0ff] text-[9px] font-mono text-[#00f0ff] uppercase shadow-[0_0_12px_#00f0ff]">
                    FEATURE: PRIMARY ROOF // 4,820 m²
                  </div>
                </div>
              )}
            </div>

            {/* Floating Target Badge matching Reference */}
            <div className="absolute top-[18%] left-1/2 -translate-x-1/2 px-4 py-2 rounded-xl bg-[#020712]/95 border border-[#00f0ff] shadow-[0_0_24px_rgba(0,240,255,0.7)] backdrop-blur-md flex flex-col items-center">
              <span className="text-[10px] font-mono text-[#00f0ff] tracking-wider uppercase font-bold flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-[#00f0ff] animate-pulse" />
                TARGET DETECTED
              </span>
              <span className="text-xs font-mono font-bold text-white">Building Complex // AOI-04</span>
            </div>
          </div>

          {/* Center: Geographic Context Card & Pipeline Chain */}
          <div className="lg:col-span-3 space-y-4">
            {/* Metadata Context Card with progressive reveals */}
            <div
              className={`p-5 rounded-2xl bg-[#040c18] border transition-all duration-300 backdrop-blur-xl shadow-[0_0_30px_rgba(0,0,0,0.6)] ${
                isTargetHovered
                  ? "border-[#00f0ff] shadow-[0_0_30px_rgba(0,240,255,0.3)] bg-[#051426]"
                  : "border-cyan-500/30"
              }`}
            >
              <div className="space-y-3 font-mono text-xs">
                {/* 1. Geographic Context */}
                <div
                  className={`transition-all duration-500 ${
                    isContextVisible ? "opacity-100 translate-y-0" : "opacity-30 translate-y-2"
                  }`}
                >
                  <div className="text-[10px] text-slate-500 uppercase tracking-wider mb-0.5">
                    GEOGRAPHIC CONTEXT
                  </div>
                  <div className="text-white font-bold flex items-center gap-1.5">
                    <MapPin className="w-3.5 h-3.5 text-[#00f0ff]" />
                    <span>Indira Nagar, Lucknow</span>
                  </div>
                </div>

                {/* 2. Evidence */}
                <div
                  className={`pt-2.5 border-t border-slate-800/80 transition-all duration-500 ${
                    isEvidenceVisible ? "opacity-100 translate-y-0" : "opacity-30 translate-y-2"
                  }`}
                >
                  <div className="text-[10px] text-slate-500 uppercase tracking-wider mb-0.5">
                    EVIDENCE
                  </div>
                  <div className="text-cyan-300 font-semibold">AI-detected structure footprint</div>
                </div>

                {/* 3. Confidence Bar & Counter */}
                <div
                  className={`pt-2.5 border-t border-slate-800/80 transition-all duration-500 ${
                    isConfidenceVisible ? "opacity-100 translate-y-0" : "opacity-30 translate-y-2"
                  }`}
                >
                  <div className="flex justify-between items-center mb-1">
                    <div className="text-[10px] text-slate-500 uppercase tracking-wider">
                      CONFIDENCE
                    </div>
                    <div className="text-lg font-bold text-[#00f0ff] font-mono">
                      {isConfidenceVisible ? `${confidenceScore}%` : "--"}
                    </div>
                  </div>
                  {/* Progress meter bar */}
                  <div className="w-full h-1.5 bg-slate-800 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-cyan-500 to-[#00f0ff] transition-all duration-300 shadow-[0_0_8px_#00f0ff]"
                      style={{
                        width: isConfidenceVisible ? `${(confidenceScore / 100) * 100}%` : "0%",
                      }}
                    />
                  </div>
                </div>

                {/* 4. Source */}
                <div
                  className={`pt-2.5 border-t border-slate-800/80 transition-all duration-500 ${
                    isSourceVisible ? "opacity-100 translate-y-0" : "opacity-30 translate-y-2"
                  }`}
                >
                  <div className="text-[10px] text-slate-500 uppercase tracking-wider mb-0.5">
                    SOURCE
                  </div>
                  <div className="text-slate-300 text-[11px]">Sentinel-2 L2A • 12 Mar 2024</div>
                </div>
              </div>
            </div>

            {/* Flow Chain matching Reference */}
            <div className="p-4 rounded-xl bg-[#040c18]/80 border border-slate-800 flex flex-col gap-2 font-mono text-xs">
              <div className="flex items-center gap-2 text-slate-300">
                <Layers className="w-3.5 h-3.5 text-cyan-400" />
                <span className="text-[11px]">SATELLITE IMAGE</span>
              </div>
              <ArrowDown className="w-3 h-3 text-cyan-500 ml-4" />
              <div className="flex items-center gap-2 text-slate-300">
                <Cpu className="w-3.5 h-3.5 text-cyan-400" />
                <span className="text-[11px]">AI ANALYSIS</span>
              </div>
              <ArrowDown className="w-3 h-3 text-cyan-500 ml-4" />
              <div className="flex items-center gap-2 text-[#00f0ff] font-bold">
                <Globe2 className="w-3.5 h-3.5 text-[#00f0ff]" />
                <span className="text-[11px]">GEOGRAPHIC RESULT</span>
              </div>
            </div>
          </div>

          {/* Right: VERIFIED INTELLIGENCE Seal Climax with Mask Reveal (Requirement 23) */}
          <div
            className={`lg:col-span-3 flex flex-col items-center text-center p-8 rounded-2xl bg-[#040c18] border transition-all duration-700 backdrop-blur-xl ${
              isClimaxAchieved
                ? "border-[#00f0ff] shadow-[0_0_50px_rgba(0,240,255,0.4)] scale-102 bg-[#051426]"
                : "border-slate-800/80 opacity-70"
            }`}
          >
            {/* Glowing Shield Check Icon with climax pulse */}
            <div
              className={`w-20 h-20 rounded-2xl border-2 flex items-center justify-center transition-all duration-500 mb-6 ${
                isClimaxAchieved
                  ? "bg-[#00f0ff]/20 border-[#00f0ff] text-[#00f0ff] shadow-[0_0_35px_rgba(0,240,255,0.6)] scale-110"
                  : "bg-slate-900 border-slate-700 text-slate-500"
              }`}
            >
              <ShieldCheck className="w-10 h-10 stroke-[2.2]" />
            </div>

            {/* Masked Line Climax Reveal: Cyan phrase arrives last */}
            <div className="mb-2">
              <span className="block overflow-hidden">
                <span className="block text-xl sm:text-2xl font-extrabold font-mono uppercase tracking-wider text-white">
                  VERIFIED
                </span>
              </span>
              <span className="block overflow-hidden">
                <span
                  className={`block text-xl sm:text-2xl font-extrabold font-mono uppercase tracking-wider text-[#00f0ff] drop-shadow-[0_0_30px_rgba(0,240,255,0.5)] transition-all duration-700 delay-300 ${
                    isClimaxAchieved ? "translate-y-0 opacity-100" : "translate-y-4 opacity-0"
                  }`}
                >
                  INTELLIGENCE
                </span>
              </span>
            </div>

            <p className="text-slate-400 text-xs sm:text-sm font-light leading-relaxed mb-6">
              Real data. Transparent analysis. Actionable insights.
            </p>

            <Link
              href="/evidence"
              data-cursor="interactive"
              className="w-full py-2.5 px-4 rounded-xl bg-slate-900 border border-cyan-500/40 hover:border-cyan-400 text-cyan-300 hover:text-white font-mono text-xs font-semibold flex items-center justify-center gap-2 transition-all hover:shadow-[0_0_16px_rgba(0,240,255,0.3)] cursor-pointer"
            >
              <span>Explore Evidence Store</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </Link>
          </div>
        </div>
      </div>
    </section>
  );
}
