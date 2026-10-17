"use client";

import React, { useState, useEffect, useRef } from "react";
import { Database, Cpu, HelpCircle, ArrowUpRight } from "lucide-react";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";

interface ProblemCard {
  id: string;
  tag: string;
  title: string;
  desc: string;
  icon: React.ElementType;
  stat: string;
  statLabel: string;
  baseZ: number; // Z-depth in px
  baseScale: number;
}

const PROBLEM_CARDS: ProblemCard[] = [
  {
    id: "massive-data",
    tag: "01 // SCALE",
    title: "MASSIVE DATA",
    desc: "Petabytes of raw multi-spectral satellite imagery collected daily across the globe.",
    icon: Database,
    stat: "100+ TB",
    statLabel: "Daily Ingestion",
    baseZ: -40,
    baseScale: 0.96,
  },
  {
    id: "complex-analysis",
    tag: "02 // LATENCY",
    title: "COMPLEX ANALYSIS",
    desc: "Requires manual photointerpretation, georeferencing, and multi-day domain overhead.",
    icon: Cpu,
    stat: "48-72h",
    statLabel: "Manual Overhead",
    baseZ: 0,
    baseScale: 1.0,
  },
  {
    id: "unclear-answers",
    tag: "03 // UNCERTAINTY",
    title: "UNCLEAR ANSWERS",
    desc: "Isolated pixels without automated geodetic evidence leave operators uncertain.",
    icon: HelpCircle,
    stat: "< 12%",
    statLabel: "Direct Utility",
    baseZ: 40,
    baseScale: 1.04,
  },
];

export default function ProblemScene() {
  const containerRef = useRef<HTMLDivElement>(null);
  const [hoveredCard, setHoveredCard] = useState<string | null>(null);
  const [convergence, setConvergence] = useState(0); // 0 = separated, 1 = converged

  useEffect(() => {
    gsap.registerPlugin(ScrollTrigger);

    const el = containerRef.current;
    if (!el) return;

    const isReduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (isReduced) return;

    // Scroll convergence: As section enters viewport, cards converge toward center plane, then separate
    const st = ScrollTrigger.create({
      trigger: el,
      start: "top 80%",
      end: "bottom 20%",
      scrub: 0.5,
      onUpdate: (self) => {
        // Peaks at center (self.progress = 0.5 -> convergence = 1.0)
        const p = self.progress;
        const conv = Math.sin(p * Math.PI);
        setConvergence(conv);
      },
    });

    return () => {
      st.kill();
    };
  }, []);

  return (
    <section
      ref={containerRef}
      id="problem"
      data-spatial="true"
      className="relative min-h-screen w-full bg-[#02050d] text-slate-100 py-28 px-6 sm:px-10 lg:px-14 flex flex-col justify-center overflow-hidden border-t border-cyan-950/30 select-none"
    >
      {/* Background Satellite Imagery Mosaic with High-Pass Contrast & Dark Overlay */}
      <div
        className="absolute inset-0 opacity-20 pointer-events-none mix-blend-screen"
        style={{
          backgroundImage: "url('/imagery/satellite_urban_surface.jpg')",
          backgroundSize: "cover",
          backgroundPosition: "center",
          filter: "grayscale(100%) contrast(140%) brightness(0.6)",
        }}
      />

      {/* Atmospheric Vignette & Grid */}
      <div className="absolute inset-0 bg-gradient-to-b from-[#010308] via-transparent to-[#010308] pointer-events-none" />
      <div
        className="absolute inset-0 opacity-15 pointer-events-none"
        style={{
          backgroundImage:
            "linear-gradient(to right, rgba(0,240,255,0.06) 1px, transparent 1px), linear-gradient(to bottom, rgba(0,240,255,0.06) 1px, transparent 1px)",
          backgroundSize: "64px 64px",
        }}
      />

      {/* Ambient Volumetric Radial Glow */}
      <div className="absolute top-1/2 right-1/4 -translate-y-1/2 w-[700px] h-[700px] bg-cyan-600/5 rounded-full blur-[160px] pointer-events-none" />

      <div className="max-w-[1440px] mx-auto w-full relative z-10">
        {/* Top Header Row matching Reference Panel 02 */}
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4 mb-16 pb-6 border-b border-slate-800/60">
          <div className="flex items-center gap-3">
            <span className="text-xs font-mono font-bold text-[#00f0ff] tracking-[0.25em] uppercase">
              02
            </span>
            <span className="text-slate-600 font-mono text-xs">/</span>
            <span className="text-xs font-mono tracking-[0.2em] text-slate-400 uppercase">
              0:04 — 0:08 THE PROBLEM
            </span>
          </div>

          <div className="text-[11px] font-mono tracking-[0.22em] text-slate-400 uppercase">
            MORE DATA. MORE COMPLEXITY. BIGGER QUESTIONS.
          </div>
        </div>

        {/* Main Headline & Context Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 items-end mb-20">
          <div className="lg:col-span-8">
            <h2 className="text-4xl sm:text-6xl lg:text-[72px] font-extrabold tracking-tight text-white leading-[1.06]">
              Earth Generates <br />
              Billions of Images. <br />
              <span className="text-white">But Raw Data </span>
              <span className="text-[#00f0ff] drop-shadow-[0_0_30px_rgba(0,240,255,0.4)]">
                Isn’t Enough.
              </span>
            </h2>
          </div>

          <div className="lg:col-span-4">
            <p className="text-slate-400 text-sm sm:text-base leading-relaxed font-light">
              Satellites orbit constantly, capturing massive terrain swaths. Yet organizations still
              struggle with manual image interpretation, delayed answers, and disconnected geospatial
              evidence.
            </p>
          </div>
        </div>

        {/* Spatial 3D Depth Card Environment (Requirement 6: Breaking the Grid in Z-Space) */}
        <div
          className="grid grid-cols-1 md:grid-cols-3 gap-6 lg:gap-8"
          style={{ perspective: "1200px" }}
        >
          {PROBLEM_CARDS.map((card, index) => {
            const Icon = card.icon;
            const isHovered = hoveredCard === card.id;
            const isAnyHovered = hoveredCard !== null;

            // Compute spatial Z depth:
            // When hovered: card comes forward +60px, siblings move back -35px
            // Otherwise: baseZ is modulated by scroll convergence
            let targetZ = card.baseZ * (1 - convergence * 0.7);
            let targetScale = card.baseScale;

            if (isHovered) {
              targetZ = 65;
              targetScale = 1.05;
            } else if (isAnyHovered) {
              targetZ = -35;
              targetScale = 0.94;
            }

            return (
              <div
                key={card.id}
                onMouseEnter={() => setHoveredCard(card.id)}
                onMouseLeave={(e) => {
                  setHoveredCard(null);
                  e.currentTarget.style.transform = `perspective(1200px) translateZ(${targetZ}px) rotateX(0deg) rotateY(0deg) scale(${targetScale})`;
                }}
                onMouseMove={(e) => {
                  const rect = e.currentTarget.getBoundingClientRect();
                  const x = (e.clientX - rect.left) / rect.width;
                  const y = (e.clientY - rect.top) / rect.height;
                  const rotX = (0.5 - y) * 4; // ±2 deg
                  const rotY = (x - 0.5) * 5; // ±2.5 deg
                  e.currentTarget.style.transform = `perspective(1200px) translateZ(${targetZ}px) rotateX(${rotX}deg) rotateY(${rotY}deg) scale(${targetScale})`;
                }}
                data-cursor="interactive"
                className={`group relative p-8 rounded-2xl bg-[#040c18]/85 border backdrop-blur-xl flex flex-col justify-between min-h-[320px] cursor-pointer overflow-hidden ${
                  isHovered
                    ? "border-cyan-400 shadow-[0_20px_60px_rgba(0,240,255,0.3)] bg-[#05162a]/95 z-30 opacity-100"
                    : isAnyHovered
                    ? "border-slate-800/40 opacity-45 z-10"
                    : "border-slate-800/80 hover:border-slate-700/80 z-20 opacity-95"
                }`}
                style={{
                  transform: `perspective(1200px) translateZ(${targetZ}px) scale(${targetScale})`,
                  transition:
                    "transform 0.35s cubic-bezier(0.16, 1, 0.3, 1), opacity 0.4s ease, border-color 0.4s ease, box-shadow 0.4s ease",
                  transformStyle: "preserve-3d",
                }}
              >
                {/* Dynamic Radial Glare following cursor */}
                <div
                  className={`absolute inset-0 pointer-events-none transition-opacity duration-300 ${
                    isHovered ? "opacity-100" : "opacity-0"
                  }`}
                  style={{
                    background:
                      "radial-gradient(circle 280px at center, rgba(0,240,255,0.08), transparent 70%)",
                  }}
                />

                {/* Subtle Cyan Top Edge Glow on Hover */}
                <div
                  className={`absolute top-0 left-6 right-6 h-px bg-gradient-to-r from-transparent via-[#00f0ff] to-transparent transition-opacity duration-500 ${
                    isHovered ? "opacity-100" : "opacity-0"
                  }`}
                />

                {/* Card Top: Icon & Micro Tag */}
                <div className="relative z-10">
                  <div className="flex items-center justify-between mb-8">
                    <div
                      className={`w-14 h-14 rounded-2xl flex items-center justify-center border transition-all duration-500 ${
                        isHovered
                          ? "bg-[#00f0ff]/15 border-[#00f0ff]/70 text-[#00f0ff] shadow-[0_0_24px_rgba(0,240,255,0.4)] scale-110"
                          : "bg-[#061222] border-slate-800 text-cyan-400"
                      }`}
                    >
                      <Icon className="w-6 h-6 stroke-[1.8]" />
                    </div>

                    <span className="text-[10px] font-mono tracking-widest text-slate-500 group-hover:text-cyan-400 transition-colors uppercase font-bold">
                      {card.tag}
                    </span>
                  </div>

                  {/* Title & Description */}
                  <h3 className="text-xl sm:text-2xl font-bold font-mono tracking-wide text-white mb-3 group-hover:text-[#00f0ff] transition-colors">
                    {card.title}
                  </h3>
                  <p className="text-slate-400 text-sm leading-relaxed font-light mb-6">
                    {card.desc}
                  </p>
                </div>

                {/* Card Bottom: Technical Spec Metric */}
                <div className="pt-5 border-t border-slate-800/80 flex items-center justify-between relative z-10">
                  <div>
                    <div className="text-[9px] font-mono uppercase tracking-wider text-slate-500">
                      {card.statLabel}
                    </div>
                    <div className="text-lg font-mono font-bold text-white group-hover:text-[#00f0ff] transition-colors">
                      {card.stat}
                    </div>
                  </div>

                  <div className="w-8 h-8 rounded-full border border-slate-800 flex items-center justify-center text-slate-500 group-hover:border-cyan-400 group-hover:text-[#00f0ff] group-hover:scale-110 transition-all">
                    <ArrowUpRight className="w-4 h-4" />
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
}
