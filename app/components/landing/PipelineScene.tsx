"use client";

import React, { useState, useEffect, useRef } from "react";
import { FileText, Layers, Brain, ShieldCheck, CheckCircle2, ArrowRight } from "lucide-react";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";

interface PipelineNode {
  number: string;
  title: string;
  desc: string;
  icon: React.ElementType;
}

const NODES: PipelineNode[] = [
  {
    number: "01",
    title: "UNDERSTAND",
    desc: "Query, location and imagery requirements parsed into geodetic bounds.",
    icon: FileText,
  },
  {
    number: "02",
    title: "DISCOVER",
    desc: "Find and catalog the best multispectral satellite passes with lowest cloud cover.",
    icon: Layers,
  },
  {
    number: "03",
    title: "ANALYZE",
    desc: "Apply deep spatial segmentation, feature extraction, and change detection.",
    icon: Brain,
  },
  {
    number: "04",
    title: "VERIFY",
    desc: "Deliver audited evidence with verifiable geodetic metadata and confidence scores.",
    icon: ShieldCheck,
  },
];

export default function PipelineScene() {
  const containerRef = useRef<HTMLDivElement>(null);
  const [scrollProgress, setScrollProgress] = useState(0);
  const [activeNode, setActiveNode] = useState(0);
  const [hoveredNode, setHoveredNode] = useState<number | null>(null);

  useEffect(() => {
    gsap.registerPlugin(ScrollTrigger);

    const el = containerRef.current;
    if (!el) return;

    const isMobile = window.innerWidth < 1024;
    const isReduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    if (isReduced || isMobile) {
      // Auto cycle on mobile or reduced motion
      const interval = setInterval(() => {
        setActiveNode((prev) => (prev >= 3 ? 0 : prev + 1));
      }, 3500);
      return () => clearInterval(interval);
    }

    const st = ScrollTrigger.create({
      trigger: el,
      start: "top top",
      end: "+=120%",
      pin: true,
      scrub: 0.4,
      onUpdate: (self) => {
        setScrollProgress(self.progress);
        // Map 0 -> 1 to active node 0 -> 3
        const nodeIdx = Math.min(3, Math.floor(self.progress * 4));
        setActiveNode(nodeIdx);
      },
    });

    return () => {
      st.kill();
    };
  }, []);

  // Compute line progress percentage (0% to 100%)
  const linePercentage =
    scrollProgress > 0
      ? Math.min(100, scrollProgress * 100)
      : ((activeNode + 1) / NODES.length) * 100;

  return (
    <section
      ref={containerRef}
      id="pipeline"
      className="relative min-h-screen w-full bg-[#010308] text-slate-100 py-24 px-6 sm:px-10 lg:px-14 flex flex-col justify-center overflow-hidden border-t border-cyan-950/30 select-none"
    >
      {/* Curved Earth Horizon Background with Glowing Atmospheric Limb matching Reference Panel 05 */}
      <div className="absolute inset-0 pointer-events-none overflow-hidden">
        {/* Curving Earth Limb SVG */}
        <svg
          className="absolute -bottom-[20%] left-1/2 -translate-x-1/2 w-[180vw] max-w-[2400px] h-[750px] opacity-40 pointer-events-none"
          viewBox="0 0 1600 600"
          fill="none"
        >
          <defs>
            <radialGradient id="atmoGlow" cx="50%" cy="100%" r="50%">
              <stop offset="0%" stopColor="#00f0ff" stopOpacity="0.45" />
              <stop offset="45%" stopColor="#0077cc" stopOpacity="0.2" />
              <stop offset="85%" stopColor="#001830" stopOpacity="0" />
            </radialGradient>
            <linearGradient id="limbGrad" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#002244" stopOpacity="0" />
              <stop offset="25%" stopColor="#00a0d4" stopOpacity="0.8" />
              <stop offset="50%" stopColor="#00f0ff" stopOpacity="1" />
              <stop offset="75%" stopColor="#00a0d4" stopOpacity="0.8" />
              <stop offset="100%" stopColor="#002244" stopOpacity="0" />
            </linearGradient>
          </defs>

          {/* Glowing Atmospheric Dome */}
          <path d="M -100 600 Q 800 160 1700 600 Z" fill="url(#atmoGlow)" />

          {/* Thin Sharp Luminous Horizon Line */}
          <path
            d="M -100 600 Q 800 160 1700 600"
            stroke="url(#limbGrad)"
            strokeWidth="2.5"
            fill="none"
            filter="drop-shadow(0 0 16px rgba(0,240,255,0.9))"
          />
        </svg>

        {/* Ambient starfield dots */}
        <div
          className="absolute inset-0 opacity-25"
          style={{
            backgroundImage:
              "radial-gradient(1px 1px at 20% 30%, #fff 100%, transparent), radial-gradient(1px 1px at 70% 25%, #00f0ff 100%, transparent), radial-gradient(1px 1px at 40% 70%, #fff 100%, transparent)",
            backgroundSize: "320px 320px",
          }}
        />
      </div>

      <div className="max-w-[1440px] mx-auto w-full relative z-10 flex flex-col justify-between">
        {/* Section Header Row matching Reference Panel 05 */}
        <div className="flex items-center justify-between pb-3 border-b border-slate-800/60 mb-8">
          <div className="flex items-center gap-3">
            <span className="text-xs font-mono font-bold text-[#00f0ff] tracking-[0.25em] uppercase">
              05
            </span>
            <span className="text-slate-600 font-mono text-xs">/</span>
            <span className="text-xs font-mono tracking-[0.2em] text-slate-400 uppercase">
              0:22 — 0:27 INVESTIGATION PIPELINE
            </span>
          </div>

          <div className="hidden sm:flex items-center gap-2 font-mono text-xs text-[#00f0ff]">
            <span className="w-2 h-2 rounded-full bg-[#00f0ff] animate-ping" />
            <span className="tracking-widest uppercase">
              STAGE {NODES[activeNode].number}: {NODES[activeNode].title} ACTIVE
            </span>
          </div>
        </div>

        {/* Section Title */}
        <div className="text-center max-w-3xl mx-auto mb-16">
          <h2 className="text-3xl sm:text-5xl lg:text-6xl font-extrabold tracking-tight text-white mb-4 uppercase font-sans">
            INVESTIGATION <span className="text-[#00f0ff]">PIPELINE</span>
          </h2>
          <p className="text-slate-400 text-xs sm:text-base leading-relaxed font-light">
            A deterministic multi-stage perception framework translating unstructured human intent
            into verified spatial truth.
          </p>
        </div>

        {/* 4 Connected Spatial Nodes with Traveling Cyan Connection Line */}
        <div className="relative mb-16">
          {/* Desktop Connecting Line between cards with physical drawing effect */}
          <div className="hidden lg:block absolute top-1/2 left-[10%] right-[10%] -translate-y-1/2 h-[3px] bg-slate-800 pointer-events-none z-0">
            {/* Active Cyan Traveling Pulse */}
            <div
              className="h-full bg-gradient-to-r from-cyan-600 via-[#00f0ff] to-[#00f0ff] transition-all duration-300 shadow-[0_0_16px_#00f0ff]"
              style={{
                width: `${linePercentage}%`,
              }}
            />
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 relative z-10">
            {NODES.map((node, index) => {
              const Icon = node.icon;
              const isActive = index === activeNode;
              const isPast = index < activeNode;
              const isHovered = hoveredNode === index;

              // Spatial depth arc positioning over horizon dome (Requirement 12)
              const depthArcZ = index === 1 || index === 2 ? 18 : -15;
              const depthArcY = index === 1 || index === 2 ? -8 : 8;
              const nodeZ = isActive ? depthArcZ + 40 : depthArcZ;
              const nodeScale = isActive ? 1.05 : 0.98;

              return (
                <div
                  key={node.number}
                  onClick={() => setActiveNode(index)}
                  onMouseEnter={() => setHoveredNode(index)}
                  onMouseLeave={() => setHoveredNode(null)}
                  data-cursor="interactive"
                  className={`group relative p-7 rounded-2xl bg-[#040c18]/90 border transition-all duration-500 backdrop-blur-xl flex flex-col justify-between min-h-[290px] cursor-pointer ${
                    isActive
                      ? "border-cyan-400 shadow-[0_0_40px_rgba(0,240,255,0.4)] bg-[#051426]/95 z-20"
                      : isHovered
                      ? "border-cyan-500/60 bg-[#041020] z-10"
                      : isPast
                      ? "border-cyan-900/60 bg-[#030914]/80 opacity-80"
                      : "border-slate-800/80 hover:border-slate-700 opacity-60"
                  }`}
                  style={{
                    transform: `perspective(1000px) translateY(${depthArcY}px) translateZ(${nodeZ}px) scale(${nodeScale})`,
                    transformStyle: "preserve-3d",
                  }}
                >
                  {/* Top Bar: Number + Completed Marker Checkmark */}
                  <div className="flex items-center justify-between mb-6">
                    <span
                      className={`text-sm font-mono font-bold transition-colors ${
                        isActive
                          ? "text-[#00f0ff]"
                          : isPast
                          ? "text-cyan-400"
                          : "text-slate-500"
                      }`}
                    >
                      {node.number}
                    </span>

                    <div
                      className={`w-6 h-6 rounded-full flex items-center justify-center transition-all ${
                        isActive || isPast
                          ? "bg-[#00f0ff]/20 text-[#00f0ff] border border-cyan-400/80 shadow-[0_0_12px_rgba(0,240,255,0.5)]"
                          : "bg-slate-800 text-slate-500"
                      }`}
                    >
                      <CheckCircle2 className="w-3.5 h-3.5" />
                    </div>
                  </div>

                  {/* Icon & Title */}
                  <div>
                    <div
                      className={`w-12 h-12 rounded-xl flex items-center justify-center mb-5 transition-all ${
                        isActive
                          ? "bg-[#00f0ff]/25 text-[#00f0ff] border border-cyan-400/80 shadow-[0_0_20px_rgba(0,240,255,0.45)] scale-110"
                          : "bg-[#061222] border border-slate-800 text-slate-400 group-hover:text-cyan-300"
                      }`}
                    >
                      <Icon className="w-6 h-6" />
                    </div>

                    <h3
                      className={`text-xl font-bold font-mono tracking-wider mb-2 uppercase transition-colors ${
                        isActive ? "text-[#00f0ff]" : "text-white group-hover:text-[#00f0ff]"
                      }`}
                    >
                      {node.title}
                    </h3>
                    <p
                      className={`text-xs sm:text-sm leading-relaxed font-light transition-colors ${
                        isActive || isHovered ? "text-slate-200" : "text-slate-400"
                      }`}
                    >
                      {node.desc}
                    </p>
                  </div>

                  {/* Bottom Active Glow Accent */}
                  <div
                    className={`h-0.5 rounded-full mt-6 transition-all duration-500 ${
                      isActive
                        ? "bg-[#00f0ff] shadow-[0_0_10px_#00f0ff]"
                        : isPast
                        ? "bg-cyan-800/80"
                        : "bg-transparent group-hover:bg-slate-700"
                    }`}
                  />
                </div>
              );
            })}
          </div>
        </div>

        {/* Bottom Subtitle matching Reference Panel 05 */}
        <div className="text-center">
          <span className="text-xs font-mono tracking-[0.28em] text-slate-400 uppercase">
            A COMPLETE INVESTIGATION LIFECYCLE
          </span>
        </div>
      </div>
    </section>
  );
}
