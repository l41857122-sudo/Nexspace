"use client";

import React, { useState, useEffect, useRef, useMemo, useCallback } from "react";
import Image from "next/image";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import {
  ArrowLeft,
  ArrowRight,
  Shield,
  Satellite,
  Radio,
  ChevronDown,
  Crosshair
} from "lucide-react";

interface TeamMember {
  id: string;
  num: string;
  name: string;
  role: string;
  department: string;
  dossierId: string;
  coordinates: string;
  satellitePass: string;
  image: string;
  isPlaceholder?: boolean;
  bio: string;
  skills: string[];
  accentMetric: string;
}

const TEAM_MEMBERS: TeamMember[] = [
  {
    id: "praval",
    num: "01",
    name: "PRAVAL SRIVASTAV",
    role: "FOUNDER & LEAD ARCHITECT",
    department: "SYSTEM ARCHITECTURE & NEURAL PIPELINES",
    dossierId: "NX-DOSSIER-01 // SIGMA-CORE",
    coordinates: "37.7749° N, 122.4194° W",
    satellitePass: "SENTINEL-2A // ORBIT 98.2°",
    image: "/team/praval.png",
    bio: "Pioneering autonomous remote sensing foundation models and distributed execution graphs that translate petabytes of raw earth observation data into verified real-world intelligence.",
    skills: ["Spatial AI", "Autonomous Systems", "Neural Graphs", "Orbital Geocoding"],
    accentMetric: "MODEL RES: 0.3M // LATENCY: 240MS",
  },
  {
    id: "lakshya",
    num: "02",
    name: "LAKSHYA VERMA",
    role: "CO-FOUNDER & HEAD OF COMPUTER VISION",
    department: "REMOTE SENSING & OPTICAL FUSION",
    dossierId: "NX-DOSSIER-02 // OPTIC-SAR",
    coordinates: "28.6139° N, 77.2090° E",
    satellitePass: "LANDSAT-9 // ORBIT 97.4°",
    image: "/team/lakshya.png",
    bio: "Leading multispectral vision algorithms and zero-shot geospatial models capable of sub-meter change forensics, optical sensor fusion, and micro-anomaly detection across global constellations.",
    skills: ["Multispectral Vision", "Zero-Shot Detection", "Change Forensics", "Edge Inference"],
    accentMetric: "SPECTRAL BANDS: 13 // PRECISION: 99.4%",
  },
  {
    id: "nandini",
    num: "03",
    name: "NANDINI SINGH GAUR",
    role: "CO-FOUNDER & LEAD GEOSPATIAL ENGINEER",
    department: "EARTH INTELLIGENCE & TOPOLOGY",
    dossierId: "NX-DOSSIER-03 // GEO-TOPOLOGY",
    coordinates: "12.9716° N, 77.5946° E",
    satellitePass: "SENTINEL-1 SAR // ORBIT 98.7°",
    image: "/team/nandini.png",
    bio: "Directing high-throughput geospatial ingestion pipelines, vector topological grounding, and automated verification protocols bridging raw orbital telemetry to verifiable geometric truth.",
    skills: ["Geospatial Indexing", "Vector Grounding", "SAR Telemetry", "GIS Architecture"],
    accentMetric: "INGESTION: 18.4 TB/DAY // TOPOLOGY: L3",
  },
  {
    id: "naina",
    num: "04",
    name: "NAINA MISRA",
    role: "HEAD OF PRODUCT STRATEGY & SPATIAL INTEL",
    department: "MISSION ANALYTICS & PRODUCT",
    dossierId: "NX-DOSSIER-04 // MISSION-INTEL",
    coordinates: "51.5074° N, 0.1278° W",
    satellitePass: "WORLDVIEW-3 // ORBIT 96.8°",
    image: "/team/naina.png",
    bio: "Shaping the interactive intelligence paradigm at NexSpace, translating complex orbital telemetry and AI verification into mission-critical discovery workflows for global decision-makers.",
    skills: ["Product Architecture", "Mission Analytics", "Spatial UX", "Domain Ontology"],
    accentMetric: "DISCOVERY WORKFLOWS: VERIFIED 24/7",
  },
  {
    id: "prasuk",
    num: "05",
    name: "PRASUK JAIN",
    role: "LEAD SYSTEMS & INFRASTRUCTURE ENGINEER",
    department: "DISTRIBUTED PLATFORM & CLUSTERS",
    dossierId: "NX-DOSSIER-05 // HYPER-CLUSTER",
    coordinates: "40.7128° N, 74.0060° W",
    satellitePass: "GEO-SYNC CONSTELLATION // 99.1°",
    image: "/team/prasuk.png",
    bio: "Building resilient, ultra-low-latency backend infrastructure capable of querying global satellite constellations, orchestrating multi-agent reasoning, and streaming vector proofs under extreme load.",
    skills: ["Distributed Systems", "GPU Clusters", "Real-Time Telemetry", "API Gateway"],
    accentMetric: "NODE UPTIME: 99.998% // P99: 42MS",
  },
  {
    id: "piyush",
    num: "06",
    name: "PIYUSH GUPTA",
    role: "SPATIAL INTELLIGENCE & MISSION RESEARCH",
    department: "ORBITAL REASONING & RESEARCH",
    dossierId: "NX-DOSSIER-06 // ORBITAL-RESEARCH",
    coordinates: "35.6762° N, 139.6503° E",
    satellitePass: "EARTH EXPLORER // ORBIT 98.0°",
    image: "/team/piyush.png",
    bio: "Spearheading planetary-scale predictive intelligence frameworks, sensor calibration models, and orbital reasoning algorithms for environmental monitoring and global climate resilience.",
    skills: ["Spatial Grounding", "Constellation Modeling", "Earth Observation", "Planetary Analytics"],
    accentMetric: "ORBITAL REASONING // SENSOR SYNC",
  },
];

// Continuous scroll timeline ranges per member
const MEMBER_RANGES = [
  { start: 0.12, enterEnd: 0.18, leaveStart: 0.25, end: 0.30 }, // 01 Praval
  { start: 0.25, enterEnd: 0.31, leaveStart: 0.38, end: 0.43 }, // 02 Lakshya
  { start: 0.38, enterEnd: 0.44, leaveStart: 0.51, end: 0.56 }, // 03 Nandini
  { start: 0.51, enterEnd: 0.57, leaveStart: 0.64, end: 0.69 }, // 04 Naina
  { start: 0.64, enterEnd: 0.70, leaveStart: 0.77, end: 0.82 }, // 05 Prasuk
  { start: 0.77, enterEnd: 0.83, leaveStart: 0.89, end: 0.93 }, // 06 Piyush
];

// Multi-layer interpolation calculation
function getMemberTransition(p: number, idx: number) {
  const r = MEMBER_RANGES[idx];
  if (!r) return { visible: false, opacity: 0, cardOpacity: 0, cardTranslateY: 0, scale: 0.94, blur: 8, translateX: 30, numTranslateX: 50 };

  if (p < r.start || p > r.end) {
    return {
      visible: false,
      opacity: 0,
      cardOpacity: 0,
      cardTranslateY: p < r.start ? 20 : -20,
      scale: p < r.start ? 0.94 : 0.96,
      blur: 8,
      translateX: p < r.start ? 30 : -30,
      numTranslateX: p < r.start ? 50 : -50,
    };
  }

  // Entering phase
  if (p < r.enterEnd) {
    const rawT = (p - r.start) / (r.enterEnd - r.start);
    const t = Math.max(0, Math.min(1, rawT));
    const ease = t * t * (3 - 2 * t);
    const cardT = Math.max(0, Math.min(1, (t - 0.25) / 0.75));
    const cardEase = cardT * cardT * (3 - 2 * cardT);
    return {
      opacity: ease,
      cardOpacity: cardEase,
      cardTranslateY: 18 * (1 - cardEase),
      scale: 0.94 + 0.06 * ease,
      blur: 8 * (1 - ease),
      translateX: 24 * (1 - ease),
      numTranslateX: 45 * (1 - ease),
      visible: true,
    };
  }

  // Active stable phase
  if (p <= r.leaveStart) {
    return {
      opacity: 1,
      cardOpacity: 1,
      cardTranslateY: 0,
      scale: 1,
      blur: 0,
      translateX: 0,
      numTranslateX: 0,
      visible: true,
    };
  }

  // Leaving phase
  const rawT = (p - r.leaveStart) / (r.end - r.leaveStart);
  const t = Math.max(0, Math.min(1, rawT));
  const ease = t * t * (3 - 2 * t);
  const cardT = Math.max(0, Math.min(1, (0.75 - t) / 0.75));
  const cardEase = cardT * cardT * (3 - 2 * cardT);
  return {
    opacity: 1 - ease,
    cardOpacity: cardEase,
    cardTranslateY: -18 * (1 - cardEase),
    scale: 1 - 0.04 * ease,
    blur: 6 * ease,
    translateX: -24 * ease,
    numTranslateX: -45 * ease,
    visible: true,
  };
}

export default function AboutTeamSection() {
  const containerRef = useRef<HTMLDivElement>(null);
  const stageRef = useRef<HTMLDivElement>(null);
  const scrollTriggerRef = useRef<ScrollTrigger | null>(null);

  // Continuous scroll progress float [0.00 .. 1.00]
  const [scrollProgress, setScrollProgress] = useState(0);
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 });
  const [isReducedMotion, setIsReducedMotion] = useState(false);

  // Progress anchors for jumping cleanly to each stage's active center
  const progressMap = useMemo(
    () => [
      0.07,  // 0: INTRO
      0.215, // 1: PRAVAL
      0.345, // 2: LAKSHYA
      0.475, // 3: NANDINI
      0.605, // 4: NAINA
      0.735, // 5: PRASUK
      0.855, // 6: PIYUSH
      0.96,  // 7: FINAL TEAM
    ],
    []
  );

  // Current active step index for UI indicators (0 to 7)
  const activeStep = useMemo(() => {
    if (scrollProgress < 0.15) return 0;
    if (scrollProgress < 0.28) return 1;
    if (scrollProgress < 0.41) return 2;
    if (scrollProgress < 0.54) return 3;
    if (scrollProgress < 0.67) return 4;
    if (scrollProgress < 0.80) return 5;
    if (scrollProgress < 0.91) return 6;
    return 7;
  }, [scrollProgress]);

  // Reduced motion preference
  useEffect(() => {
    if (typeof window !== "undefined") {
      const mediaQuery = window.matchMedia("(prefers-reduced-motion: reduce)");
      setIsReducedMotion(mediaQuery.matches);
      const listener = (e: MediaQueryListEvent) => setIsReducedMotion(e.matches);
      mediaQuery.addEventListener("change", listener);
      return () => mediaQuery.removeEventListener("change", listener);
    }
  }, []);

  // Subtle mouse tracking for gentle parallax
  const handleMouseMove = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      if (isReducedMotion) return;
      const rect = e.currentTarget.getBoundingClientRect();
      const x = ((e.clientX - rect.left) / rect.width - 0.5) * 2;
      const y = ((e.clientY - rect.top) / rect.height - 0.5) * 2;
      setMousePos({ x, y });
    },
    [isReducedMotion]
  );

  // High-performance continuous scroll update
  useEffect(() => {
    gsap.registerPlugin(ScrollTrigger);

    const container = containerRef.current;
    if (!container) return;

    const updateScrollProgress = (progVal?: number) => {
      let p: number;
      if (typeof progVal === "number") {
        p = progVal;
      } else {
        const rect = container.getBoundingClientRect();
        const totalScrollable = container.offsetHeight - window.innerHeight;
        if (totalScrollable <= 0) return;
        const currentY = -rect.top;
        p = Math.max(0, Math.min(1, currentY / totalScrollable));
      }
      setScrollProgress(p);
    };

    // GSAP ScrollTrigger synchronization
    const st = ScrollTrigger.create({
      trigger: container,
      start: "top top",
      end: "bottom bottom",
      scrub: 0.1,
      onUpdate: (self) => {
        updateScrollProgress(self.progress);
      },
    });

    scrollTriggerRef.current = st;

    // Native scroll listener backup for instant responsiveness
    const onScroll = () => {
      updateScrollProgress();
    };
    window.addEventListener("scroll", onScroll, { passive: true });

    return () => {
      st.kill();
      scrollTriggerRef.current = null;
      window.removeEventListener("scroll", onScroll);
    };
  }, []);

  // Programmatic jump to a step's active center (secondary navigation)
  const jumpToStep = useCallback(
    (stepIndex: number) => {
      const container = containerRef.current;
      if (container) {
        const targetProgress = progressMap[stepIndex] ?? 0.07;
        const containerTop = container.getBoundingClientRect().top + window.scrollY;
        const totalScrollable = container.offsetHeight - window.innerHeight;
        const targetScrollY = containerTop + targetProgress * totalScrollable;

        if (typeof window !== "undefined") {
          const lenis = (window as any).__lenis;
          if (lenis) {
            lenis.scrollTo(targetScrollY, { duration: 0.9 });
          } else {
            window.scrollTo({ top: targetScrollY, behavior: "smooth" });
          }
        }
      }
    },
    [progressMap]
  );

  const nextMember = useCallback(() => {
    if (activeStep < 7) {
      jumpToStep(activeStep + 1);
    }
  }, [activeStep, jumpToStep]);

  const prevMember = useCallback(() => {
    if (activeStep > 0) {
      jumpToStep(activeStep - 1);
    }
  }, [activeStep, jumpToStep]);

  // Interpolation metrics for INTRO (0 to 15%)
  const introTransform = useMemo(() => {
    if (scrollProgress >= 0.15) {
      return { opacity: 0, translateY: -35, scale: 0.95, visible: false };
    }
    if (scrollProgress <= 0.09) {
      return { opacity: 1, translateY: 0, scale: 1, visible: true };
    }
    const t = (scrollProgress - 0.09) / 0.06;
    const ease = t * t * (3 - 2 * t);
    return {
      opacity: 1 - ease,
      translateY: -30 * ease,
      scale: 1 - 0.04 * ease,
      visible: true,
    };
  }, [scrollProgress]);

  // Interpolation metrics for FINAL TEAM COHORT (90% to 100%)
  const cohortTransform = useMemo(() => {
    if (scrollProgress < 0.88) {
      return { opacity: 0, translateY: 30, scale: 0.96, visible: false };
    }
    if (scrollProgress >= 0.93) {
      return { opacity: 1, translateY: 0, scale: 1, visible: true };
    }
    const t = (scrollProgress - 0.88) / 0.05;
    const ease = t * t * (3 - 2 * t);
    return {
      opacity: ease,
      translateY: 25 * (1 - ease),
      scale: 0.96 + 0.04 * ease,
      visible: true,
    };
  }, [scrollProgress]);

  return (
    <section
      ref={containerRef}
      id="about"
      aria-label="About NexSpace Team"
      className="relative w-full bg-[#01040a] text-slate-100 select-none z-20"
      style={{ height: "620vh" }}
      onMouseMove={handleMouseMove}
    >
      {/* ========================================================= */}
      {/* STICKY FULL-SCREEN TEAM STAGE (100vh) */}
      {/* ========================================================= */}
      <div
        ref={stageRef}
        className="sticky top-0 h-screen w-full overflow-hidden flex flex-col justify-between border-t border-cyan-500/10 bg-[#01040a]"
      >
        {/* ------------------------------------------------------- */}
        {/* CINEMATIC BACKGROUND: High-Res Earth & Satellite Panorama */}
        {/* Exactly matching Image 2 visual direction                */}
        {/* ------------------------------------------------------- */}
        <div className="absolute inset-0 pointer-events-none overflow-hidden" aria-hidden="true">
          {/* Panoramic Earth & Satellite Background */}
          <div
            className="absolute inset-0 w-full h-full pointer-events-none transition-transform duration-700 ease-out"
            style={{
              transform: isReducedMotion
                ? "none"
                : `translate3d(${mousePos.x * 4}px, ${(scrollProgress - 0.5) * -30 + mousePos.y * 4}px, 0) scale(1.02)`,
            }}
          >
            <Image
              src="/team/earth_cinematic_bg.png"
              alt="Orbital Earth Horizon with Satellite"
              fill
              className="object-cover object-center filter saturate-[1.1] contrast-[1.05] brightness-[0.72]"
              priority
            />
          </div>

          {/* Deep dark gradient overlay for optimal text contrast */}
          <div className="absolute inset-0 bg-gradient-to-t from-[#01040a]/70 via-transparent to-[#01040a]/40 pointer-events-none" />

          {/* Geographic Grid Overlay */}
          <div className="absolute inset-0 bg-[linear-gradient(to_right,#00f0ff06_1px,transparent_1px),linear-gradient(to_bottom,#00f0ff06_1px,transparent_1px)] bg-[size:6rem_6rem] [mask-image:radial-gradient(ellipse_75%_65%_at_50%_50%,#000_60%,transparent_100%)] opacity-30 pointer-events-none" />

          {/* Atmospheric Cyan Rim Glow */}
          <div className="absolute bottom-0 left-1/2 -translate-x-1/2 w-[120vw] h-[35vw] rounded-[100%] bg-[radial-gradient(ellipse_at_center,rgba(0,240,255,0.08)_0%,rgba(56,189,248,0.03)_50%,transparent_75%)] blur-[70px] pointer-events-none" />
        </div>

        {/* ------------------------------------------------------- */}
        {/* TOP STATUS BAR: Dossier Header & Numbered Pills (Image 2)*/}
        {/* ------------------------------------------------------- */}
        <header className="relative z-30 w-full pt-16 sm:pt-20 px-6 sm:px-12 max-w-[1540px] mx-auto flex items-center justify-between pointer-events-auto">
          {/* Left Metadata (Image 2) */}
          <div className="flex flex-col gap-0.5">
            <span className="font-mono text-[9px] sm:text-[10px] text-cyan-400/60 tracking-[0.25em] uppercase">
              NEXSPACE / PEOPLE // HUMAN INTELLIGENCE × SPATIAL AI
            </span>
            <div className="flex items-center gap-2">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[#00f0ff] opacity-75" />
                <span className="relative inline-flex rounded-full h-2 w-2 bg-[#00f0ff]" />
              </span>
              <span className="font-mono text-xs sm:text-[13px] tracking-wider text-cyan-300 font-bold uppercase">
                NEXSPACE / PEOPLE
              </span>
              <span className="text-slate-600 hidden sm:inline">|</span>
              <span className="font-mono text-[10px] sm:text-[11px] tracking-wider text-slate-400 hidden sm:inline uppercase">
                SCROLL DRIVEN DOSSIER SEQUENCE
              </span>
            </div>
          </div>

          {/* Center: Numbered Indicator Pills (Image 2) */}
          <div className="flex items-center gap-1.5 sm:gap-2 overflow-x-auto max-w-full py-1">
            <button
              onClick={() => jumpToStep(0)}
              className={`font-mono text-[10px] sm:text-xs px-3 py-1 rounded-full transition-all duration-300 border cursor-pointer ${
                activeStep === 0
                  ? "border-cyan-400 bg-cyan-950/70 text-[#00f0ff] font-bold shadow-[0_0_15px_rgba(0,240,255,0.4)] scale-105"
                  : "border-slate-800 bg-[#020914]/60 text-slate-400 hover:text-white hover:border-slate-700"
              }`}
            >
              INTRO
            </button>

            {TEAM_MEMBERS.map((member, idx) => {
              const stepIdx = idx + 1;
              const isActive = activeStep === stepIdx;
              return (
                <button
                  key={member.id}
                  onClick={() => jumpToStep(stepIdx)}
                  title={`${member.num} — ${member.name}`}
                  className={`font-mono text-[10px] sm:text-xs px-2.5 sm:px-3 py-1 rounded-full transition-all duration-300 border cursor-pointer ${
                    isActive
                      ? "border-cyan-400 bg-cyan-950/70 text-[#00f0ff] font-bold shadow-[0_0_16px_rgba(0,240,255,0.5)] scale-105"
                      : "border-slate-800 bg-[#020914]/50 text-slate-400 hover:text-white hover:border-cyan-500/40"
                  }`}
                >
                  {member.num}
                </button>
              );
            })}

            <button
              onClick={() => jumpToStep(7)}
              className={`font-mono text-[10px] sm:text-xs px-3 py-1 rounded-full transition-all duration-300 border cursor-pointer ${
                activeStep === 7
                  ? "border-cyan-400 bg-cyan-950/70 text-[#00f0ff] font-bold shadow-[0_0_15px_rgba(0,240,255,0.4)] scale-105"
                  : "border-slate-800 bg-[#020914]/60 text-slate-400 hover:text-white hover:border-slate-700"
              }`}
            >
              TEAM
            </button>
          </div>

          {/* Right: Live Telemetry Indicator (Image 2) */}
          <div className="font-mono text-[10px] sm:text-xs text-cyan-400/80 tracking-widest uppercase hidden lg:block">
            ORBITAL TELEMETRY: ACTIVE // SENTINEL-2A
          </div>
        </header>

        {/* ------------------------------------------------------- */}
        {/* MAIN CINEMATIC VIEWPORT STAGE */}
        {/* ------------------------------------------------------- */}
        <main className="relative z-20 flex-1 w-full max-w-[1540px] mx-auto px-6 sm:px-12 flex items-center justify-center overflow-hidden">
          {/* ===================================================== */}
          {/* STATE 00: INTRO SLIDE (0% to 15%)                     */}
          {/* ===================================================== */}
          {introTransform.visible && (
            <div
              className="absolute inset-0 flex flex-col items-center justify-center text-center px-6 transition-all duration-300 pointer-events-auto"
              style={{
                opacity: introTransform.opacity,
                transform: `translate3d(0, ${introTransform.translateY}px, 0) scale(${introTransform.scale})`,
              }}
            >
              {/* Giant Background TEAM Typography */}
              <div
                className="absolute inset-0 flex items-center justify-center pointer-events-none select-none z-0"
                style={{
                  transform: isReducedMotion
                    ? "none"
                    : `translate3d(${mousePos.x * -6}px, ${mousePos.y * -6}px, 0)`,
                }}
              >
                <span className="text-[28vw] font-black tracking-[-0.08em] text-white/[0.04] leading-none uppercase select-none">
                  TEAM
                </span>
              </div>

              {/* Dossier Badge */}
              <div className="inline-flex items-center gap-2 px-3.5 py-1 rounded-full bg-cyan-950/40 border border-cyan-500/30 backdrop-blur-md mb-6 shadow-[0_0_20px_rgba(0,240,255,0.15)] relative z-10">
                <Radio className="w-3.5 h-3.5 text-[#00f0ff] animate-pulse" />
                <span className="font-mono text-[10px] sm:text-xs text-cyan-300 tracking-[0.2em] uppercase">
                  NEXSPACE / PEOPLE // DOSSIER SEQUENCE
                </span>
              </div>

              {/* Main Headline */}
              <h2 className="text-4xl sm:text-6xl md:text-7xl lg:text-8xl font-black tracking-tight text-white uppercase max-w-4xl leading-[1.05] relative z-10">
                THE PEOPLE <br />
                <span className="text-transparent bg-clip-text bg-gradient-to-r from-white via-cyan-100 to-cyan-400">
                  BEHIND NEXSPACE
                </span>
              </h2>

              <p className="mt-5 text-sm sm:text-base md:text-lg text-slate-300 max-w-2xl font-light leading-relaxed relative z-10">
                Six minds converging at the nexus of orbital remote sensing, planetary neural networks, and spatial artificial intelligence.
              </p>

              {/* Supporting Micro-Metadata */}
              <div className="mt-7 flex flex-wrap items-center justify-center gap-3 font-mono text-[10px] sm:text-xs text-slate-400 relative z-10">
                <span className="px-3 py-1 rounded bg-[#030914] border border-slate-800">
                  HUMAN INTELLIGENCE × SPATIAL AI
                </span>
                <span className="px-3 py-1 rounded bg-[#030914] border border-cyan-500/30 text-cyan-300">
                  06 INDIVIDUAL OPERATIVE DOSSIERS
                </span>
              </div>

              {/* Scroll Callout */}
              <div className="mt-10 sm:mt-12 flex flex-col items-center gap-2 relative z-10">
                <div className="flex items-center gap-2 font-mono text-[10px] sm:text-xs text-cyan-300 tracking-[0.25em] uppercase">
                  <span>SCROLL TO EXPLORE</span>
                  <ChevronDown className="w-3.5 h-3.5 text-[#00f0ff] animate-bounce" />
                </div>
                <span className="font-mono text-[9px] text-slate-500 tracking-widest uppercase">
                  Normal mouse wheel or trackpad controls the experience
                </span>
              </div>
            </div>
          )}

          {/* ===================================================== */}
          {/* STATES 01–06: INDIVIDUAL MEMBER DOSSIER SLIDES        */}
          {/* EXACT IMAGE 2 LAYOUT: Card on Left, Portrait Right,   */}
          {/* Giant Outlined Number Spanning Across Center          */}
          {/* ===================================================== */}
          {TEAM_MEMBERS.map((member, idx) => {
            const tr = getMemberTransition(scrollProgress, idx);
            if (!tr.visible) return null;

            return (
              <div
                key={member.id}
                className="absolute inset-0 flex items-center justify-center pointer-events-auto transition-opacity duration-150"
                style={{
                  opacity: tr.opacity,
                  pointerEvents: tr.opacity > 0.5 ? "auto" : "none",
                }}
              >
                {/* 2-Column Split: Card on Left, Portrait in Center-Right */}
                <div className="w-full flex flex-col lg:flex-row items-center justify-between gap-8 lg:gap-14 relative">
                  {/* ----------------------------------------------- */}
                  {/* LEFT: INFORMATION DOSSIER CARD (IMAGE 2)        */}
                  {/* ----------------------------------------------- */}
                  <div
                    className="w-full lg:w-[480px] xl:w-[530px] shrink-0 flex flex-col justify-center relative z-20"
                    style={{
                      opacity: tr.cardOpacity,
                      transform: isReducedMotion
                        ? "none"
                        : `translate3d(${tr.translateX * 0.4 + mousePos.x * 3}px, ${tr.cardTranslateY + mousePos.y * 3}px, 0)`,
                    }}
                  >
                    <div className="bg-[#020a16]/85 border border-cyan-500/35 rounded-2xl p-6 sm:p-8 backdrop-blur-xl shadow-[0_20px_60px_rgba(0,0,0,0.85),0_0_30px_rgba(0,240,255,0.08)] relative overflow-hidden group">
                      {/* Subtle Corner Tech Accents */}
                      <div className="absolute top-0 left-0 w-3 h-3 border-t-2 border-l-2 border-[#00f0ff]/80" />
                      <div className="absolute top-0 right-0 w-3 h-3 border-t-2 border-r-2 border-[#00f0ff]/80" />
                      <div className="absolute bottom-0 left-0 w-3 h-3 border-b-2 border-l-2 border-[#00f0ff]/80" />
                      <div className="absolute bottom-0 right-0 w-3 h-3 border-b-2 border-r-2 border-[#00f0ff]/80" />

                      {/* Header Meta: Operative Number & Satellite Pass */}
                      <div className="flex items-center justify-between mb-4">
                        <div className="inline-flex items-center gap-2 font-mono text-[11px] tracking-[0.2em] text-[#00f0ff] uppercase font-bold">
                          <Crosshair className="w-3.5 h-3.5" />
                          <span>OPERATIVE {member.num} / 06</span>
                        </div>
                        <div className="font-mono text-[10px] text-slate-400 tracking-wider">
                          {member.satellitePass}
                        </div>
                      </div>

                      {/* Member Name: White Uppercase */}
                      <h3 className="text-2xl sm:text-3xl lg:text-4xl font-black tracking-tight text-white uppercase leading-none mb-2">
                        {member.name}
                      </h3>

                      {/* Member Role: NexSpace Aqua / Cyan */}
                      <div className="text-xs sm:text-sm font-mono text-[#00f0ff] font-bold tracking-wide uppercase mb-4 flex items-center gap-2">
                        <span className="w-2 h-2 rounded-full bg-[#00f0ff] shadow-[0_0_8px_#00f0ff]" />
                        <span>{member.role}</span>
                      </div>

                      {/* Department & Telemetry Tagline */}
                      <div className="font-mono text-[10px] text-slate-400 tracking-[0.12em] uppercase mb-4 pb-3 border-b border-slate-800/80 flex items-center justify-between gap-2">
                        <span>DIV: {member.department}</span>
                        <span className="text-cyan-400 font-semibold">{member.accentMetric}</span>
                      </div>

                      {/* Biography */}
                      <p className="text-xs sm:text-sm text-slate-300 font-light leading-relaxed mb-6">
                        {member.bio}
                      </p>

                      {/* Skills & Contribution Tags */}
                      <div>
                        <div className="font-mono text-[9px] text-slate-400 tracking-widest uppercase mb-2.5">
                          DOMAIN EXPERTISE & ATTRIBUTION:
                        </div>
                        <div className="flex flex-wrap gap-2">
                          {member.skills.map((skill) => (
                            <span
                              key={skill}
                              className="px-3 py-1 rounded-full bg-cyan-950/40 border border-cyan-500/40 text-cyan-200 font-mono text-[10px] sm:text-[11px] tracking-wide"
                            >
                              {skill}
                            </span>
                          ))}
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* ----------------------------------------------- */}
                  {/* CENTER / RIGHT: PORTRAIT + GIANT NUMBER BEHIND  */}
                  {/* ----------------------------------------------- */}
                  <div
                    className="flex-1 flex items-center justify-center relative min-h-[460px] sm:min-h-[540px] lg:min-h-[580px]"
                    style={{
                      transform: isReducedMotion
                        ? "none"
                        : `translate3d(${tr.translateX * -0.3 + mousePos.x * 6}px, ${mousePos.y * 6}px, 0) scale(${tr.scale})`,
                      filter: `blur(${tr.blur}px)`,
                    }}
                  >
                    {/* LAYER 2: GIANT OUTLINED NUMBER BEHIND PORTRAIT & CARD */}
                    {/* Spanning across center-stage exactly as shown in Image 2 */}
                    <div
                      className="absolute inset-0 flex items-center justify-center pointer-events-none select-none z-0"
                      style={{
                        transform: `translate3d(${tr.numTranslateX}px, 0, 0)`,
                      }}
                    >
                      <span className="text-[20rem] sm:text-[24rem] lg:text-[28rem] font-black tracking-tighter text-transparent [-webkit-text-stroke:2px_rgba(0,240,255,0.22)] leading-none select-none">
                        {member.num}
                      </span>
                    </div>

                    {/* LAYER 5: ORBITAL GRAPHICS (Surrounding Portrait) */}
                    <div
                      className="absolute w-[340px] h-[340px] sm:w-[420px] sm:h-[420px] rounded-full border border-cyan-500/20 animate-[spin_45s_linear_infinite] pointer-events-none shadow-[0_0_50px_rgba(0,240,255,0.08)] z-5"
                      style={{
                        transform: isReducedMotion
                          ? "none"
                          : `translate3d(${mousePos.x * 6}px, ${mousePos.y * 6}px, 0)`,
                      }}
                    />
                    <div
                      className="absolute w-[280px] h-[280px] sm:w-[360px] sm:h-[360px] rounded-full border border-dashed border-sky-400/20 animate-[spin_32s_linear_infinite_reverse] pointer-events-none z-5"
                      style={{
                        transform: isReducedMotion
                          ? "none"
                          : `translate3d(${mousePos.x * 4}px, ${mousePos.y * 4}px, 0)`,
                      }}
                    />

                    {/* LAYER 3: NORMALIZED PORTRAIT CONTAINER */}
                    <div className="relative z-10 w-[270px] sm:w-[330px] md:w-[370px] lg:w-[410px] h-[380px] sm:h-[470px] md:h-[530px] lg:h-[570px] max-h-[66vh] flex items-end justify-center group">
                      {/* Dossier ID Badge Above Head (Image 2) */}
                      <div className="absolute -top-7 left-1/2 -translate-x-1/2 font-mono text-[9px] text-[#00f0ff] tracking-widest bg-[#020b18]/90 border border-cyan-500/40 px-2.5 py-0.5 rounded backdrop-blur-sm pointer-events-none shadow-[0_0_12px_rgba(0,240,255,0.25)] whitespace-nowrap z-20">
                        {member.dossierId}
                      </div>

                      {member.isPlaceholder ? (
                        <div className="w-full h-full flex flex-col items-center justify-center rounded-2xl border border-cyan-500/35 bg-[#020a16]/75 backdrop-blur-md relative overflow-hidden shadow-[0_0_50px_rgba(0,240,255,0.15)]">
                          {/* Tactical Radar Reticle */}
                          <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                            <div className="w-64 h-64 rounded-full border border-cyan-500/20 border-dashed animate-[spin_30s_linear_infinite]" />
                            <div className="w-48 h-48 rounded-full border border-cyan-400/25" />
                            <div className="w-32 h-32 rounded-full border border-[#00f0ff]/35 border-dotted animate-[spin_20s_linear_infinite_reverse]" />
                            <div className="w-16 h-16 rounded-full border border-[#00f0ff]/50" />
                            <div className="w-2.5 h-2.5 rounded-full bg-[#00f0ff] shadow-[0_0_16px_#00f0ff]" />
                          </div>
                          <div className="relative z-10 flex flex-col items-center gap-3 text-center px-6">
                            <div className="p-3 rounded-full bg-cyan-950/60 border border-cyan-500/40 shadow-[0_0_20px_rgba(0,240,255,0.3)]">
                              <Satellite className="w-8 h-8 text-[#00f0ff] animate-pulse" />
                            </div>
                            <span className="font-mono text-sm text-white font-bold tracking-widest uppercase">
                              PIYUSH GUPTA
                            </span>
                            <span className="font-mono text-[10px] text-[#00f0ff] tracking-wider px-2.5 py-0.5 rounded-full bg-cyan-950/40 border border-cyan-500/30">
                              OPERATIVE 06 // ACTIVE ORBIT DOSSIER
                            </span>
                            <span className="font-mono text-[9px] text-slate-400 max-w-[200px] leading-relaxed">
                              Planetary Predictive Intelligence & Environmental Modeling
                            </span>
                          </div>
                        </div>
                      ) : (
                        <div className="relative w-full h-full transition-transform duration-500 group-hover:scale-[1.018]">
                          <Image
                            src={member.image}
                            alt={member.name}
                            fill
                            className="object-contain object-bottom drop-shadow-[0_20px_45px_rgba(0,0,0,0.9)] filter contrast-105 group-hover:brightness-105 group-hover:drop-shadow-[0_0_35px_rgba(0,240,255,0.3)] transition-all duration-500"
                            priority
                            sizes="(max-width: 768px) 300px, (max-width: 1200px) 380px, 420px"
                          />
                        </div>
                      )}

                      {/* Gradient Fade at Portrait Baseline */}
                      <div className="absolute bottom-0 left-0 right-0 h-16 bg-gradient-to-t from-[#01040a]/90 to-transparent pointer-events-none" />

                      {/* Tactical Coordinate Badge (Image 2) */}
                      <div className="absolute bottom-6 right-2 font-mono text-[10px] text-slate-300 tracking-wider bg-[#020712]/90 border border-slate-700/80 px-2.5 py-1 rounded backdrop-blur-md pointer-events-none shadow-lg z-20 hidden sm:block">
                        {member.coordinates}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            );
          })}

          {/* ===================================================== */}
          {/* STATE 07: FINAL TEAM COHORT ENSEMBLE (90% to 100%)    */}
          {/* ===================================================== */}
          {cohortTransform.visible && (
            <div
              className="absolute inset-0 flex flex-col items-center justify-center text-center px-6 transition-all duration-300 pointer-events-auto"
              style={{
                opacity: cohortTransform.opacity,
                transform: `translate3d(0, ${cohortTransform.translateY}px, 0) scale(${cohortTransform.scale})`,
              }}
            >
              {/* Giant Background TEAM Typography */}
              <div
                className="absolute inset-0 flex items-center justify-center pointer-events-none select-none z-0"
                style={{
                  transform: isReducedMotion
                    ? "none"
                    : `translate3d(${mousePos.x * -4}px, ${mousePos.y * -4}px, 0)`,
                }}
              >
                <span className="text-[28vw] font-black tracking-[-0.08em] text-white/[0.04] leading-none uppercase">
                  TEAM
                </span>
              </div>

              {/* Sub-header Badge */}
              <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-cyan-950/40 border border-cyan-500/30 backdrop-blur-md mb-3 shadow-[0_0_20px_rgba(0,240,255,0.15)] relative z-10">
                <Shield className="w-3.5 h-3.5 text-[#00f0ff]" />
                <span className="font-mono text-[10px] sm:text-xs text-cyan-300 tracking-[0.2em] uppercase">
                  NEXSPACE COHORT // UNITED PERSPECTIVE
                </span>
              </div>

              {/* Main Heading: PEOPLE × TECHNOLOGY × EARTH × IMPACT */}
              <h2 className="text-2xl sm:text-4xl md:text-5xl font-black text-white uppercase tracking-tight mb-2 relative z-10">
                <span>PEOPLE</span>
                <span className="text-[#00f0ff] mx-2 font-light">×</span>
                <span>TECHNOLOGY</span>
                <span className="text-[#00f0ff] mx-2 font-light">×</span>
                <span>EARTH</span>
                <span className="text-[#00f0ff] mx-2 font-light">×</span>
                <span>IMPACT</span>
              </h2>

              <p className="text-xs sm:text-sm text-cyan-300 font-light max-w-xl mx-auto mb-6 sm:mb-8 relative z-10">
                Different perspectives. One clearer view of our planet.
              </p>

              {/* Panoramic 6-Member Lineup with Normalized Baseline & Height */}
              <div className="relative z-10 w-full max-w-6xl mx-auto grid grid-cols-3 sm:grid-cols-6 gap-3 sm:gap-4 items-end mb-6">
                {TEAM_MEMBERS.map((member, idx) => (
                  <div
                    key={member.id}
                    onClick={() => jumpToStep(idx + 1)}
                    className="group cursor-pointer flex flex-col items-center transition-all duration-300 hover:scale-105"
                  >
                    <div className="relative w-full h-[180px] sm:h-[260px] md:h-[300px] flex items-end justify-center rounded-xl bg-gradient-to-b from-[#020b14]/50 to-[#031122]/90 border border-slate-800 group-hover:border-cyan-400/60 overflow-hidden transition-all duration-300 shadow-[0_4px_20px_rgba(0,0,0,0.6)]">
                      {member.isPlaceholder ? (
                        <div className="w-full h-full flex flex-col items-center justify-center p-3 text-center relative overflow-hidden">
                          <div className="absolute inset-0 flex items-center justify-center opacity-30">
                            <div className="w-24 h-24 rounded-full border border-cyan-500/40 border-dashed animate-[spin_25s_linear_infinite]" />
                          </div>
                          <div className="relative z-10 flex flex-col items-center gap-1.5">
                            <Satellite className="w-6 h-6 text-[#00f0ff] animate-pulse" />
                            <span className="font-mono text-[9px] text-white font-bold tracking-wider">PIYUSH</span>
                            <span className="font-mono text-[7px] text-[#00f0ff] tracking-widest uppercase px-1.5 py-0.5 rounded bg-cyan-950/60 border border-cyan-500/30">
                              ORBITAL
                            </span>
                          </div>
                        </div>
                      ) : (
                        <Image
                          src={member.image}
                          alt={member.name}
                          fill
                          className="object-contain object-bottom filter group-hover:contrast-110 transition-transform duration-300 group-hover:scale-105"
                          sizes="(max-width: 768px) 100px, 180px"
                        />
                      )}
                      <div className="absolute top-2 left-2 font-mono text-[8px] sm:text-[9px] text-[#00f0ff] font-bold">
                        {member.num}
                      </div>
                    </div>
                    <div className="mt-2 text-center">
                      <div className="text-[10px] sm:text-xs font-bold text-white uppercase tracking-tight group-hover:text-cyan-300 transition-colors">
                        {member.name.split(" ")[0]}
                      </div>
                      <div className="font-mono text-[8px] text-[#00f0ff] line-clamp-1">
                        {member.role.split("&")[0]}
                      </div>
                    </div>
                  </div>
                ))}
              </div>

              {/* Bottom Mission Statement */}
              <div className="relative z-10 flex flex-wrap items-center justify-center gap-6 font-mono text-[10px] sm:text-xs text-slate-400">
                <span className="flex items-center gap-1.5 text-cyan-300">
                  <span className="w-1.5 h-1.5 rounded-full bg-[#00f0ff]" />
                  AUTONOMOUS REASONING
                </span>
                <span>•</span>
                <span className="flex items-center gap-1.5 text-cyan-300">
                  <span className="w-1.5 h-1.5 rounded-full bg-[#00f0ff]" />
                  ORBITAL PRECISION
                </span>
                <span>•</span>
                <span className="flex items-center gap-1.5 text-cyan-300">
                  <span className="w-1.5 h-1.5 rounded-full bg-[#00f0ff]" />
                  HUMAN RIGOR
                </span>
              </div>
            </div>
          )}
        </main>

        {/* ------------------------------------------------------- */}
        {/* BOTTOM CONTROLS: Progress Bar + Secondary Arrow Buttons */}
        {/* Exactly matching Image 2                                */}
        {/* ------------------------------------------------------- */}
        <footer className="relative z-30 w-full pb-6 sm:pb-8 px-6 sm:px-12 max-w-[1540px] mx-auto flex items-center justify-between pointer-events-auto">
          {/* Left: Scroll Instruction & Active Dossier Progress (Image 2) */}
          <div className="flex items-center gap-4">
            <div className="font-mono text-xs text-slate-400 tracking-widest hidden sm:flex items-center gap-2">
              <span className="text-[#00f0ff] font-bold">STAGE:</span>
              <span className="text-white font-semibold">
                {activeStep === 0
                  ? "00 // MISSION INTRO"
                  : activeStep === 7
                  ? "07 // COHORT ENSEMBLE"
                  : `0${activeStep} // ${TEAM_MEMBERS[activeStep - 1]?.name || ""}`}
              </span>
            </div>

            {/* Continuous Micro Progress Bar */}
            <div className="w-28 sm:w-36 h-1.5 bg-slate-800 rounded-full overflow-hidden flex">
              <div
                className="h-full bg-[#00f0ff] shadow-[0_0_10px_#00f0ff] transition-all duration-150"
                style={{ width: `${Math.round(scrollProgress * 100)}%` }}
              />
            </div>
          </div>

          {/* Right: Secondary Circular Cyan Navigation Buttons (Image 2) */}
          <div className="flex items-center gap-3">
            <button
              onClick={prevMember}
              disabled={activeStep === 0}
              aria-label="Previous Team Member"
              className={`w-9 h-9 rounded-full flex items-center justify-center border transition-all duration-300 ${
                activeStep === 0
                  ? "border-slate-800 text-slate-600 cursor-not-allowed"
                  : "border-cyan-500/40 bg-cyan-950/30 text-cyan-300 hover:text-white hover:border-[#00f0ff] hover:bg-[#00f0ff]/15 hover:shadow-[0_0_20px_rgba(0,240,255,0.4)] active:scale-95 cursor-pointer"
              }`}
            >
              <ArrowLeft className="w-4 h-4" />
            </button>

            <button
              onClick={nextMember}
              disabled={activeStep === 7}
              aria-label="Next Team Member"
              className={`w-9 h-9 rounded-full flex items-center justify-center border transition-all duration-300 ${
                activeStep === 7
                  ? "border-slate-800 text-slate-600 cursor-not-allowed"
                  : "border-cyan-500/40 bg-cyan-950/30 text-[#00f0ff] hover:text-white hover:border-[#00f0ff] hover:bg-[#00f0ff]/20 hover:shadow-[0_0_24px_rgba(0,240,255,0.7)] active:scale-95 cursor-pointer"
              }`}
            >
              <ArrowRight className="w-4 h-4" />
            </button>
          </div>
        </footer>
      </div>
    </section>
  );
}
