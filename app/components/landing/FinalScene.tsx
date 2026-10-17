"use client";

import React, { useRef, useEffect, useState } from "react";
import Link from "next/link";
import {
  ArrowRight,
  Terminal,
  FileText,
  Activity,
  ShieldCheck,
  Globe2,
  Sparkles,
  ExternalLink,
} from "lucide-react";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import NexSpaceLogo from "./NexSpaceLogo";

export default function FinalScene() {
  const containerRef = useRef<HTMLDivElement>(null);
  const [satelliteAngle, setSatelliteAngle] = useState(0);

  useEffect(() => {
    gsap.registerPlugin(ScrollTrigger);

    const el = containerRef.current;
    if (!el) return;

    const isReduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (isReduced) return;

    const st = ScrollTrigger.create({
      trigger: el,
      start: "top 85%",
      end: "bottom bottom",
      scrub: 0.6,
      onUpdate: (self) => {
        setSatelliteAngle(self.progress * 180);
      },
    });

    return () => {
      st.kill();
    };
  }, []);

  return (
    <section
      ref={containerRef}
      id="brand"
      data-spatial="true"
      className="relative min-h-screen w-full bg-[#01040a] text-slate-100 py-24 sm:py-32 px-6 sm:px-10 lg:px-14 flex flex-col justify-between overflow-hidden border-t border-cyan-500/20 select-none"
    >
      {/* Planetary Horizon Glow Curve & Deep Space Nebula */}
      <div className="absolute inset-0 pointer-events-none overflow-hidden">
        {/* Deep blue/cyan radial glow simulating Earth's illuminated horizon */}
        <div className="absolute bottom-[-40%] left-1/2 -translate-x-1/2 w-[160%] sm:w-[130%] h-[750px] rounded-[100%] bg-gradient-to-t from-cyan-600/20 via-sky-500/10 to-transparent blur-3xl pointer-events-none" />
        <div className="absolute bottom-[-30%] left-1/2 -translate-x-1/2 w-[140%] sm:w-[110%] h-[400px] rounded-[100%] border-t border-cyan-400/40 shadow-[0_-20px_80px_rgba(6,182,212,0.35)] pointer-events-none" />
        
        {/* Ambient starfield grid points */}
        <div 
          className="absolute inset-0 opacity-15"
          style={{
            backgroundImage: "radial-gradient(#38bdf8 1px, transparent 1px)",
            backgroundSize: "40px 40px",
          }}
        />
      </div>

      {/* Orbiting Satellite Track Representation */}
      <div className="relative max-w-6xl mx-auto w-full my-auto flex flex-col items-center text-center z-10">
        {/* Subtle Orbit Badge */}
        <div className="inline-flex items-center gap-2.5 px-4 py-1.5 rounded-full border border-cyan-500/30 bg-cyan-950/40 backdrop-blur-md text-cyan-300 text-xs font-mono tracking-widest uppercase mb-8 shadow-[0_0_20px_rgba(6,182,212,0.2)]">
          <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
          <span>08 // CONVERGENCE & VERIFICATION</span>
        </div>

        {/* Brand Climax Logo & Title */}
        <div className="mb-6 transform transition-transform duration-500 hover:scale-[1.02]">
          <NexSpaceLogo size="xl" showText={true} />
        </div>

        <h2 className="text-3xl sm:text-5xl lg:text-6xl font-black tracking-tight text-white max-w-4xl leading-tight">
          Translating Orbital Sight into{" "}
          <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-300 via-sky-400 to-teal-300">
            Ground-Truth Decisions
          </span>
        </h2>

        <p className="mt-6 text-base sm:text-lg text-slate-300 max-w-2xl font-sans leading-relaxed">
          From multi-spectral optical and SAR constellations to calibrated geospatial bounding coordinates.
          Experience the autonomous remote-sensing intelligence engine in real time.
        </p>

        {/* Capability Metrics Row */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 sm:gap-6 w-full max-w-3xl mt-12 mb-12">
          <div className="p-4 rounded-xl border border-cyan-500/20 bg-[#06111f]/60 backdrop-blur-md text-left">
            <div className="flex items-center gap-2 text-cyan-400 text-xs font-mono mb-1">
              <ShieldCheck className="w-3.5 h-3.5" />
              <span>CONFIDENCE</span>
            </div>
            <div className="text-2xl font-bold font-mono text-white">99.4%</div>
            <div className="text-[11px] text-slate-400 font-sans">Deterministic CRS</div>
          </div>

          <div className="p-4 rounded-xl border border-cyan-500/20 bg-[#06111f]/60 backdrop-blur-md text-left">
            <div className="flex items-center gap-2 text-cyan-400 text-xs font-mono mb-1">
              <Activity className="w-3.5 h-3.5" />
              <span>LATENCY</span>
            </div>
            <div className="text-2xl font-bold font-mono text-white">&lt;850ms</div>
            <div className="text-[11px] text-slate-400 font-sans">End-to-end pipeline</div>
          </div>

          <div className="p-4 rounded-xl border border-cyan-500/20 bg-[#06111f]/60 backdrop-blur-md text-left">
            <div className="flex items-center gap-2 text-cyan-400 text-xs font-mono mb-1">
              <Globe2 className="w-3.5 h-3.5" />
              <span>RESOLUTION</span>
            </div>
            <div className="text-2xl font-bold font-mono text-white">10m / px</div>
            <div className="text-[11px] text-slate-400 font-sans">Sentinel-2 & Landsat</div>
          </div>

          <div className="p-4 rounded-xl border border-cyan-500/20 bg-[#06111f]/60 backdrop-blur-md text-left">
            <div className="flex items-center gap-2 text-cyan-400 text-xs font-mono mb-1">
              <Sparkles className="w-3.5 h-3.5" />
              <span>PIPELINE</span>
            </div>
            <div className="text-2xl font-bold font-mono text-white">12-Stage</div>
            <div className="text-[11px] text-slate-400 font-sans">Audited forensic telemetry</div>
          </div>
        </div>

        {/* Magnetic High-Impact Action CTAs */}
        <div className="flex flex-col sm:flex-row items-center justify-center gap-4 sm:gap-6 w-full max-w-md">
          <Link
            href="/query"
            className="group relative w-full sm:w-auto inline-flex items-center justify-center gap-3 px-8 py-4 rounded-xl bg-gradient-to-r from-cyan-500 to-sky-500 hover:from-cyan-400 hover:to-sky-400 text-black font-bold font-mono text-sm tracking-wider uppercase transition-all duration-300 shadow-[0_0_30px_rgba(6,182,212,0.4)] hover:shadow-[0_0_50px_rgba(6,182,212,0.7)] hover:-translate-y-0.5 active:translate-y-0"
          >
            <Terminal className="w-4 h-4 text-black group-hover:rotate-12 transition-transform" />
            <span>Launch NLP Terminal</span>
            <ArrowRight className="w-4 h-4 text-black group-hover:translate-x-1 transition-transform" />
          </Link>

          <a
            href="/NexSpace_Backend_And_Agent_Controller_PRD_Audit.pdf"
            target="_blank"
            rel="noopener noreferrer"
            className="w-full sm:w-auto inline-flex items-center justify-center gap-2.5 px-6 py-4 rounded-xl border border-cyan-500/30 hover:border-cyan-400 bg-slate-900/60 hover:bg-cyan-950/30 text-cyan-300 font-mono text-xs tracking-wider uppercase transition-all duration-200"
          >
            <FileText className="w-4 h-4 text-cyan-400" />
            <span>PRD Technical Audit</span>
            <ExternalLink className="w-3 h-3 text-cyan-400/70" />
          </a>
        </div>
      </div>

      {/* Modern Aerospace Terminal Footer */}
      <footer className="relative max-w-6xl mx-auto w-full pt-16 border-t border-cyan-900/30 flex flex-col md:flex-row items-center justify-between gap-6 text-xs text-slate-400 font-mono z-10">
        <div className="flex items-center gap-3">
          <div className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-ping" />
          <span className="text-slate-200 font-bold">CORE ORBITAL ENGINE: READY</span>
          <span className="text-slate-600">|</span>
          <span className="text-slate-400">FASTAPI :8000 &bull; NEXT.JS :3000</span>
        </div>

        <div className="flex items-center gap-6">
          <Link href="/query" className="hover:text-cyan-300 transition-colors">
            Query Terminal
          </Link>
          <a
            href="/NexSpace_Tech_Stack_Frameworks_Models.pdf"
            target="_blank"
            rel="noopener noreferrer"
            className="hover:text-cyan-300 transition-colors"
          >
            Architecture Spec
          </a>
          <span className="text-slate-600">&copy; 2026 NEXSPACE DEFENSE & SENSING</span>
        </div>
      </footer>
    </section>
  );
}
