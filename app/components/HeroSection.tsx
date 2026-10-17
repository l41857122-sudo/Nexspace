"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import { Play, ArrowRight, Sparkles, Crosshair, Radio } from "lucide-react";
import WebThreads from "./WebThreads";
import TopNavigationBar from "./TopNavigationBar";

export default function HeroSection() {
  return (
    <section className="relative w-full min-h-screen bg-[#06111d] overflow-hidden flex flex-col justify-between select-none">
      {/* ----------------------------------------------------
          FULL-SCREEN BACKGROUND LAYER: WebThreads
          ---------------------------------------------------- */}
      <div className="absolute inset-0 z-0">
        <WebThreads
          color1="#5227FF"
          color2="#1339eb"
          color3="#06B6D4"
          speed={0.16}
          threadCount={5}
          frequency={3.2}
          spread={0.14}
          taper={1}
          position={0.5}
          fanMode="center"
          glow={0.032}
          falloff={0.58}
          thickness={0.95}
          brightness={0.65}
          opacity={0.72}
          mirror
          shimmer={false}
          grain
          grainIntensity={0.04}
          mouseInteraction
          mouseStrength={0.18}
        />
      </div>

      {/* Subtle Dark Overlay to keep text legible without dimming threads */}
      <div className="absolute inset-0 z-0 bg-[radial-gradient(ellipse_at_center,_rgba(6,17,29,0.3)_0%,_rgba(6,17,29,0.55)_60%,_rgba(6,17,29,0.85)_100%)] pointer-events-none" />

      {/* ----------------------------------------------------
          Subtle Mission Control HUD Corner Reticles & Telemetry
          ---------------------------------------------------- */}
      {/* Top Left HUD Reticle */}
      <div className="hidden lg:flex absolute top-28 left-8 z-10 items-center gap-2 font-mono text-[10px] text-cyan-400/70 pointer-events-none">
        <Crosshair size={12} className="text-cyan-400/50" />
        <span className="tracking-widest uppercase">SAT-PASS: SENTINEL-2B · ORBIT 412</span>
      </div>

      {/* Top Right HUD Reticle */}
      <div className="hidden lg:flex absolute top-28 right-8 z-10 items-center gap-2 font-mono text-[10px] text-cyan-400/70 pointer-events-none">
        <span className="w-1.5 h-1.5 rounded-full bg-cyan-400/70 animate-pulse" />
        <span className="tracking-widest uppercase">RES: 0.5M GSD · SENSOR: SAR-X</span>
      </div>

      {/* Bottom Left HUD Reticle */}
      <div className="hidden lg:flex absolute bottom-8 left-8 z-10 items-center gap-2 font-mono text-[10px] text-slate-400 pointer-events-none">
        <Radio size={11} className="text-cyan-400/60" />
        <span className="tracking-wider">GRID: EPSG:4326 · ELEV: 142M · NORTH-UP</span>
      </div>

      {/* ----------------------------------------------------
          Global Top Navigation
          ---------------------------------------------------- */}
      <TopNavigationBar />

      {/* ----------------------------------------------------
          Central Hero Content (NexSpace Branding)
          ---------------------------------------------------- */}
      <div className="relative z-20 flex flex-col items-center text-center px-4 sm:px-6 pt-32 sm:pt-40 pb-20 max-w-4xl mx-auto my-auto">
        {/* System Status Pill Badge */}
        <motion.div
          initial={{ opacity: 0, y: -12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.45, ease: "easeOut" }}
          className="flex items-center gap-2.5 mb-6 text-xs text-emerald-300 font-mono bg-[#0c1c2c]/85 backdrop-blur-md border border-emerald-500/25 px-4 py-1.5 rounded-full shadow-[0_0_15px_rgba(16,185,129,0.15)]"
        >
          <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse shadow-[0_0_8px_rgba(52,211,153,0.9)]" />
          <span className="tracking-wide">System Online · Orbital Intelligence Active</span>
        </motion.div>

        {/* Hero Headline: NexSpace. Redefined by AI. */}
        <motion.h1
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.1, ease: "easeOut" }}
          className="text-4xl sm:text-6xl lg:text-7xl font-bold leading-[1.1] tracking-tight text-white drop-shadow-[0_2px_12px_rgba(0,0,0,0.8)]"
        >
          <span>NexSpace.</span>
          <br />
          <span className="bg-gradient-to-r from-cyan-300 via-sky-300 to-emerald-300 bg-clip-text text-transparent drop-shadow-[0_0_35px_rgba(6,182,212,0.45)]">
            Redefined by AI.
          </span>
        </motion.h1>

        {/* Subtitle / Description */}
        <motion.p
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2, ease: "easeOut" }}
          className="mt-6 text-sm sm:text-base lg:text-lg text-slate-300/90 max-w-2xl leading-relaxed font-sans"
        >
          Transform planetary-scale satellite imagery into actionable
          geospatial intelligence with autonomous neural telemetry. Deploy
          high-precision multispectral models in milliseconds.
        </motion.p>

        {/* Interactive CTA Buttons */}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.3, ease: "easeOut" }}
          className="mt-8 flex flex-wrap items-center justify-center gap-4"
        >
          {/* Primary Button */}
          <Link
            href="/query"
            className="group relative inline-flex items-center gap-2 bg-gradient-to-r from-cyan-500 via-sky-400 to-cyan-400 hover:from-cyan-400 hover:to-sky-300 text-[#071320] text-sm font-bold px-7 py-3.5 rounded-xl transition-all duration-180 shadow-[0_0_25px_rgba(6,182,212,0.45)] hover:shadow-[0_0_35px_rgba(6,182,212,0.7)] hover:scale-[1.02] active:scale-[0.98] cursor-pointer overflow-hidden"
          >
            <Sparkles size={16} className="text-[#071320]" />
            <span>Launch Investigation</span>
            <ArrowRight
              size={15}
              className="group-hover:translate-x-1 transition-transform duration-180"
            />
          </Link>

          {/* Secondary Button */}
          <Link
            href="/dashboard"
            className="group inline-flex items-center gap-2.5 bg-[#0b1726]/80 hover:bg-slate-800/80 backdrop-blur-md border border-slate-800 hover:border-cyan-500/40 text-slate-200 hover:text-cyan-300 text-sm font-semibold px-6 py-3.5 rounded-xl transition-all duration-180 hover:scale-[1.02] active:scale-[0.98] shadow-sm hover:shadow-[0_0_15px_rgba(6,182,212,0.2)] cursor-pointer"
          >
            <div className="w-5 h-5 rounded-full bg-cyan-500/15 border border-cyan-500/30 flex items-center justify-center text-cyan-400 group-hover:scale-110 transition-transform duration-180">
              <Play size={10} className="fill-cyan-400/60 translate-x-[1px]" />
            </div>
            <span>Mission Control</span>
          </Link>
        </motion.div>

        {/* ----------------------------------------------------
            Cinematic Intelligence Output Journey Flow
            SATELLITE IMAGERY → AI ANALYSIS → TARGETS FOUND → EVIDENCE → LOCATION → INTELLIGENCE
            ---------------------------------------------------- */}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.4, ease: "easeOut" }}
          className="mt-12 w-full max-w-3xl"
        >
          {/* Journey Steps Bar */}
          <div className="flex items-center justify-between gap-1 p-2 rounded-xl bg-slate-950/70 border border-slate-800/80 backdrop-blur-md font-mono text-[10px] text-slate-400 overflow-x-auto">
            <span className="text-cyan-300 font-semibold px-2 py-1 rounded bg-cyan-500/10">SATELLITE IMAGE</span>
            <span className="text-slate-600">→</span>
            <span className="text-sky-300 font-semibold px-2 py-1 rounded bg-sky-500/10">AI ANALYSIS</span>
            <span className="text-slate-600">→</span>
            <span className="text-emerald-300 font-bold px-2 py-1 rounded bg-emerald-500/15 border border-emerald-500/30">TARGETS FOUND</span>
            <span className="text-slate-600">→</span>
            <span className="text-cyan-300 font-semibold px-2 py-1 rounded bg-cyan-500/10">EVIDENCE</span>
            <span className="text-slate-600">→</span>
            <span className="text-purple-300 font-semibold px-2 py-1 rounded bg-purple-500/10">LOCATION</span>
            <span className="text-slate-600">→</span>
            <span className="text-amber-300 font-bold px-2 py-1 rounded bg-amber-500/10">INTELLIGENCE</span>
          </div>

          {/* Live Interactive Intelligence Showcase Pill */}
          <div className="mt-4 p-4 rounded-2xl bg-gradient-to-br from-slate-950/90 via-[#071322]/90 to-slate-950/90 border border-cyan-500/30 shadow-[0_0_30px_rgba(6,182,212,0.15)] text-left flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
            <div className="space-y-1">
              <div className="flex items-center gap-2">
                <span className="inline-block w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                <span className="text-[11px] font-mono font-bold text-emerald-300 uppercase tracking-wider">
                  Live Investigation Output
                </span>
                <span className="text-[10px] font-mono text-slate-400 bg-slate-900 px-2 py-0.5 rounded border border-slate-800">
                  Mumbai Harbour · Sentinel-2
                </span>
              </div>
              <p className="text-sm font-semibold text-white">
                "6 vessels identified near the port docks."
              </p>
              <div className="text-xs text-slate-400 font-mono flex items-center gap-3 pt-0.5">
                <span>Targets: <strong className="text-cyan-300">6 Verified</strong></span>
                <span>•</span>
                <span>Focal: <strong className="text-slate-200">18.9402° N, 72.8354° E</strong></span>
                <span>•</span>
                <span>Confidence: <strong className="text-emerald-400">High (91%)</strong></span>
              </div>
            </div>

            <Link
              href="/query"
              className="shrink-0 inline-flex items-center gap-1.5 px-4 py-2 rounded-lg bg-cyan-500/15 border border-cyan-500/40 text-cyan-300 hover:bg-cyan-500/25 hover:text-white text-xs font-mono font-bold transition-all cursor-pointer"
            >
              <span>Explore Evidence</span>
              <ArrowRight size={13} />
            </Link>
          </div>
        </motion.div>
      </div>

      {/* ----------------------------------------------------
          Bottom Telemetry Readout
          ---------------------------------------------------- */}
      <div className="relative sm:absolute sm:bottom-6 sm:right-8 z-20 text-center sm:text-right p-4 text-[10px] font-mono text-slate-400">
        <div className="uppercase tracking-widest text-slate-500 font-mono">
          Planetary Coordinate Resolution
        </div>
        <div className="text-cyan-300/90 font-mono mt-0.5">
          EPSG:4326 · 0.5M GSD · MULTISPECTRAL OPTICAL
        </div>
      </div>
    </section>
  );
}