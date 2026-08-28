"use client";

import { usePathname } from "next/navigation";
import { useMemo } from "react";

export default function OrbitalEnergyBackground() {
  const pathname = usePathname();

  // Page-specific background intensity
  const intensity = useMemo(() => {
    if (!pathname || pathname === "/") return 0;
    if (pathname.startsWith("/dashboard")) return 0.42;
    if (pathname.startsWith("/query")) return 0.28;
    if (pathname.startsWith("/comparison")) return 0.32;
    if (pathname.startsWith("/upload")) return 0.22;
    if (pathname.startsWith("/reports")) return 0.14;
    if (pathname.startsWith("/results")) return 0.36;
    if (pathname.startsWith("/execution")) return 0.35;
    if (pathname.startsWith("/evidence")) return 0.30;
    if (pathname.startsWith("/settings")) return 0.12;
    return 0.30;
  }, [pathname]);

  return (
    <div
      aria-hidden="true"
      className="pointer-events-none fixed inset-0 -z-10 overflow-hidden bg-[#06111d] select-none transition-opacity duration-700 ease-in-out"
      style={{ opacity: intensity }}
    >
      {/* ----------------------------------------------------
          LAYER 1: Atmospheric Ambient Blue Glows
          ---------------------------------------------------- */}
      <div className="absolute -top-[12%] -left-[8%] w-[60vw] h-[60vw] max-w-[800px] max-h-[800px] rounded-full bg-gradient-to-br from-[#008cff]/[0.10] via-[#00d4ff]/[0.06] to-transparent blur-[120px] animate-ambient-drift" />
      <div
        className="absolute top-[35%] -right-[12%] w-[55vw] h-[55vw] max-w-[700px] max-h-[700px] rounded-full bg-gradient-to-bl from-[#1557a8]/[0.11] via-[#008cff]/[0.07] to-transparent blur-[130px] animate-ambient-drift"
        style={{ animationDelay: "-8s" }}
      />
      <div
        className="absolute -bottom-[15%] left-[25%] w-[50vw] h-[50vw] max-w-[650px] max-h-[650px] rounded-full bg-gradient-to-t from-[#00d4ff]/[0.07] via-[#1557a8]/[0.06] to-transparent blur-[140px] animate-ambient-drift"
        style={{ animationDelay: "-12s" }}
      />

      {/* Subtle Micro Coordinate Grid */}
      <div
        className="absolute inset-0 opacity-[0.035]"
        style={{
          backgroundImage:
            "linear-gradient(rgba(0, 212, 255, 0.4) 1px, transparent 1px), linear-gradient(90deg, rgba(0, 212, 255, 0.4) 1px, transparent 1px)",
          backgroundSize: "64px 64px",
        }}
      />

      {/* ----------------------------------------------------
          LAYER 2: Curved Electromagnetic Paths & Energy Trails
          ---------------------------------------------------- */}
      <svg
        className="absolute inset-0 w-full h-full"
        preserveAspectRatio="none"
        viewBox="0 0 1600 1000"
      >
        <defs>
          <linearGradient id="em-cyan-blue" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#00d4ff" stopOpacity="0.8" />
            <stop offset="40%" stopColor="#008cff" stopOpacity="0.6" />
            <stop offset="100%" stopColor="#1557a8" stopOpacity="0.1" />
          </linearGradient>

          <linearGradient id="em-blue-deep" x1="100%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="#008cff" stopOpacity="0.75" />
            <stop offset="60%" stopColor="#00d4ff" stopOpacity="0.4" />
            <stop offset="100%" stopColor="#06111d" stopOpacity="0" />
          </linearGradient>

          <linearGradient id="em-transverse" x1="0%" y1="50%" x2="100%" y2="50%">
            <stop offset="0%" stopColor="#1557a8" stopOpacity="0" />
            <stop offset="25%" stopColor="#00d4ff" stopOpacity="0.6" />
            <stop offset="75%" stopColor="#008cff" stopOpacity="0.7" />
            <stop offset="100%" stopColor="#1557a8" stopOpacity="0" />
          </linearGradient>

          <linearGradient id="em-harmonic" x1="0%" y1="100%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#008cff" stopOpacity="0.2" />
            <stop offset="50%" stopColor="#00d4ff" stopOpacity="0.8" />
            <stop offset="100%" stopColor="#008cff" stopOpacity="0.1" />
          </linearGradient>

          <filter id="em-path-glow" x="-20%" y="-20%" width="140%" height="140%">
            <feGaussianBlur stdDeviation="4" result="blur" />
            <feComposite in="SourceGraphic" in2="blur" operator="over" />
          </filter>
        </defs>

        {/* Path 1: Primary High-Altitude Orbital Arc */}
        <path
          d="M -100,160 C 350,190 750,50 1150,300 C 1450,490 1600,820 1750,880"
          fill="none"
          stroke="url(#em-cyan-blue)"
          strokeWidth="1.4"
          strokeDasharray="260 550"
          filter="url(#em-path-glow)"
          opacity="0.18"
          className="animate-orbital-1"
        />

        {/* Path 2: Resonant Harmonic Companion Wave */}
        <path
          d="M -100,190 C 350,220 750,80 1150,330 C 1450,520 1600,850 1750,910"
          fill="none"
          stroke="#008cff"
          strokeWidth="0.9"
          strokeDasharray="180 650"
          opacity="0.13"
          className="animate-orbital-2"
        />

        {/* Path 3: Ascending Electromagnetic Carrier Wave */}
        <path
          d="M -80,860 C 280,760 640,860 980,590 C 1280,350 1480,180 1720,90"
          fill="none"
          stroke="url(#em-blue-deep)"
          strokeWidth="1.3"
          strokeDasharray="220 520"
          filter="url(#em-path-glow)"
          opacity="0.16"
          className="animate-orbital-4"
        />

        {/* Path 4: Secondary Trans-Horizon Wave */}
        <path
          d="M -80,890 C 280,790 640,890 980,620 C 1280,380 1480,210 1720,120"
          fill="none"
          stroke="#00d4ff"
          strokeWidth="0.8"
          strokeDasharray="160 680"
          opacity="0.11"
          className="animate-orbital-3"
        />

        {/* Path 5: Transverse Mid-Latitude Telemetry Stream */}
        <path
          d="M -100,500 Q 420,430 820,530 T 1750,480"
          fill="none"
          stroke="url(#em-transverse)"
          strokeWidth="1.2"
          strokeDasharray="190 480"
          opacity="0.15"
          className="animate-orbital-5"
        />

        {/* Path 6: Low-Orbit Polar Sweep Arc */}
        <path
          d="M -50,320 C 400,240 850,480 1250,220 C 1450,110 1620,150 1720,200"
          fill="none"
          stroke="url(#em-harmonic)"
          strokeWidth="1.0"
          strokeDasharray="150 560"
          opacity="0.13"
          className="animate-orbital-2"
          style={{ animationDuration: "18s" }}
        />

        {/* Path 7: Equatorial Deep-Space Boundary Trail */}
        <path
          d="M -100,720 C 300,640 700,780 1100,680 C 1350,620 1550,740 1750,700"
          fill="none"
          stroke="#1557a8"
          strokeWidth="0.9"
          strokeDasharray="140 600"
          opacity="0.14"
          className="animate-orbital-3"
          style={{ animationDuration: "25s" }}
        />
      </svg>

      {/* ----------------------------------------------------
          LAYER 3: Small Blue Energy Telemetry Nodes
          ---------------------------------------------------- */}
      <div className="absolute top-[26%] left-[42%] w-2 h-2 rounded-full bg-[#00d4ff] shadow-[0_0_12px_rgba(0,212,255,0.8),0_0_24px_rgba(0,140,255,0.5)] animate-pulse-energy-node" />
      <div
        className="absolute top-[52%] left-[68%] w-1.5 h-1.5 rounded-full bg-[#008cff] shadow-[0_0_10px_rgba(0,140,255,0.8),0_0_20px_rgba(0,212,255,0.4)] animate-pulse-energy-node"
        style={{ animationDelay: "-3.5s" }}
      />
      <div
        className="absolute top-[74%] left-[28%] w-2 h-2 rounded-full bg-[#00d4ff] shadow-[0_0_14px_rgba(0,212,255,0.85)] animate-pulse-energy-node"
        style={{ animationDelay: "-5.2s" }}
      />
      <div
        className="absolute top-[38%] left-[84%] w-1.5 h-1.5 rounded-full bg-[#008cff] shadow-[0_0_10px_rgba(0,140,255,0.7)] animate-pulse-energy-node"
        style={{ animationDelay: "-2.1s" }}
      />
    </div>
  );
}
