"use client";

import React from "react";

interface SatelliteImageryRevealProps {
  progress: number; // 0 to 1 master timeline progress
  opacity: number; // from master timeline (imageryAlpha)
  scale?: number;
  gridAlpha?: number;
  scanBeamProgress?: number;
  analysisReadyAlpha?: number;
  isReducedMotion?: boolean;
}

export default function SatelliteImageryReveal({
  progress,
  opacity,
  scale = 1.0,
  gridAlpha = 0,
  scanBeamProgress = 0,
  analysisReadyAlpha = 0,
  isReducedMotion = false,
}: SatelliteImageryRevealProps) {
  // Only mount when opacity is above threshold to preserve WebGL performance
  if (opacity <= 0.01) return null;

  // Deliberate cinematic scale settling (1.06 -> 1.00), preventing any zoomed-crop distortion
  const effectiveScale = isReducedMotion ? 1.0 : scale;

  // Cinematic Focus Sharpening: Imagery enters slightly blurred and focuses crisply as camera settles
  const focusBlur = isReducedMotion ? 0 : Math.max(0, (0.76 - progress) * 35);

  // Atmospheric Cloud/Surface Entry Motion Streak (Visible during sub-orbital dive, then dissolves)
  const diveStreakOpacity = isReducedMotion
    ? 0
    : Math.max(0, Math.min(0.65, (0.74 - progress) / 0.16));

  // Self-drawing AOI boundary progress: draws border as camera locks on surface
  const aoiDrawOffset = Math.max(0, Math.min(1, (0.84 - progress) / 0.18)) * 3200;
  const aoiOpacity = Math.max(0, Math.min(0.85, (progress - 0.62) / 0.14));

  return (
    <div
      className="absolute inset-0 w-full h-full z-15 pointer-events-none overflow-hidden transition-opacity duration-150 select-none"
      style={{ opacity }}
      aria-hidden="true"
    >
      {/* Full-Viewport High-Resolution Real Satellite Surface Imagery (Coastal / Land Interface) */}
      <div
        className="absolute inset-0 w-full h-full transition-transform duration-200"
        style={{ transform: `scale(${effectiveScale})` }}
      >
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src="/imagery/satellite_coastline_water.jpg"
          alt="High-Resolution Real Earth Observation Coastal Surface"
          className="w-full h-full object-cover object-center transition-all duration-300"
          style={{
            filter: `blur(${focusBlur}px) brightness(0.92) contrast(1.08)`,
          }}
        />

        {/* Cinematic Atmospheric Dive Streak Effect (Clouds & Surface rapid entry) */}
        {diveStreakOpacity > 0.02 && (
          <div
            className="absolute inset-0 pointer-events-none mix-blend-screen transition-opacity duration-200"
            style={{
              opacity: diveStreakOpacity,
              background:
                "radial-gradient(circle at center, transparent 30%, rgba(200, 240, 255, 0.25) 85%, rgba(0, 212, 255, 0.4) 100%), linear-gradient(180deg, rgba(255, 255, 255, 0.15) 0%, transparent 60%)",
            }}
          />
        )}

        {/* Self-Drawing AOI Boundary Box (Traces itself as imagery stabilizes) */}
        <svg
          className="absolute inset-8 sm:inset-14 w-[calc(100%-4rem)] sm:w-[calc(100%-7rem)] h-[calc(100%-4rem)] sm:h-[calc(100%-7rem)] pointer-events-none"
          viewBox="0 0 1000 600"
          preserveAspectRatio="none"
        >
          <rect
            x="2"
            y="2"
            width="996"
            height="596"
            fill="rgba(0, 240, 255, 0.02)"
            stroke="#00f0ff"
            strokeWidth="1.5"
            strokeDasharray="3200"
            strokeDashoffset={aoiDrawOffset}
            strokeOpacity={aoiOpacity}
          />
        </svg>

        {/* Precision Corner Framing Brackets */}
        <div className="absolute inset-8 sm:inset-14 pointer-events-none">
          <div className="absolute top-0 left-0 w-6 h-6 border-t-2 border-l-2 border-[#00f0ff] shadow-[0_0_10px_#00f0ff]" />
          <div className="absolute top-0 right-0 w-6 h-6 border-t-2 border-r-2 border-[#00f0ff] shadow-[0_0_10px_#00f0ff]" />
          <div className="absolute bottom-0 left-0 w-6 h-6 border-b-2 border-l-2 border-[#00f0ff] shadow-[0_0_10px_#00f0ff]" />
          <div className="absolute bottom-0 right-0 w-6 h-6 border-b-2 border-r-2 border-[#00f0ff] shadow-[0_0_10px_#00f0ff]" />
        </div>

        {/* Cinematic Radial Vignette */}
        <div className="absolute inset-0 pointer-events-none bg-[radial-gradient(ellipse_at_center,_transparent_45%,_rgba(1,4,10,0.85)_100%)]" />

        {/* Top Edge Gradient Blend */}
        <div className="absolute top-0 left-0 right-0 h-32 bg-gradient-to-b from-[#01040a] via-[#01040a]/40 to-transparent pointer-events-none" />

        {/* Bottom Edge Gradient Blend (Prevents any giant black gap underneath) */}
        <div className="absolute bottom-0 left-0 right-0 h-44 bg-gradient-to-t from-[#01040a] via-[#01040a]/50 to-transparent pointer-events-none" />

        {/* Global Optical Coordinate Grid Overlay (Fades in gradually) */}
        <div
          className="absolute inset-0 pointer-events-none transition-opacity duration-300"
          style={{
            opacity: Math.max(0, gridAlpha) * 0.22,
            backgroundImage:
              "linear-gradient(rgba(0, 240, 255, 0.4) 1px, transparent 1px), linear-gradient(90deg, rgba(0, 240, 255, 0.4) 1px, transparent 1px)",
            backgroundSize: "80px 80px",
          }}
        />

        {/* Subtle Sweeping Cyan Scan Beam */}
        {scanBeamProgress > 0 && scanBeamProgress < 1 && (
          <div
            className="absolute top-0 bottom-0 pointer-events-none z-20 transition-all duration-75"
            style={{
              left: `${scanBeamProgress * 100}%`,
              width: "120px",
              transform: "translateX(-50%)",
              background:
                "linear-gradient(90deg, transparent 0%, rgba(0, 240, 255, 0.04) 50%, rgba(0, 240, 255, 0.32) 96%, #ffffff 100%)",
              boxShadow: "0 0 20px rgba(0, 240, 255, 0.3)",
            }}
          >
            {/* Vertical Leading Edge Laser Line */}
            <div className="absolute right-0 top-0 bottom-0 w-[1.5px] bg-[#00f0ff] shadow-[0_0_12px_#00f0ff]" />
          </div>
        )}

        {/* Center Optical Crosshair */}
        <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 w-20 h-20 pointer-events-none opacity-60">
          <div className="w-full h-full border border-cyan-400/40 rounded-full flex items-center justify-center">
            <div className="w-1.5 h-1.5 bg-cyan-400 rounded-full" />
          </div>
          <div className="absolute -top-3 left-1/2 -translate-x-1/2 w-px h-3 bg-cyan-400/70" />
          <div className="absolute -bottom-3 left-1/2 -translate-x-1/2 w-px h-3 bg-cyan-400/70" />
          <div className="absolute top-1/2 -left-3 -translate-y-1/2 h-px w-3 bg-cyan-400/70" />
          <div className="absolute top-1/2 -right-3 -translate-y-1/2 h-px w-3 bg-cyan-400/70" />
        </div>

        {/* Telemetry Corner Reticles & Geodetic Data (Clean & Restrained) */}
        <div className="absolute top-20 left-8 text-[#00f0ff] font-mono text-[10px] tracking-widest flex items-center gap-2">
          <span className="w-2 h-2 border-t border-l border-[#00f0ff]" />
          <span>AOI: 18.9499° N, 72.9515° E // ORBITAL DESCENT</span>
        </div>
        <div className="absolute top-20 right-8 text-[#00f0ff] font-mono text-[10px] tracking-widest flex items-center gap-2">
          <span>RES: 0.5M GSD // CARTOSAT-2S / SENTINEL-2</span>
          <span className="w-2 h-2 border-t border-r border-[#00f0ff]" />
        </div>
        <div className="absolute bottom-8 left-8 text-slate-400 font-mono text-[10px] tracking-widest flex items-center gap-2">
          <span className="w-2 h-2 border-b border-l border-[#00f0ff]" />
          <span>WGS 84 / EPSG:4326 // CALIBRATED SURFACE</span>
        </div>
        <div className="absolute bottom-8 right-8 text-[#00f0ff] font-mono text-[10px] tracking-widest flex items-center gap-2">
          <span>MULTISPECTRAL L2A // SENSOR LOCK</span>
          <span className="w-2 h-2 border-b border-r border-[#00f0ff]" />
        </div>

        {/* "IMAGERY ACQUIRED // ANALYSIS READY" HUD Capsule */}
        {analysisReadyAlpha > 0.02 && (
          <div
            className="absolute bottom-12 sm:bottom-16 left-1/2 -translate-x-1/2 z-30 pointer-events-none transition-all duration-500"
            style={{
              opacity: analysisReadyAlpha,
              transform: `translate(-50%, ${(1 - analysisReadyAlpha) * 12}px)`,
            }}
          >
            <div className="flex items-center gap-3 px-5 py-2.5 rounded-full bg-[#030914]/90 border border-cyan-400/60 backdrop-blur-md shadow-[0_0_30px_rgba(0,240,255,0.25)]">
              <span className="w-2 h-2 rounded-full bg-cyan-400 animate-ping" />
              <span className="font-mono text-xs font-bold tracking-[0.22em] text-[#00f0ff] uppercase">
                IMAGERY ACQUIRED // ANALYSIS READY
              </span>
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
              <span className="font-mono text-[10.5px] tracking-wider text-slate-300 uppercase">
                FOOTPRINT STABILIZED
              </span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
