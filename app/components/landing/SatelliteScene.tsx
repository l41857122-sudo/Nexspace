"use client";

import React from "react";

interface SatelliteSceneProps {
  scrollProgress: number; // 0 to 1
  opacity?: number; // controlled by master timeline
  isReducedMotion?: boolean;
}

/**
 * SatelliteScene / Orbital Trajectory Overlay
 * Renders the scientific curved orbital trajectory line and HUD telemetry leader.
 * The 3D spacecraft itself is rendered in genuine 3D WebGL inside EarthScene.
 */
export default function SatelliteScene({
  scrollProgress,
  opacity,
}: SatelliteSceneProps) {
  const effectiveOpacity =
    opacity !== undefined
      ? opacity
      : scrollProgress < 0.70
      ? 1.0
      : Math.max(0, 1 - (scrollProgress - 0.70) / 0.15);

  if (effectiveOpacity <= 0.01) return null;

  return (
    <div
      className="absolute inset-0 pointer-events-none overflow-hidden z-20 select-none transition-opacity duration-200"
      style={{ opacity: effectiveOpacity }}
      aria-hidden="true"
    >
      {/* Luminous deep atmospheric radial glow behind the particle globe */}
      <div
        className="absolute top-1/2 -translate-y-1/2 pointer-events-none"
        style={{
          right: "2%",
          width: "58vw",
          height: "58vw",
          maxHeight: "820px",
          maxWidth: "820px",
          background:
            "radial-gradient(circle, rgba(0,240,255,0.12) 0%, rgba(0,100,220,0.06) 42%, rgba(0,20,60,0) 72%)",
        }}
      />

      {/* Scientific Orbital Telemetry Ellipses & Satellite Labels (matching approved mockup) */}
      <svg
        className="absolute inset-0 w-full h-full pointer-events-none"
        viewBox="0 0 1440 900"
        preserveAspectRatio="xMidYMid meet"
      >
        <defs>
          <linearGradient id="orbit-cyan-grad" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#00f0ff" stopOpacity="0.75" />
            <stop offset="50%" stopColor="#00b0ff" stopOpacity="0.35" />
            <stop offset="100%" stopColor="#0044ff" stopOpacity="0.10" />
          </linearGradient>
          <linearGradient id="orbit-dash-grad" x1="100%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="#00f0ff" stopOpacity="0.85" />
            <stop offset="60%" stopColor="#00e5ff" stopOpacity="0.30" />
            <stop offset="100%" stopColor="#002288" stopOpacity="0.05" />
          </linearGradient>
          <filter id="cyan-glow" x="-20%" y="-20%" width="140%" height="140%">
            <feGaussianBlur stdDeviation="2.0" result="blur" />
            <feComposite in="SourceGraphic" in2="blur" operator="over" />
          </filter>
        </defs>

        {/* Globe Center Anchor: (1030, 450) */}
        <g transform="translate(1030, 450)">
          {/* Orbital Ring 1: Tilted -28 deg */}
          <g transform="rotate(-28)">
            <ellipse
              cx="0"
              cy="0"
              rx="330"
              ry="240"
              fill="none"
              stroke="url(#orbit-cyan-grad)"
              strokeWidth="1.2"
              filter="url(#cyan-glow)"
              className="opacity-70"
            />
            {/* Dashed Telemetry Segment */}
            <ellipse
              cx="0"
              cy="0"
              rx="330"
              ry="240"
              fill="none"
              stroke="#00f0ff"
              strokeWidth="1.5"
              strokeDasharray="6 8"
              strokeDashoffset={-scrollProgress * 240}
              className="opacity-80"
            />

            {/* Satellite 1 node with telemetry leader */}
            <g transform="translate(230, -172)">
              <circle r="3.5" fill="#00f0ff" filter="url(#cyan-glow)" />
              <circle r="7" fill="none" stroke="#00f0ff" strokeWidth="0.8" opacity="0.6" className="animate-ping" />
              {/* Miniature satellite icon shape */}
              <rect x="-1.5" y="-3.5" width="3" height="7" fill="#ffffff" />
              <rect x="-7" y="-1.5" width="4.5" height="3" fill="#00e5ff" opacity="0.85" />
              <rect x="2.5" y="-1.5" width="4.5" height="3" fill="#00e5ff" opacity="0.85" />
              <line x1="8" y1="-6" x2="28" y2="-18" stroke="#00f0ff" strokeWidth="0.75" opacity="0.7" />
              <text x="32" y="-16" fill="#00f0ff" fontSize="7.5" fontFamily="monospace" letterSpacing="0.16em" opacity="0.85">
                SAT-NET STATUS
              </text>
            </g>
          </g>

          {/* Orbital Ring 2: Tilted +36 deg */}
          <g transform="rotate(36)">
            <ellipse
              cx="0"
              cy="0"
              rx="315"
              ry="220"
              fill="none"
              stroke="url(#orbit-dash-grad)"
              strokeWidth="1.0"
              strokeDasharray="4 6"
              filter="url(#cyan-glow)"
              className="opacity-65"
            />

            {/* Satellite 2 node */}
            <g transform="translate(-220, 155)">
              <circle r="3" fill="#00f0ff" />
              <rect x="-1.5" y="-3" width="3" height="6" fill="#ffffff" />
              <rect x="-6" y="-1.2" width="3.5" height="2.4" fill="#00c8ff" />
              <rect x="2.5" y="-1.2" width="3.5" height="2.4" fill="#00c8ff" />
              <line x1="-8" y1="6" x2="-26" y2="18" stroke="#00f0ff" strokeWidth="0.75" opacity="0.6" />
              <text x="-124" y="22" fill="#00f0ff" fontSize="7" fontFamily="monospace" letterSpacing="0.14em" opacity="0.75">
                ORBITAL TELEMETRY
              </text>
            </g>
          </g>

          {/* Orbital Ring 3: Subtle outer coordinate boundary */}
          <g transform="rotate(8)">
            <ellipse
              cx="0"
              cy="0"
              rx="355"
              ry="260"
              fill="none"
              stroke="#00f0ff"
              strokeWidth="0.75"
              strokeDasharray="2 12"
              className="opacity-40"
            />
            {/* Scientific annotation label top */}
            <text x="-40" y="-268" fill="#00f0ff" fontSize="7" fontFamily="monospace" letterSpacing="0.18em" opacity="0.7">
              GEODETIC DATA: GL4874
            </text>
          </g>
        </g>
      </svg>
    </div>
  );
}
