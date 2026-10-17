"use client";

import React, { useState } from "react";
import Link from "next/link";
import { ArrowRight, MapPin, Eye, Compass, Scan } from "lucide-react";

interface ShowcaseItem {
  id: string;
  title: string;
  location: string;
  desc: string;
  image: string;
  queryParam: string;
  tag: string;
  metrics: string;
  colSpan: string; // Asymmetrical grid span
  baseZ: number;
}

const SHOWCASE_ITEMS: ShowcaseItem[] = [
  {
    id: "port-detection",
    title: "PORT DETECTION",
    location: "Mumbai, India",
    desc: "Identify and monitor maritime vessel density and berth status in real-time.",
    image: "/imagery/satellite_mumbai_port.jpg",
    queryParam: "Mumbai Port maritime vessels",
    tag: "MARITIME AOI",
    metrics: "48 Vessels • 94.2% Conf",
    colSpan: "lg:col-span-7",
    baseZ: 30,
  },
  {
    id: "deforestation",
    title: "DEFORESTATION ANALYSIS",
    location: "Amazon Rainforest",
    desc: "Track canopy depletion and logging activity over multi-temporal sensor series.",
    image: "/imagery/satellite_vegetation_canopy.jpg",
    queryParam: "Amazon Rainforest canopy cover",
    tag: "CANOPY LOSS",
    metrics: "-14.2 ha • Sentinel-2",
    colSpan: "lg:col-span-5",
    baseZ: -20,
  },
  {
    id: "infrastructure",
    title: "INFRASTRUCTURE MAPPING",
    location: "Delhi, India",
    desc: "Map transportation corridors, arterial roads, and municipal growth vectors.",
    image: "/imagery/satellite_infrastructure_roads.jpg",
    queryParam: "Delhi transportation corridors",
    tag: "ARTERIAL GRID",
    metrics: "128 km Road • EPSG:4326",
    colSpan: "lg:col-span-5",
    baseZ: 15,
  },
  {
    id: "land-use",
    title: "LAND USE CLASSIFICATION",
    location: "Dubai, UAE",
    desc: "Understand coastal reclamation, urbanization, and municipal zoning boundaries.",
    image: "/imagery/satellite_coastline_water.jpg",
    queryParam: "Dubai coastline urban expansion",
    tag: "URBAN GROWTH",
    metrics: "6 Zones • 10m Ground Res",
    colSpan: "lg:col-span-7",
    baseZ: -10,
  },
];

export default function RealWorldIntelligenceScene() {
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);
  const [imgShift, setImgShift] = useState<{ [key: string]: { x: number; y: number } }>({});

  const handleCardMouseMove = (
    e: React.MouseEvent<HTMLAnchorElement>,
    itemId: string
  ) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width;
    const y = (e.clientY - rect.top) / rect.height;
    const rotX = (0.5 - y) * 4; // ±2 deg
    const rotY = (x - 0.5) * 5; // ±2.5 deg
    e.currentTarget.style.transform = `perspective(1000px) rotateX(${rotX}deg) rotateY(${rotY}deg) translateY(-6px) scale(1.015)`;

    // Texture displacement shift (Requirement 7: subtle 4-6px shift)
    const shiftX = (x - 0.5) * 8;
    const shiftY = (y - 0.5) * 8;
    setImgShift((prev) => ({ ...prev, [itemId]: { x: shiftX, y: shiftY } }));
  };

  const handleCardMouseLeave = (
    e: React.MouseEvent<HTMLAnchorElement>,
    itemId: string
  ) => {
    setHoveredIdx(null);
    e.currentTarget.style.transform =
      "perspective(1000px) rotateX(0deg) rotateY(0deg) translateY(0px) scale(1)";
    setImgShift((prev) => ({ ...prev, [itemId]: { x: 0, y: 0 } }));
  };

  return (
    <section
      id="realworld"
      data-spatial="true"
      className="relative min-h-screen w-full bg-[#010308] text-slate-100 py-24 px-6 sm:px-10 lg:px-14 flex flex-col justify-center overflow-hidden border-t border-cyan-950/30 select-none"
    >
      {/* Ambient Lighting */}
      <div className="absolute top-1/2 right-1/3 -translate-y-1/2 w-[750px] h-[750px] bg-cyan-600/5 rounded-full blur-[180px] pointer-events-none" />

      <div className="max-w-[1440px] mx-auto w-full relative z-10 flex flex-col justify-between">
        {/* Section Header Row matching Reference Panel 06 */}
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 mb-12 pb-3 border-b border-slate-800/60">
          <div className="flex items-center gap-3">
            <span className="text-xs font-mono font-bold text-[#00f0ff] tracking-[0.25em] uppercase">
              06
            </span>
            <span className="text-slate-600 font-mono text-xs">/</span>
            <span className="text-xs font-mono tracking-[0.2em] text-slate-400 uppercase">
              0:27 — 0:35 REAL-WORLD INTELLIGENCE
            </span>
          </div>

          <div className="text-[11px] font-mono tracking-[0.22em] text-slate-400 uppercase">
            REAL DATA. REAL PLACES. REAL IMPACT.
          </div>
        </div>

        {/* Section Headline */}
        <div className="max-w-3xl mb-12">
          <h2 className="text-3xl sm:text-5xl lg:text-6xl font-extrabold tracking-tight text-white mb-3 uppercase font-sans">
            REAL-WORLD <span className="text-[#00f0ff]">INTELLIGENCE</span>
          </h2>
          <p className="text-slate-400 text-xs sm:text-base leading-relaxed font-light">
            Deploy autonomous satellite investigations across commercial maritime, environmental,
            and municipal scales.
          </p>
        </div>

        {/* Asymmetrical Staggered Spatial Gallery (Requirement 13: Break the standard 2x2 grid) */}
        <div
          className="grid grid-cols-1 lg:grid-cols-12 gap-6 mb-4"
          style={{ perspective: "1200px" }}
        >
          {SHOWCASE_ITEMS.map((item, index) => {
            const isHovered = hoveredIdx === index;
            const shift = imgShift[item.id] || { x: 0, y: 0 };

            return (
              <Link
                key={item.id}
                href={`/query?prompt=${encodeURIComponent(item.queryParam)}`}
                onMouseEnter={() => setHoveredIdx(index)}
                onMouseLeave={(e) => handleCardMouseLeave(e, item.id)}
                onMouseMove={(e) => handleCardMouseMove(e, item.id)}
                data-cursor="explore"
                className={`group relative rounded-2xl overflow-hidden bg-[#040c18] border transition-all duration-300 flex flex-col justify-between cursor-pointer ${
                  item.colSpan
                } ${
                  isHovered
                    ? "border-cyan-400 shadow-[0_16px_40px_rgba(0,240,255,0.25)] bg-[#051426] z-20"
                    : "border-slate-800/80 hover:border-slate-700 z-10"
                }`}
                style={{
                  transform: `perspective(1000px) translateZ(${isHovered ? item.baseZ + 30 : item.baseZ}px)`,
                  transition: "transform 0.3s cubic-bezier(0.16, 1, 0.3, 1), border-color 0.3s ease, box-shadow 0.3s ease",
                  transformStyle: "preserve-3d",
                }}
              >
                {/* Dynamic Radial Glare following cursor */}
                <div
                  className={`absolute inset-0 pointer-events-none transition-opacity duration-300 z-20 ${
                    isHovered ? "opacity-100" : "opacity-0"
                  }`}
                  style={{
                    background:
                      "radial-gradient(circle 260px at center, rgba(0,240,255,0.08), transparent 70%)",
                  }}
                />

                {/* Image Container with Texture Displacement (Requirement 7) */}
                <div className="relative aspect-[16/10] overflow-hidden bg-black">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={item.image}
                    alt={item.title}
                    className="w-full h-full object-cover transition-transform duration-200 ease-out"
                    style={{
                      transform: `translate3d(${shift.x}px, ${shift.y}px, 0) scale(${isHovered ? 1.06 : 1.02})`,
                    }}
                  />
                  <div className="absolute inset-0 bg-gradient-to-t from-[#040c18] via-transparent to-transparent opacity-80" />

                  {/* Micro Category Tag */}
                  <div className="absolute top-3 left-3 px-2 py-0.5 rounded bg-[#020712]/85 border border-slate-700/80 backdrop-blur-md text-[9px] font-mono text-cyan-300 uppercase tracking-wider z-10">
                    {item.tag}
                  </div>

                  {/* Analytical Layer Reveal on Hover */}
                  <div
                    className={`absolute inset-0 border-2 border-[#00f0ff] pointer-events-none transition-all duration-300 z-10 flex flex-col justify-between p-2.5 ${
                      isHovered ? "opacity-100 bg-[#00f0ff]/10" : "opacity-0"
                    }`}
                  >
                    <div className="flex justify-end">
                      <div className="px-2 py-0.5 rounded bg-[#020712]/95 border border-[#00f0ff] text-[8.5px] font-mono font-bold text-[#00f0ff] uppercase flex items-center gap-1 shadow-[0_0_10px_#00f0ff]">
                        <Scan className="w-2.5 h-2.5 animate-pulse" />
                        <span>ANALYSIS ACTIVE</span>
                      </div>
                    </div>
                    <div className="text-[9px] font-mono text-white bg-[#020712]/90 px-2 py-0.5 rounded border border-cyan-500/40 w-fit">
                      {item.metrics}
                    </div>
                  </div>
                </div>

                {/* Card Content Body */}
                <div className="p-5 flex flex-col justify-between flex-1 relative z-10">
                  <div>
                    <h3 className="text-base font-bold font-mono tracking-wide text-white uppercase mb-1 group-hover:text-[#00f0ff] transition-colors">
                      {item.title}
                    </h3>
                    <div className="text-xs font-mono text-[#00f0ff] mb-2 flex items-center gap-1">
                      <MapPin className="w-3 h-3" />
                      <span>{item.location}</span>
                    </div>
                    <p className="text-slate-400 text-xs leading-relaxed font-light mb-4">
                      {item.desc}
                    </p>
                  </div>

                  {/* Action Link Row */}
                  <div className="pt-3 border-t border-slate-800/70 flex items-center justify-between">
                    <span className="text-[11px] font-mono text-slate-400 group-hover:text-white transition-colors">
                      Explore Case
                    </span>
                    <div className="w-7 h-7 rounded-full bg-slate-900 border border-slate-800 flex items-center justify-center text-cyan-400 group-hover:bg-[#00f0ff] group-hover:text-[#020712] group-hover:border-[#00f0ff] group-hover:shadow-[0_0_12px_#00f0ff] transition-all">
                      <ArrowRight className="w-3.5 h-3.5 group-hover:translate-x-1 transition-transform" />
                    </div>
                  </div>
                </div>
              </Link>
            );
          })}
        </div>
      </div>
    </section>
  );
}
