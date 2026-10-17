"use client";

import React, { useEffect, useState } from "react";
import { LANDING_SECTIONS, NavSection } from "../../lib/landing/scrollTimeline";

export default function ScrollProgress() {
  const [activeId, setActiveId] = useState("hero");

  useEffect(() => {
    const handleScroll = () => {
      const scrollHeight = document.documentElement.scrollHeight - window.innerHeight;
      if (scrollHeight <= 0) return;

      const progress = Math.min(1, Math.max(0, window.scrollY / scrollHeight));

      // Match user's exact required scroll percentage breakdown:
      // 0–12%: 01 HOME
      // 12–25%: 02 ORBIT
      // 25–40%: 03 IMAGERY
      // 40–55%: 04 ANALYSIS
      // 55–70%: 05 QUERY
      // 70–82%: 06 EVIDENCE
      // 82–92%: 07 MAP
      // 92–100%: 08 INTELLIGENCE
      const matched = LANDING_SECTIONS.find(
        (sec) => progress >= sec.minPct && progress < sec.maxPct
      );

      if (matched) {
        setActiveId(matched.id);
      } else if (progress >= 0.92) {
        setActiveId("intelligence");
      } else {
        setActiveId("hero");
      }
    };

    window.addEventListener("scroll", handleScroll, { passive: true });
    handleScroll();
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  const scrollToSection = (sec: NavSection) => {
    if (sec.id === "hero") {
      window.scrollTo({ top: 0, behavior: "smooth" });
      return;
    }

    const el = document.getElementById(sec.id);
    if (el) {
      el.scrollIntoView({ behavior: "smooth" });
    }
  };

  return (
    <nav
      aria-label="Section Navigation"
      className={`fixed right-6 lg:right-8 top-1/2 -translate-y-1/2 z-40 hidden lg:flex flex-col gap-3.5 font-mono text-[10.5px] select-none transition-opacity duration-300 ${activeId === "hero" ? "opacity-0 pointer-events-none" : "opacity-100 pointer-events-auto"
        }`}
    >
      {/* Subtle vertical spine connecting items */}
      <div className="absolute top-2 bottom-2 right-[4px] w-px bg-slate-800/80 z-0 pointer-events-none" />

      {LANDING_SECTIONS.map((sec) => {
        const isActive = activeId === sec.id;
        return (
          <button
            key={sec.id}
            type="button"
            onClick={() => scrollToSection(sec)}
            className="group relative z-10 flex items-center justify-end gap-3 transition-all duration-200 py-0.5 cursor-pointer text-right"
          >
            {/* Label (visible when active or on group hover) */}
            <span
              className={`tracking-widest uppercase transition-all duration-200 ${isActive
                  ? "text-[#00f0ff] font-bold opacity-100 drop-shadow-[0_0_8px_rgba(0,240,255,0.6)]"
                  : "text-slate-500 opacity-60 group-hover:opacity-100 group-hover:text-slate-300"
                }`}
            >
              {sec.num} {sec.label}
            </span>

            {/* Indicator Dot */}
            <span
              className={`block transition-all duration-300 rounded-full ${isActive
                  ? "w-2.5 h-2.5 bg-[#00f0ff] border border-white shadow-[0_0_12px_#00f0ff]"
                  : "w-2 h-2 bg-[#010612] border border-slate-700 group-hover:border-slate-500"
                }`}
            />
          </button>
        );
      })}
    </nav>
  );
}
