"use client";

import React, { useEffect, useState, useRef } from "react";

export default function CustomCursor() {
  const [mounted, setMounted] = useState(false);
  const [cursorType, setCursorType] = useState<"default" | "interactive" | "explore" | "spatial">("default");
  const [isVisible, setIsVisible] = useState(false);

  const dotRef = useRef<HTMLDivElement>(null);
  const ringRef = useRef<HTMLDivElement>(null);
  const spotlightRef = useRef<HTMLDivElement>(null);

  const posRef = useRef({ x: -200, y: -200 });
  const ringPosRef = useRef({ x: -200, y: -200 });
  const animFrameRef = useRef<number | null>(null);

  useEffect(() => {
    // Only enable on desktop pointer devices
    const isTouch = window.matchMedia("(pointer: coarse)").matches;
    const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    if (isTouch || prefersReducedMotion) return;

    setMounted(true);

    const handleMouseMove = (e: MouseEvent) => {
      posRef.current.x = e.clientX;
      posRef.current.y = e.clientY;
      if (!isVisible) setIsVisible(true);

      // Check hovered element type
      const target = e.target as HTMLElement | null;
      if (!target) return;

      const spatial = target.closest('[data-spatial="true"], canvas, #hero, #pipeline, #pixels');
      const explore = target.closest('[data-cursor="explore"], .cursor-explore, .editorial-card');
      const interactive = target.closest(
        'button, a, input, [role="button"], [data-cursor="interactive"]'
      );

      if (interactive) {
        setCursorType("interactive");
      } else if (explore) {
        setCursorType("explore");
      } else if (spatial) {
        setCursorType("spatial");
      } else {
        setCursorType("default");
      }
    };

    const handleMouseLeave = () => setIsVisible(false);
    const handleMouseEnter = () => setIsVisible(true);

    window.addEventListener("mousemove", handleMouseMove, { passive: true });
    document.addEventListener("mouseleave", handleMouseLeave);
    document.addEventListener("mouseenter", handleMouseEnter);

    // Smooth fluid animation loop for trailing ring and spotlight
    const render = () => {
      // Lerp ring towards dot
      ringPosRef.current.x += (posRef.current.x - ringPosRef.current.x) * 0.18;
      ringPosRef.current.y += (posRef.current.y - ringPosRef.current.y) * 0.18;

      if (dotRef.current) {
        dotRef.current.style.transform = `translate3d(${posRef.current.x}px, ${posRef.current.y}px, 0) translate(-50%, -50%)`;
      }
      if (ringRef.current) {
        ringRef.current.style.transform = `translate3d(${ringPosRef.current.x}px, ${ringPosRef.current.y}px, 0) translate(-50%, -50%)`;
      }
      if (spotlightRef.current) {
        spotlightRef.current.style.transform = `translate3d(${posRef.current.x}px, ${posRef.current.y}px, 0) translate(-50%, -50%)`;
      }

      animFrameRef.current = requestAnimationFrame(render);
    };

    animFrameRef.current = requestAnimationFrame(render);

    return () => {
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
      window.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseleave", handleMouseLeave);
      document.removeEventListener("mouseenter", handleMouseEnter);
    };
  }, [isVisible]);

  if (!mounted || !isVisible) return null;

  return (
    <div className="fixed inset-0 pointer-events-none z-[9999] overflow-hidden">
      {/* Soft Radial Spotlight (Subtly illuminates underlying spatial layer) */}
      <div
        ref={spotlightRef}
        className={`fixed top-0 left-0 w-[420px] h-[420px] rounded-full pointer-events-none transition-opacity duration-500 ease-out ${
          cursorType === "spatial" ? "opacity-100" : "opacity-30"
        }`}
        style={{
          background:
            "radial-gradient(circle at center, rgba(0, 240, 255, 0.07) 0%, rgba(0, 180, 255, 0.02) 45%, transparent 70%)",
        }}
      />

      {/* Central Cyan Laser Point */}
      <div
        ref={dotRef}
        className="fixed top-0 left-0 w-2 h-2 rounded-full bg-[#00f0ff] shadow-[0_0_10px_#00f0ff] pointer-events-none transition-transform duration-75 ease-out"
      />

      {/* Fluid Trailing Orbital Ring */}
      <div
        ref={ringRef}
        className={`fixed top-0 left-0 rounded-full border border-cyan-400/60 pointer-events-none transition-[width,height,background-color,border-color] duration-300 ease-out flex items-center justify-center ${
          cursorType === "interactive"
            ? "w-11 h-11 bg-cyan-500/10 border-[#00f0ff] shadow-[0_0_20px_rgba(0,240,255,0.4)]"
            : cursorType === "explore"
            ? "w-16 h-16 bg-[#030914]/80 border-[#00f0ff] shadow-[0_0_24px_rgba(0,240,255,0.5)]"
            : cursorType === "spatial"
            ? "w-9 h-9 bg-transparent border-cyan-400/70 shadow-[0_0_16px_rgba(0,240,255,0.3)]"
            : "w-7 h-7 bg-transparent border-cyan-400/40"
        }`}
      >
        {cursorType === "explore" && (
          <span className="text-[8px] font-mono font-bold tracking-widest text-[#00f0ff] uppercase scale-90">
            EXPLORE
          </span>
        )}
        {cursorType === "spatial" && (
          <div className="w-1.5 h-1.5 rounded-full bg-[#00f0ff]/40 animate-ping" />
        )}
      </div>
    </div>
  );
}
