"use client";

import { useRef, useEffect, useCallback } from "react";

interface TiltOptions {
  maxRotateX?: number; // degrees, default 2.5
  maxRotateY?: number; // degrees, default 3.0
  perspective?: number; // px, default 1000
  scale?: number; // default 1.015
  translateY?: number; // px, default -6
}

export function useCardTilt<T extends HTMLElement = HTMLDivElement>(
  options: TiltOptions = {}
) {
  const ref = useRef<T | null>(null);

  const {
    maxRotateX = 2.5,
    maxRotateY = 3.0,
    perspective = 1000,
    scale = 1.015,
    translateY = -6,
  } = options;

  const handleMouseMove = useCallback(
    (e: MouseEvent) => {
      const el = ref.current;
      if (!el) return;

      const rect = el.getBoundingClientRect();
      const x = (e.clientX - rect.left) / rect.width; // 0 to 1
      const y = (e.clientY - rect.top) / rect.height; // 0 to 1

      const rotX = (0.5 - y) * 2 * maxRotateX;
      const rotY = (x - 0.5) * 2 * maxRotateY;

      const glareX = x * 100;
      const glareY = y * 100;

      el.style.transform = `perspective(${perspective}px) rotateX(${rotX}deg) rotateY(${rotY}deg) translateY(${translateY}px) scale(${scale})`;
      el.style.setProperty("--glare-x", `${glareX}%`);
      el.style.setProperty("--glare-y", `${glareY}%`);
      el.style.setProperty("--glare-opacity", "1");
    },
    [maxRotateX, maxRotateY, perspective, scale, translateY]
  );

  const handleMouseLeave = useCallback(() => {
    const el = ref.current;
    if (!el) return;

    el.style.transform = `perspective(${perspective}px) rotateX(0deg) rotateY(0deg) translateY(0px) scale(1)`;
    el.style.setProperty("--glare-opacity", "0");
  }, [perspective]);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;

    // Check prefers-reduced-motion
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;

    el.style.transition = "transform 0.4s cubic-bezier(0.16, 1, 0.3, 1)";
    el.style.willChange = "transform";

    el.addEventListener("mousemove", handleMouseMove as any, { passive: true });
    el.addEventListener("mouseleave", handleMouseLeave as any);

    return () => {
      el.removeEventListener("mousemove", handleMouseMove as any);
      el.removeEventListener("mouseleave", handleMouseLeave as any);
    };
  }, [handleMouseMove, handleMouseLeave]);

  return ref;
}
