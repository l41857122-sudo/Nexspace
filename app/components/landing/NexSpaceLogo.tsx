"use client";

import React from "react";

interface NexSpaceLogoProps {
  size?: "sm" | "md" | "lg" | "xl";
  showText?: boolean;
  className?: string;
}

export default function NexSpaceLogo({
  size = "md",
  showText = true,
  className = "",
}: NexSpaceLogoProps) {
  const iconSizes = {
    sm: "w-5 h-5",
    md: "w-7 h-7",
    lg: "w-10 h-10",
    xl: "w-14 h-14",
  };

  const textSizes = {
    sm: "text-base tracking-[0.2em]",
    md: "text-lg sm:text-xl tracking-[0.24em]",
    lg: "text-2xl sm:text-3xl tracking-[0.28em]",
    xl: "text-4xl sm:text-5xl tracking-[0.32em]",
  };

  return (
    <div className={`inline-flex items-center gap-3 select-none ${className}`}>
      {/* Cyan Aerospace Delta Glyph matching Reference Panels 01 & 08 */}
      <svg
        className={`${iconSizes[size]} shrink-0 drop-shadow-[0_0_12px_rgba(0,240,255,0.75)]`}
        viewBox="0 0 40 40"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
      >
        <path
          d="M20 4L36 34H26.5L20 21.5L13.5 34H4L20 4Z"
          fill="#00f0ff"
        />
        <path
          d="M20 15L24 23H16L20 15Z"
          fill="#020712"
        />
      </svg>

      {showText && (
        <span
          className={`font-extrabold uppercase font-sans text-white ${textSizes[size]}`}
          style={{ letterSpacing: "0.22em" }}
        >
          NEXSPACE
        </span>
      )}
    </div>
  );
}
