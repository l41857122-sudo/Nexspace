"use client";

import React, { useEffect } from "react";
import Lenis from "lenis";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import TopNavigationBar from "../TopNavigationBar";
import CustomCursor from "./CustomCursor";
import Hero3D from "./Hero3D";
import ProblemScene from "./ProblemScene";
import LiveInvestigationScene from "./LiveInvestigationScene";
import PixelsToIntelligenceScene from "./PixelsToIntelligenceScene";
import PipelineScene from "./PipelineScene";
import RealWorldIntelligenceScene from "./RealWorldIntelligenceScene";
import EvidenceScene from "./EvidenceScene";
import AboutTeamSection from "./AboutTeamSection";
import FinalScene from "./FinalScene";
import ScrollProgress from "./ScrollProgress";

export default function LandingPageContainer() {
  useEffect(() => {
    // 1. Initialize Lenis Smooth Scrolling
    const lenis = new Lenis({
      lerp: 0.08,
      smoothWheel: true,
    });
    if (typeof window !== "undefined") {
      (window as any).__lenis = lenis;
    }

    // 2. Synchronize Lenis with GSAP ScrollTrigger
    gsap.registerPlugin(ScrollTrigger);

    lenis.on("scroll", ScrollTrigger.update);

    const updateTicker = (time: number) => {
      lenis.raf(time * 1000);
    };

    gsap.ticker.add(updateTicker);
    gsap.ticker.lagSmoothing(0);

    return () => {
      lenis.destroy();
      gsap.ticker.remove(updateTicker);
    };
  }, []);

  return (
    <div className="relative w-full bg-[#010308] text-slate-100 overflow-x-clip">
      {/* Desktop Fluid Cyan Custom Cursor */}
      <CustomCursor />

      {/* Fixed Top Navigation Bar */}
      <TopNavigationBar />

      {/* Subtle Right-Side Vertical Navigation */}
      <ScrollProgress />

      {/* Scene 01: Hero 3D (Space, Orbit, Earth, Satellite) */}
      <Hero3D />

      {/* Scene 02: The Problem (Massive Data, Complex Analysis, Unclear Answers) */}
      <ProblemScene />

      {/* Scene 03: Live Investigation (Zooming Through Spatial Scales, Search, Sentinel-2 L2A) */}
      <LiveInvestigationScene />

      {/* Scene 04: Pixels Become Intelligence (Raw Imagery -> Spatial Analysis -> Verified Geometry) */}
      <PixelsToIntelligenceScene />

      {/* Scene 05: Investigation Pipeline (Understand -> Discover -> Analyze -> Verify) */}
      <PipelineScene />

      {/* Scene 06: Real-World Intelligence (Port, Deforestation, Infrastructure, Land Use) */}
      <RealWorldIntelligenceScene />

      {/* Scene 07: Evidence / Verification (Target Detected, Context, Pipeline, Verified Intelligence) */}
      <EvidenceScene />

      {/* Scene: Dedicated About / Team Cinematic Interactive Experience */}
      <AboutTeamSection />

      {/* Scene 08: Final Brand (Horizon, Orbiting Satellite, NEXSPACE Climax, Magnetic CTA) */}
      <FinalScene />
    </div>
  );
}
