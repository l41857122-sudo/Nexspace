"use client";

import React, { useEffect, useRef, useState } from "react";

interface HeroHeadlineSequenceProps {
  onPhaseChange?: (phase: number) => void;
  isReducedMotion?: boolean;
}

interface Particle {
  x: number;
  y: number;
  vx: number;
  vy: number;
  size: number;
  alpha: number;
  maxAlpha: number;
  life: number;
  maxLife: number;
  color: string;
}

const FONT_FAMILY = 'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';

// ==============================================================
// TIMING CONSTANTS (Total: 18.4 seconds per full continuous loop)
// ==============================================================
// Phase 1: NEXSPACE Brand Showcase Hold  ->  5.0s  (0.0s  to  5.0s) [Cut from 8s]
// Transition 1: NEXSPACE -> ASK Morph     ->  1.8s  (5.0s  to  6.8s)
// Phase 2: ASK Visible & Query Subtitle   ->  2.0s  (6.8s  to  8.8s)
// Transition 2: ASK -> ANALYZE Laser Scan ->  1.8s  (8.8s  to 10.6s)
// Phase 3: ANALYZE Visible & Detection    ->  2.0s  (10.6s to 12.6s)
// Transition 3: ANALYZE -> VERIFY Converg ->  1.8s  (12.6s to 14.4s)
// Phase 4: VERIFY Visible & Locked Ring   ->  2.0s  (14.4s to 16.4s)
// Transition 4: VERIFY -> NEXSPACE Morph  ->  2.0s  (16.4s to 18.4s)
// -> Loops continuously back to NEXSPACE 5-second brand hold!
// ==============================================================
const T_NEXSPACE_HOLD = 5.0;
const T_NEX_TO_ASK    = 6.8;
const T_ASK_HOLD      = 8.8;
const T_ASK_TO_ANA    = 10.6;
const T_ANA_HOLD      = 12.6;
const T_ANA_TO_VER    = 14.4;
const T_VER_HOLD      = 16.4;
const TOTAL_CYCLE     = 18.4;

export default function HeroHeadlineSequence({
  onPhaseChange,
  isReducedMotion = false,
}: HeroHeadlineSequenceProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animFrameRef = useRef<number | null>(null);

  // Store onPhaseChange in a ref so useEffect NEVER re-mounts on parent re-renders
  const onPhaseChangeRef = useRef(onPhaseChange);
  useEffect(() => {
    onPhaseChangeRef.current = onPhaseChange;
  }, [onPhaseChange]);

  // Deterministic inspection parameter: ?heroState=nexspace | ask | analyze | verify | morph or ?heroTime=X
  const [forcedTime] = useState<number | null>(() => {
    if (typeof window !== "undefined") {
      const params = new URLSearchParams(window.location.search);
      const timeParam = params.get("heroTime");
      if (timeParam) return parseFloat(timeParam);
      const stateParam = params.get("heroState") || params.get("heroPhase");
      if (stateParam) {
        if (stateParam === "nexspace" || stateParam === "1") return 2.5;   // Middle of 5s NEXSPACE hold
        if (stateParam === "ask"      || stateParam === "2") return 7.8;   // ASK visible
        if (stateParam === "analyze"  || stateParam === "3") return 11.6;  // ANALYZE visible
        if (stateParam === "verify"   || stateParam === "4") return 15.4;  // VERIFY visible
        if (stateParam === "morph")                          return 17.4;  // VERIFY -> NEXSPACE morph
      }
    }
    return null;
  });

  const currentPhaseRef = useRef<number>(1);
  const particlesRef = useRef<Particle[]>([]);
  const startTimeRef = useRef<number>(0);

  // Smooth easing functions
  const smoothstep = (min: number, max: number, value: number) => {
    const x = Math.max(0, Math.min(1, (value - min) / (max - min)));
    return x * x * (3 - 2 * x);
  };

  const easeInOutCubic = (t: number) => {
    return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
  };

  const easeOutQuart = (t: number) => {
    return 1 - Math.pow(1 - t, 4);
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let isRunning = true;

    // Handle high-DPI retina display crispness
    const handleResize = () => {
      if (!canvas || !containerRef.current) return;
      const dpr = window.devicePixelRatio || 1;
      const rect = containerRef.current.getBoundingClientRect();
      const w = Math.max(rect.width || 0, 680);
      const h = Math.max(rect.height || 0, 130);
      canvas.width = Math.floor(w * dpr);
      canvas.height = Math.floor(h * dpr);
      canvas.style.width = `${w}px`;
      canvas.style.height = `${h}px`;
    };

    handleResize();
    window.addEventListener("resize", handleResize);

    // Initialize ambient telemetry particles
    const initParticles = () => {
      const particles: Particle[] = [];
      for (let i = 0; i < 35; i++) {
        particles.push({
          x: Math.random() * 800,
          y: Math.random() * 140,
          vx: (Math.random() - 0.5) * 0.35,
          vy: (Math.random() - 0.5) * 0.25,
          size: Math.random() * 1.5 + 0.5,
          alpha: Math.random() * 0.35,
          maxAlpha: Math.random() * 0.45 + 0.15,
          life: Math.random() * 120,
          maxLife: Math.random() * 120 + 120,
          color: Math.random() > 0.3 ? "#00f0ff" : "#ffffff",
        });
      }
      particlesRef.current = particles;
    };
    initParticles();

    startTimeRef.current = performance.now();

    // RENDER LOOP (Continuous Modulo Timeline)
    const render = (now: number) => {
      if (!isRunning) return;

      const dpr = window.devicePixelRatio || 1;
      const rect = containerRef.current?.getBoundingClientRect();
      const width = rect && rect.width > 0 ? Math.max(rect.width, 680) : 700;
      const height = rect && rect.height > 0 ? Math.max(rect.height, 130) : 135;

      // Ensure canvas internal resolution matches container on the fly if needed
      if (canvas.width !== Math.floor(width * dpr) || canvas.height !== Math.floor(height * dpr)) {
        canvas.width = Math.floor(width * dpr);
        canvas.height = Math.floor(height * dpr);
        canvas.style.width = `${width}px`;
        canvas.style.height = `${height}px`;
      }

      // Set hardware-accelerated transform to DPR
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      // Continuous loop time (0.0s to 18.4s)
      let t = 0;
      if (forcedTime !== null) {
        t = forcedTime;
      } else {
        const elapsed = (now - startTimeRef.current) / 1000;
        t = elapsed % TOTAL_CYCLE;
      }

      // Determine active headline phase for eyebrow & process indicator
      // Phase 1: NEXSPACE (0.0s - 5.0s) & (16.4s - 18.4s)
      // Phase 2: ASK (5.0s - 8.8s)
      // Phase 3: ANALYZE (8.8s - 12.6s)
      // Phase 4: VERIFY (12.6s - 16.4s)
      let currentPhase = 1;
      if (t >= T_NEXSPACE_HOLD && t < T_ASK_HOLD) {
        currentPhase = 2; // ASK
      } else if (t >= T_ASK_HOLD && t < T_ANA_HOLD) {
        currentPhase = 3; // ANALYZE
      } else if (t >= T_ANA_HOLD && t < T_VER_HOLD) {
        currentPhase = 4; // VERIFY
      } else {
        currentPhase = 1; // NEXSPACE
      }

      if (currentPhase !== currentPhaseRef.current) {
        currentPhaseRef.current = currentPhase;
        onPhaseChangeRef.current?.(currentPhase);
      }

      // Clear frame
      ctx.clearRect(0, 0, width, height);

      // ----------------------------------------------------
      // DRAW AMBIENT SPATIAL PARTICLES
      // ----------------------------------------------------
      ctx.save();
      const particles = particlesRef.current;
      for (let i = 0; i < particles.length; i++) {
        const p = particles[i];
        p.x += p.vx;
        p.y += p.vy;
        p.life++;
        if (p.x < 0) p.x = width;
        if (p.x > width) p.x = 0;
        if (p.y < 0) p.y = height;
        if (p.y > height) p.y = 0;
        if (p.life > p.maxLife) {
          p.life = 0;
          p.alpha = 0;
        } else if (p.life < 20) {
          p.alpha = (p.life / 20) * p.maxAlpha;
        } else if (p.life > p.maxLife - 20) {
          p.alpha = ((p.maxLife - p.life) / 20) * p.maxAlpha;
        }

        ctx.fillStyle = p.color;
        ctx.globalAlpha = p.alpha;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.restore();

      // ----------------------------------------------------
      // TYPOGRAPHY CONFIGURATION
      // ----------------------------------------------------
      const isMobile = width < 640;
      const isTablet = width >= 640 && width < 1024;
      const baseFontSize = isMobile ? 50 : isTablet ? 68 : 82;
      const startY = height * 0.48;

      ctx.textBaseline = "middle";
      ctx.textAlign = "left";

      // ==============================================================
      // STAGE 1: NEXSPACE 5-SECOND BRAND SHOWCASE HOLD (0.0s – 5.0s)
      // ==============================================================
      if (t < T_NEXSPACE_HOLD) {
        // Completely stable brand showcase, no glitch, no typewriter, no reset
        ctx.save();
        ctx.font = `900 ${baseFontSize}px ${FONT_FAMILY}`;
        ctx.fillStyle = "#ffffff";
        ctx.shadowColor = "rgba(0, 240, 255, 0.4)";
        ctx.shadowBlur = 22;
        ctx.globalAlpha = 1.0;

        const text = "NEXSPACE";
        const letterSpacing = 0.06 * baseFontSize;
        let curX = 4;
        for (let i = 0; i < text.length; i++) {
          const char = text[i];
          ctx.fillText(char, curX, startY);
          curX += ctx.measureText(char).width + letterSpacing;
        }

        // Elegant aerospace corner brackets framing the brand
        ctx.strokeStyle = "rgba(0, 240, 255, 0.35)";
        ctx.lineWidth = 1.5;
        const totalW = curX + 6;
        const boxTop = startY - baseFontSize * 0.44;
        const boxBottom = startY + baseFontSize * 0.44;
        const bLen = 14;

        ctx.beginPath();
        // Top-left
        ctx.moveTo(0, boxTop + bLen); ctx.lineTo(0, boxTop); ctx.lineTo(bLen, boxTop);
        // Top-right
        ctx.moveTo(totalW - bLen, boxTop); ctx.lineTo(totalW, boxTop); ctx.lineTo(totalW, boxTop + bLen);
        // Bottom-left
        ctx.beginPath();
        ctx.moveTo(0, boxBottom - bLen); ctx.lineTo(0, boxBottom); ctx.lineTo(bLen, boxBottom);
        // Bottom-right
        ctx.moveTo(totalW - bLen, boxBottom); ctx.lineTo(totalW, boxBottom); ctx.lineTo(totalW, boxBottom - bLen);
        ctx.stroke();

        ctx.restore();

      // ==============================================================
      // TRANSITION 1: NEXSPACE -> ASK SLOW KINETIC MORPH (5.0s – 6.8s)
      // ==============================================================
      } else if (t >= T_NEXSPACE_HOLD && t < T_NEX_TO_ASK) {
        const p = easeInOutCubic((t - T_NEXSPACE_HOLD) / 1.8);
        const originText = "NEXSPACE";

        ctx.save();

        // 1. Kinetic decomposition: 8 letters slide horizontally & contract
        if (p < 0.88) {
          const dissolveAlpha = Math.max(0, 1 - p * 1.25);
          ctx.font = `900 ${baseFontSize}px ${FONT_FAMILY}`;
          ctx.fillStyle = "#ffffff";
          ctx.shadowColor = "#00f0ff";
          ctx.shadowBlur = 18 * (1 - p);
          ctx.globalAlpha = dissolveAlpha;

          let curX = 4;
          const letterSpacing = 0.06 * baseFontSize * (1 - p * 0.5);
          for (let i = 0; i < originText.length; i++) {
            const char = originText[i];
            const charW = ctx.measureText(char).width;
            const targetOffset = (i > 2 ? -(i - 2) * 26 * p : 0);
            const yWobble = Math.sin(p * Math.PI + i) * 3 * p;
            ctx.fillText(char, curX + targetOffset, startY + yWobble);
            curX += charW + letterSpacing;
          }
        }

        // 2. Cyan vector trails & horizontal energy lines connecting the morph
        const trailAlpha = Math.sin(p * Math.PI) * 0.85;
        if (trailAlpha > 0.05) {
          ctx.strokeStyle = "#00f0ff";
          ctx.lineWidth = 1.5;
          ctx.globalAlpha = trailAlpha;
          ctx.shadowColor = "#00f0ff";
          ctx.shadowBlur = 12;

          for (let k = 0; k < 5; k++) {
            const lineY = startY - baseFontSize * 0.25 + k * (baseFontSize * 0.14);
            const lineStartX = 4 + k * 16 * p;
            const lineEndX = lineStartX + (width * 0.42) * (1 - p * 0.5);
            ctx.beginPath();
            ctx.moveTo(lineStartX, lineY);
            ctx.lineTo(lineEndX, lineY);
            ctx.stroke();

            // Coordinate sparks along lines
            ctx.fillStyle = "#ffffff";
            ctx.beginPath();
            ctx.arc(lineStartX + (lineEndX - lineStartX) * p, lineY, 2, 0, Math.PI * 2);
            ctx.fill();
          }
        }

        // 3. ASK letters emerging smoothly from converging strokes
        const askAlpha = smoothstep(0.3, 0.95, p);
        const askScale = 0.92 + 0.08 * p;
        const askFontSize = baseFontSize * 1.05 * askScale;

        if (askAlpha > 0) {
          ctx.font = `900 ${askFontSize}px ${FONT_FAMILY}`;
          ctx.fillStyle = "#00f0ff";
          ctx.shadowColor = "rgba(0, 240, 255, 0.75)";
          ctx.shadowBlur = 28 * askAlpha;
          ctx.globalAlpha = askAlpha;
          ctx.fillText("ASK", 4, startY);

          // Underline drawing
          const lineProgress = smoothstep(0.6, 1.0, p);
          if (lineProgress > 0) {
            const askW = ctx.measureText("ASK").width;
            ctx.strokeStyle = "#00f0ff";
            ctx.lineWidth = 2.5;
            ctx.shadowColor = "#00f0ff";
            ctx.shadowBlur = 10;
            ctx.beginPath();
            ctx.moveTo(4, startY + askFontSize * 0.42);
            ctx.lineTo(4 + (askW + 20) * lineProgress, startY + askFontSize * 0.42);
            ctx.stroke();
          }
        }

        ctx.restore();

      // ==============================================================
      // STAGE 2: ASK VISIBLE HOLD (6.8s – 8.8s)
      // ==============================================================
      } else if (t >= T_NEX_TO_ASK && t < T_ASK_HOLD) {
        ctx.save();
        const askFontSize = baseFontSize * 1.05;
        ctx.font = `900 ${askFontSize}px ${FONT_FAMILY}`;
        ctx.fillStyle = "#00f0ff";
        ctx.shadowColor = "rgba(0, 240, 255, 0.75)";
        ctx.shadowBlur = 28;
        ctx.globalAlpha = 1.0;
        ctx.fillText("ASK", 4, startY);

        // Stabilized underline
        const askW = ctx.measureText("ASK").width;
        ctx.strokeStyle = "#00f0ff";
        ctx.lineWidth = 2.5;
        ctx.shadowColor = "#00f0ff";
        ctx.shadowBlur = 10;
        ctx.beginPath();
        ctx.moveTo(4, startY + askFontSize * 0.42);
        ctx.lineTo(4 + askW + 20, startY + askFontSize * 0.42);
        ctx.stroke();

        ctx.restore();

      // ==============================================================
      // TRANSITION 2: ASK -> ANALYZE LASER SCAN MORPH (8.8s – 10.6s)
      // ==============================================================
      } else if (t >= T_ASK_HOLD && t < T_ASK_TO_ANA) {
        const p = easeInOutCubic((t - T_ASK_HOLD) / 1.8);
        ctx.save();

        // 1. Vertical cyan laser scanline sweeping across
        const scanX = 4 + (width * 0.65) * p;
        const laserGrad = ctx.createLinearGradient(scanX - 35, 0, scanX + 25, 0);
        laserGrad.addColorStop(0, "rgba(0, 240, 255, 0)");
        laserGrad.addColorStop(0.7, "rgba(0, 240, 255, 0.3)");
        laserGrad.addColorStop(1, "rgba(0, 240, 255, 0.9)");

        ctx.fillStyle = laserGrad;
        ctx.fillRect(scanX - 35, startY - baseFontSize * 0.6, 40, baseFontSize * 1.25);

        // Core bright laser wavefront
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 2;
        ctx.shadowColor = "#00f0ff";
        ctx.shadowBlur = 16;
        ctx.beginPath();
        ctx.moveTo(scanX, startY - baseFontSize * 0.6);
        ctx.lineTo(scanX, startY + baseFontSize * 0.65);
        ctx.stroke();

        // 2. ASK strokes before scanline
        const askAlpha = Math.max(0, 1 - p * 1.5);
        if (askAlpha > 0) {
          ctx.font = `900 ${baseFontSize * 1.05}px ${FONT_FAMILY}`;
          ctx.fillStyle = "#00f0ff";
          ctx.globalAlpha = askAlpha;
          ctx.shadowColor = "rgba(0, 240, 255, 0.5)";
          ctx.shadowBlur = 18;
          ctx.fillText("ASK", 4, startY);
        }

        // 3. Emerging ANALYZE typography revealed by laser
        const analyzeAlpha = smoothstep(0.2, 0.85, p);
        if (analyzeAlpha > 0) {
          ctx.font = `900 ${baseFontSize * 1.02}px ${FONT_FAMILY}`;
          const grad = ctx.createLinearGradient(4, 0, 4 + baseFontSize * 5.5, 0);
          grad.addColorStop(0, "#ffffff");
          grad.addColorStop(0.65, "#f0fbff");
          grad.addColorStop(1, "#00f0ff");

          ctx.fillStyle = grad;
          ctx.globalAlpha = analyzeAlpha;
          ctx.shadowColor = "rgba(0, 240, 255, 0.4)";
          ctx.shadowBlur = 24;

          ctx.save();
          ctx.beginPath();
          ctx.rect(0, startY - baseFontSize * 0.7, scanX + 20, baseFontSize * 1.5);
          ctx.clip();
          ctx.fillText("ANALYZE", 4, startY);
          ctx.restore();
        }

        // 4. Detection Geometry Bounding Boxes tracing around ANALYZE
        const boxProgress = smoothstep(0.35, 0.95, p);
        if (boxProgress > 0) {
          ctx.strokeStyle = "rgba(0, 240, 255, 0.6)";
          ctx.lineWidth = 1.2;
          ctx.globalAlpha = boxProgress * 0.9;
          ctx.shadowColor = "#00f0ff";
          ctx.shadowBlur = 8;

          const bw = (isMobile ? 310 : isTablet ? 420 : 490) * Math.min(1, boxProgress * 1.1);
          const bh = baseFontSize * 0.82;
          const bx = 4;
          const by = startY - bh * 0.5;
          const clen = 12;

          ctx.beginPath();
          ctx.moveTo(bx, by + clen); ctx.lineTo(bx, by); ctx.lineTo(bx + clen, by);
          ctx.moveTo(bx + bw - clen, by); ctx.lineTo(bx + bw, by); ctx.lineTo(bx + bw, by + clen);
          ctx.moveTo(bx, by + bh - clen); ctx.lineTo(bx, by + bh); ctx.lineTo(bx + clen, by + bh);
          ctx.moveTo(bx + bw - clen, by + bh); ctx.lineTo(bx + bw, by + bh); ctx.lineTo(bx + bw, by + bh - clen);
          ctx.stroke();

          ctx.fillStyle = "#00f0ff";
          ctx.font = "600 10px monospace";
          ctx.fillText(`TGT_01 [CONF: ${(0.96 + p * 0.03).toFixed(3)}] // MULTI_SPECTRAL`, bx + 16, by - 6);
        }

        ctx.restore();

      // ==============================================================
      // STAGE 3: ANALYZE VISIBLE HOLD (10.6s – 12.6s)
      // ==============================================================
      } else if (t >= T_ASK_TO_ANA && t < T_ANA_HOLD) {
        ctx.save();
        ctx.font = `900 ${baseFontSize * 1.02}px ${FONT_FAMILY}`;
        const grad = ctx.createLinearGradient(4, 0, 4 + baseFontSize * 5.5, 0);
        grad.addColorStop(0, "#ffffff");
        grad.addColorStop(0.65, "#f0fbff");
        grad.addColorStop(1, "#00f0ff");

        ctx.fillStyle = grad;
        ctx.globalAlpha = 1.0;
        ctx.shadowColor = "rgba(0, 240, 255, 0.45)";
        ctx.shadowBlur = 24;
        ctx.fillText("ANALYZE", 4, startY);

        // Stabilized Bounding Box
        ctx.strokeStyle = "rgba(0, 240, 255, 0.6)";
        ctx.lineWidth = 1.2;
        ctx.shadowColor = "#00f0ff";
        ctx.shadowBlur = 8;
        const bw = (isMobile ? 310 : isTablet ? 420 : 490);
        const bh = baseFontSize * 0.82;
        const bx = 4;
        const by = startY - bh * 0.5;
        const clen = 12;

        ctx.beginPath();
        ctx.moveTo(bx, by + clen); ctx.lineTo(bx, by); ctx.lineTo(bx + clen, by);
        ctx.moveTo(bx + bw - clen, by); ctx.lineTo(bx + bw, by); ctx.lineTo(bx + bw, by + clen);
        ctx.moveTo(bx, by + bh - clen); ctx.lineTo(bx, by + bh); ctx.lineTo(bx + clen, by + bh);
        ctx.moveTo(bx + bw - clen, by + bh); ctx.lineTo(bx + bw, by + bh); ctx.lineTo(bx + bw, by + bh - clen);
        ctx.stroke();

        ctx.fillStyle = "#00f0ff";
        ctx.font = "600 10px monospace";
        ctx.fillText(`TGT_01 [CONF: 0.984] // MULTI_SPECTRAL`, bx + 16, by - 6);

        ctx.restore();

      // ==============================================================
      // TRANSITION 3: ANALYZE -> VERIFY CONVERGENCE MORPH (12.6s – 14.4s)
      // ==============================================================
      } else if (t >= T_ANA_HOLD && t < T_ANA_TO_VER) {
        const p = easeInOutCubic((t - T_ANA_HOLD) / 1.8);
        ctx.save();

        // 1. Spatial convergence rays streaming from Earth
        const rayAlpha = Math.sin(p * Math.PI) * 0.75;
        if (rayAlpha > 0.05) {
          ctx.strokeStyle = "rgba(0, 240, 255, 0.4)";
          ctx.lineWidth = 1;
          ctx.globalAlpha = rayAlpha;
          ctx.setLineDash([6, 6]);

          const targetX = width * 0.35;
          const targetY = startY;

          for (let r = 0; r < 5; r++) {
            const startRayX = width * 0.95;
            const startRayY = height * 0.1 + r * (height * 0.2);
            ctx.beginPath();
            ctx.moveTo(startRayX, startRayY);
            ctx.lineTo(targetX, targetY);
            ctx.stroke();
          }
          ctx.setLineDash([]);
        }

        // 2. Compressing geometry: ANALYZE compresses inward as VERIFY forms
        const analyzeAlpha = Math.max(0, 1 - p * 1.5);
        if (analyzeAlpha > 0) {
          ctx.font = `900 ${baseFontSize * 1.02}px ${FONT_FAMILY}`;
          ctx.fillStyle = "#ffffff";
          ctx.globalAlpha = analyzeAlpha;
          ctx.shadowColor = "#00f0ff";
          ctx.shadowBlur = 18;
          ctx.fillText("ANALYZE", 4, startY);
        }

        // 3. VERIFY Letters forming with sharp cyan precision
        const verifyAlpha = smoothstep(0.25, 0.85, p);
        if (verifyAlpha > 0) {
          ctx.font = `900 ${baseFontSize * 1.05}px ${FONT_FAMILY}`;
          ctx.fillStyle = "#00f0ff";
          ctx.globalAlpha = verifyAlpha;
          ctx.shadowColor = "rgba(0, 240, 255, 0.75)";
          ctx.shadowBlur = 28 * verifyAlpha;
          ctx.fillText("VERIFY", 4, startY);

          // 4. Verification Ring drawing around check indicator
          const ringProgress = smoothstep(0.55, 0.95, p);
          if (ringProgress > 0) {
            const verifyW = ctx.measureText("VERIFY").width;
            const ringX = 4 + verifyW + (isMobile ? 26 : 38);
            const ringY = startY;
            const radius = isMobile ? 18 : 24;

            ctx.strokeStyle = "#00f0ff";
            ctx.lineWidth = 2.5;
            ctx.shadowColor = "#00f0ff";
            ctx.shadowBlur = 16;
            ctx.globalAlpha = ringProgress;

            ctx.beginPath();
            ctx.arc(ringX, ringY, radius, -Math.PI / 2, -Math.PI / 2 + Math.PI * 2 * ringProgress);
            ctx.stroke();

            // Trace checkmark inside ring
            const checkProgress = smoothstep(0.7, 1.0, p);
            if (checkProgress > 0) {
              ctx.strokeStyle = "#ffffff";
              ctx.lineWidth = 2.5;
              ctx.lineCap = "round";
              ctx.lineJoin = "round";
              ctx.beginPath();

              const checkStartX = ringX - radius * 0.42;
              const checkStartY = ringY + radius * 0.05;
              const checkMidX = ringX - radius * 0.1;
              const checkMidY = ringY + radius * 0.38;
              const checkEndX = ringX + radius * 0.45;
              const checkEndY = ringY - radius * 0.35;

              ctx.moveTo(checkStartX, checkStartY);
              if (checkProgress < 0.5) {
                const subP = checkProgress / 0.5;
                ctx.lineTo(checkStartX + (checkMidX - checkStartX) * subP, checkStartY + (checkMidY - checkStartY) * subP);
              } else {
                ctx.lineTo(checkMidX, checkMidY);
                const subP = (checkProgress - 0.5) / 0.5;
                ctx.lineTo(checkMidX + (checkEndX - checkMidX) * subP, checkMidY + (checkEndY - checkMidY) * subP);
              }
              ctx.stroke();
            }
          }
        }

        ctx.restore();

      // ==============================================================
      // STAGE 4: VERIFY VISIBLE HOLD (14.4s – 16.4s)
      // ==============================================================
      } else if (t >= T_ANA_TO_VER && t < T_VER_HOLD) {
        ctx.save();
        ctx.font = `900 ${baseFontSize * 1.05}px ${FONT_FAMILY}`;
        ctx.fillStyle = "#00f0ff";
        ctx.globalAlpha = 1.0;
        ctx.shadowColor = "rgba(0, 240, 255, 0.75)";
        ctx.shadowBlur = 28;
        ctx.fillText("VERIFY", 4, startY);

        const verifyW = ctx.measureText("VERIFY").width;
        const ringX = 4 + verifyW + (isMobile ? 26 : 38);
        const ringY = startY;
        const radius = isMobile ? 18 : 24;

        // Fully drawn cyan verification ring
        ctx.strokeStyle = "#00f0ff";
        ctx.lineWidth = 2.5;
        ctx.shadowColor = "#00f0ff";
        ctx.shadowBlur = 16;
        ctx.beginPath();
        ctx.arc(ringX, ringY, radius, 0, Math.PI * 2);
        ctx.stroke();

        // White checkmark
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 2.5;
        ctx.lineCap = "round";
        ctx.lineJoin = "round";
        ctx.beginPath();
        ctx.moveTo(ringX - radius * 0.42, ringY + radius * 0.05);
        ctx.lineTo(ringX - radius * 0.1, ringY + radius * 0.38);
        ctx.lineTo(ringX + radius * 0.45, ringY - radius * 0.35);
        ctx.stroke();

        ctx.restore();

      // ==============================================================
      // TRANSITION 4: VERIFY -> NEXSPACE SLOW DELIBERATE MORPH (16.4s – 18.4s)
      // ==============================================================
      } else {
        const p = easeInOutCubic((t - T_VER_HOLD) / 2.0);
        ctx.save();

        // 1. VERIFY letters expand tracking & dissolve into cyan data strokes
        const verifyAlpha = Math.max(0, 1 - p * 1.25);
        if (verifyAlpha > 0) {
          ctx.font = `900 ${baseFontSize * 1.05}px ${FONT_FAMILY}`;
          ctx.fillStyle = "#00f0ff";
          ctx.globalAlpha = verifyAlpha;
          ctx.shadowColor = "#00f0ff";
          ctx.shadowBlur = 20 * (1 - p);
          ctx.fillText("VERIFY", 4, startY);

          // Verification ring expands outward and dissolves
          const verifyW = ctx.measureText("VERIFY").width;
          const ringX = 4 + verifyW + (isMobile ? 26 : 38);
          const ringY = startY;
          const radius = (isMobile ? 18 : 24) * (1 + p * 0.5);

          ctx.strokeStyle = "#00f0ff";
          ctx.lineWidth = 2.5 * (1 - p);
          ctx.beginPath();
          ctx.arc(ringX, ringY, radius, 0, Math.PI * 2);
          ctx.stroke();
        }

        // 2. Connecting data strokes bridging VERIFY into NEXSPACE
        const bridgeAlpha = Math.sin(p * Math.PI) * 0.75;
        if (bridgeAlpha > 0.05) {
          ctx.strokeStyle = "#00f0ff";
          ctx.lineWidth = 1.5;
          ctx.globalAlpha = bridgeAlpha;
          ctx.shadowColor = "#00f0ff";
          ctx.shadowBlur = 10;

          for (let k = 0; k < 4; k++) {
            const lineY = startY - baseFontSize * 0.2 + k * (baseFontSize * 0.15);
            ctx.beginPath();
            ctx.moveTo(4, lineY);
            ctx.lineTo(4 + (width * 0.5) * p, lineY);
            ctx.stroke();
          }
        }

        // 3. NEXSPACE emerging with tightening letter-spacing
        const nexAlpha = smoothstep(0.25, 0.95, p);
        if (nexAlpha > 0) {
          ctx.font = `900 ${baseFontSize}px ${FONT_FAMILY}`;
          ctx.fillStyle = "#ffffff";
          ctx.globalAlpha = nexAlpha;
          ctx.shadowColor = "rgba(0, 240, 255, 0.4)";
          ctx.shadowBlur = 22 * nexAlpha;

          const letterSpacing = (0.12 - 0.06 * p) * baseFontSize;
          const text = "NEXSPACE";
          let curX = 4;
          for (let i = 0; i < text.length; i++) {
            const char = text[i];
            ctx.fillText(char, curX, startY);
            curX += ctx.measureText(char).width + letterSpacing;
          }
        }

        ctx.restore();
      }

      // Loop forever (infinite requestAnimationFrame)
      animFrameRef.current = requestAnimationFrame(render);
    };

    animFrameRef.current = requestAnimationFrame(render);

    return () => {
      isRunning = false;
      if (animFrameRef.current) {
        cancelAnimationFrame(animFrameRef.current);
      }
      window.removeEventListener("resize", handleResize);
    };
  }, [forcedTime, isReducedMotion]);

  return (
    <div
      ref={containerRef}
      className="relative w-full h-[105px] sm:h-[125px] lg:h-[145px] flex items-center select-none overflow-visible"
      aria-live="polite"
      aria-label="NexSpace: Ask, Analyze, Verify"
    >
      <canvas
        ref={canvasRef}
        className="w-full h-full block pointer-events-none"
      />
    </div>
  );
}
