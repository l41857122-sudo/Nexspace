"use client";

import React, { useEffect, useRef } from "react";
import { Renderer, Program, Mesh, Triangle } from "ogl";
import "./WebThreads.css";

export type FanMode = "center" | "left" | "right";

export interface WebThreadsProps {
  color1?: string;
  color2?: string;
  color3?: string;
  speed?: number;
  threadCount?: number;
  frequency?: number;
  spread?: number;
  taper?: number;
  position?: number;
  fanMode?: FanMode;
  glow?: number;
  falloff?: number;
  thickness?: number;
  brightness?: number;
  opacity?: number;
  mirror?: boolean;
  shimmer?: boolean;
  grain?: boolean;
  grainIntensity?: number;
  mouseInteraction?: boolean;
  mouseStrength?: number;
  className?: string;
}

const hexToRgb = (hex: string): [number, number, number] => {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  if (!result) return [1, 1, 1];
  return [
    parseInt(result[1], 16) / 255,
    parseInt(result[2], 16) / 255,
    parseInt(result[3], 16) / 255,
  ];
};

const FAN_MODE: Record<FanMode, number> = { center: 0, left: 1, right: 2 };

const vertex300 = `#version 300 es
in vec2 position;
void main() {
  gl_Position = vec4(position, 0.0, 1.0);
}
`;

const fragment300 = `#version 300 es
precision highp float;
uniform vec2 iResolution;
uniform float iTime;
uniform float uSpeed;
uniform float uThreadCount;
uniform float uFrequency;
uniform float uSpread;
uniform float uTaper;
uniform float uPosition;
uniform float uFanMode;
uniform float uGlow;
uniform float uFalloff;
uniform float uThickness;
uniform float uBrightness;
uniform float uOpacity;
uniform float uMirror;
uniform float uShimmer;
uniform float uGrain;
uniform float uGrainIntensity;
uniform vec3 uColor1;
uniform vec3 uColor2;
uniform vec3 uColor3;
uniform vec2 uMouse;
uniform float uMouseStrength;
uniform float uEnableMouse;
uniform float uMouseActive;
out vec4 fragColor;

#define TAU 6.28318530718
#define MAX_THREADS 10

float glow(float x, float str, float dist) {
  return dist / pow(max(x, 1e-4), str);
}

void main() {
  vec2 uv = gl_FragCoord.xy / iResolution.xy;
  float n = max(uThreadCount, 1.0);

  float pinchX = uFanMode < 0.5 ? 0.5 : (uFanMode < 1.5 ? 0.0 : 1.0);
  if (uEnableMouse > 0.5) {
    pinchX = mix(pinchX, uMouse.x, clamp(uMouseStrength, 0.0, 1.0) * uMouseActive);
  }

  // Flank-weighted orbital envelope (subtle in center, prominent in wings)
  float sideWeight = 0.55 + 0.45 * pow(abs(uv.x - 0.5) * 2.0, 1.2);
  float spreadDx = uSpread * 1.25 * sideWeight;
  float baseT = iTime * uSpeed;
  float tauOverN = TAU / n;
  float mirror = uMirror > 0.5 ? sign(pinchX - uv.x) : 1.0;
  bool doShimmer = uShimmer > 0.5;
  float shimmerT = iTime * 1.6;
  float invThickness = 1.0 / max(uThickness * 0.016, 0.004);
  float xFreq = uv.x * uFrequency * 2.0;
  float yOff = uv.y - uPosition;
  float ciScale = n > 1.0 ? 1.0 / (n - 1.0) : 0.0;

  // Center typography clearing attenuation
  float centerDist = length(vec2((uv.x - 0.5) * 1.5, (uv.y - 0.5) * 2.3));
  float centerDamp = mix(0.35, 1.0, smoothstep(0.18, 0.65, centerDist));

  vec3 col = vec3(0.0);
  float gsum = 0.0;

  for (int idx = 0; idx < MAX_THREADS; idx++) {
    float i = float(idx);
    if (i >= n) break;

    // Organic orbital harmonic wave (varying inclination angles)
    float inclination = (i - (n - 1.0) * 0.5) * 0.08;
    float amplitude = spreadDx * (0.8 + i * uTaper * 0.22);
    float shimmer = doShimmer ? sin(shimmerT + i * 1.3) * 0.12 : 0.0;
    float phase = (baseT + i * tauOverN) * mirror + shimmer;

    // Multi-harmonic celestial trajectory: primary arc + resonant orbital companion
    float wave = sin(xFreq * 0.85 + phase) * 0.75 + sin(xFreq * 1.6 + phase * 1.1 + i * 0.5) * 0.25;
    float trajectoryY = wave * amplitude + inclination * (uv.x - 0.5);

    float sdf = abs(yOff + trajectoryY) * invThickness;

    float g = glow(sdf, uFalloff, uGlow);
    float ci = i * ciScale;
    vec3 threadCol = mix(uColor1, uColor2, ci);

    col += g * threadCol;
    gsum += g;
  }

  // Glowing cyan/white node illumination at orbital intersections
  float coreAmt = smoothstep(0.28, 1.5, gsum);
  col = mix(col, uColor3 * gsum * 1.35, coreAmt * 0.6);

  float bright = uBrightness * 1.35;
  if (uEnableMouse > 0.5) {
    vec2 md = uv - uMouse;
    float d2 = dot(md, md);
    bright += clamp(uMouseStrength, 0.0, 1.0) * uMouseActive * exp(-d2 * 8.0) * 0.6;
  }
  col *= bright * centerDamp;

  // Natural edge fade at viewport boundaries
  float edgeFade = smoothstep(0.0, 0.07, uv.x) * smoothstep(1.0, 0.93, uv.x);
  float alpha = clamp(gsum * edgeFade * centerDamp * 1.1, 0.0, 1.0) * uOpacity;

  vec3 outRgb = col * edgeFade;

  if (uGrain > 0.5) {
    float gv = (fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233)) + iTime) * 43758.5453) - 0.5) * uGrainIntensity;
    outRgb = clamp(outRgb + gv, 0.0, 1.0);
    alpha = clamp(alpha + gv, 0.0, 1.0);
  }

  fragColor = vec4(outRgb, alpha);
}
`;

const vertex100 = `
attribute vec2 position;
void main() {
  gl_Position = vec4(position, 0.0, 1.0);
}
`;

const fragment100 = `
precision highp float;
uniform vec2 iResolution;
uniform float iTime;
uniform float uSpeed;
uniform float uThreadCount;
uniform float uFrequency;
uniform float uSpread;
uniform float uTaper;
uniform float uPosition;
uniform float uFanMode;
uniform float uGlow;
uniform float uFalloff;
uniform float uThickness;
uniform float uBrightness;
uniform float uOpacity;
uniform float uMirror;
uniform float uShimmer;
uniform float uGrain;
uniform float uGrainIntensity;
uniform vec3 uColor1;
uniform vec3 uColor2;
uniform vec3 uColor3;
uniform vec2 uMouse;
uniform float uMouseStrength;
uniform float uEnableMouse;
uniform float uMouseActive;

#define TAU 6.28318530718
#define MAX_THREADS 10

float glow(float x, float str, float dist) {
  return dist / pow(max(x, 1e-4), str);
}

void main() {
  vec2 uv = gl_FragCoord.xy / iResolution.xy;
  float n = max(uThreadCount, 1.0);

  float pinchX = uFanMode < 0.5 ? 0.5 : (uFanMode < 1.5 ? 0.0 : 1.0);
  if (uEnableMouse > 0.5) {
    pinchX = mix(pinchX, uMouse.x, clamp(uMouseStrength, 0.0, 1.0) * uMouseActive);
  }

  float sideWeight = 0.55 + 0.45 * pow(abs(uv.x - 0.5) * 2.0, 1.2);
  float spreadDx = uSpread * 1.25 * sideWeight;
  float baseT = iTime * uSpeed;
  float tauOverN = TAU / n;
  float mirror = uMirror > 0.5 ? sign(pinchX - uv.x) : 1.0;
  bool doShimmer = uShimmer > 0.5;
  float shimmerT = iTime * 1.6;
  float invThickness = 1.0 / max(uThickness * 0.016, 0.004);
  float xFreq = uv.x * uFrequency * 2.0;
  float yOff = uv.y - uPosition;
  float ciScale = n > 1.0 ? 1.0 / (n - 1.0) : 0.0;

  float centerDist = length(vec2((uv.x - 0.5) * 1.5, (uv.y - 0.5) * 2.3));
  float centerDamp = mix(0.35, 1.0, smoothstep(0.18, 0.65, centerDist));

  vec3 col = vec3(0.0);
  float gsum = 0.0;

  for (int idx = 0; idx < MAX_THREADS; idx++) {
    float i = float(idx);
    if (i >= n) break;

    float inclination = (i - (n - 1.0) * 0.5) * 0.08;
    float amplitude = spreadDx * (0.8 + i * uTaper * 0.22);
    float shimmer = doShimmer ? sin(shimmerT + i * 1.3) * 0.12 : 0.0;
    float phase = (baseT + i * tauOverN) * mirror + shimmer;

    float wave = sin(xFreq * 0.85 + phase) * 0.75 + sin(xFreq * 1.6 + phase * 1.1 + i * 0.5) * 0.25;
    float trajectoryY = wave * amplitude + inclination * (uv.x - 0.5);

    float sdf = abs(yOff + trajectoryY) * invThickness;

    float g = glow(sdf, uFalloff, uGlow);
    float ci = i * ciScale;
    vec3 threadCol = mix(uColor1, uColor2, ci);

    col += g * threadCol;
    gsum += g;
  }

  float coreAmt = smoothstep(0.28, 1.5, gsum);
  col = mix(col, uColor3 * gsum * 1.35, coreAmt * 0.6);

  float bright = uBrightness * 1.35;
  if (uEnableMouse > 0.5) {
    vec2 md = uv - uMouse;
    float d2 = dot(md, md);
    bright += clamp(uMouseStrength, 0.0, 1.0) * uMouseActive * exp(-d2 * 8.0) * 0.6;
  }
  col *= bright * centerDamp;

  float edgeFade = smoothstep(0.0, 0.07, uv.x) * smoothstep(1.0, 0.93, uv.x);
  float alpha = clamp(gsum * edgeFade * centerDamp * 1.1, 0.0, 1.0) * uOpacity;

  vec3 outRgb = col * edgeFade;

  if (uGrain > 0.5) {
    float gv = (fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233)) + iTime) * 43758.5453) - 0.5) * uGrainIntensity;
    outRgb = clamp(outRgb + gv, 0.0, 1.0);
    alpha = clamp(alpha + gv, 0.0, 1.0);
  }

  gl_FragColor = vec4(outRgb, alpha);
}
`;

type WebThreadsCtx = {
  renderer: InstanceType<typeof Renderer>;
  program: InstanceType<typeof Program>;
  mesh: InstanceType<typeof Mesh>;
};
const ctxMap = new WeakMap<HTMLDivElement, WebThreadsCtx>();

const WebThreads: React.FC<WebThreadsProps> = ({
  color1 = "#5227FF",
  color2 = "#1339eb",
  color3 = "#06B6D4",
  speed = 0.16,
  threadCount = 5,
  frequency = 3.2,
  spread = 0.14,
  taper = 1.0,
  position = 0.5,
  fanMode = "center",
  glow = 0.032,
  falloff = 0.58,
  thickness = 0.95,
  brightness = 0.65,
  opacity = 0.72,
  mirror = true,
  shimmer = false,
  grain = true,
  grainIntensity = 0.04,
  mouseInteraction = true,
  mouseStrength = 0.18,
  className = "",
}) => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const mouseRef = useRef<{ enabled: boolean; strength: number }>({
    enabled: mouseInteraction,
    strength: mouseStrength,
  });

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    let renderer: any = null;
    let gl: any = null;
    let canvas: HTMLCanvasElement | null = null;
    let is2dFallback = false;
    let fallbackRaf = 0;
    const dpr = Math.min(typeof window !== "undefined" ? window.devicePixelRatio || 1 : 1, 2);

    try {
      renderer = new Renderer({
        webgl: 2,
        alpha: true,
        premultipliedAlpha: true,
        antialias: false,
        dpr,
      });
      gl = renderer.gl;
      if (!gl) {
        renderer = new Renderer({
          webgl: 1,
          alpha: true,
          premultipliedAlpha: true,
          antialias: false,
          dpr,
        });
        gl = renderer.gl;
      }
    } catch {
      gl = null;
    }

    if (!gl) {
      // 2D Canvas fallback with organic satellite orbital trajectories
      is2dFallback = true;
      canvas = document.createElement("canvas");
      canvas.style.position = "absolute";
      canvas.style.inset = "0";
      canvas.style.width = "100%";
      canvas.style.height = "100%";
      canvas.style.display = "block";
      container.appendChild(canvas);

      const ctx2d = canvas.getContext("2d");
      if (!ctx2d) return;

      let w = container.clientWidth || window.innerWidth;
      let h = container.clientHeight || window.innerHeight;
      canvas.width = Math.floor(w * dpr);
      canvas.height = Math.floor(h * dpr);

      const resize2d = () => {
        const rect = container.getBoundingClientRect();
        w = Math.max(1, Math.floor(rect.width));
        h = Math.max(1, Math.floor(rect.height));
        if (canvas) {
          canvas.width = Math.floor(w * dpr);
          canvas.height = Math.floor(h * dpr);
        }
      };
      const ro2d = new ResizeObserver(resize2d);
      ro2d.observe(container);

      let t0 = performance.now();
      const loop2d = (t: number) => {
        const time = (t - t0) * 0.001 * speed * 1.5;
        ctx2d.clearRect(0, 0, canvas!.width, canvas!.height);
        ctx2d.save();
        ctx2d.scale(dpr, dpr);

        const n = Math.max(threadCount, 1);
        const yCenter = h * position;

        for (let i = 0; i < n; i++) {
          const ratio = i / Math.max(n - 1, 1);
          const rgb1 = hexToRgb(color1);
          const rgb2 = hexToRgb(color2);
          const r = Math.round((rgb1[0] * (1 - ratio) + rgb2[0] * ratio) * 255);
          const g = Math.round((rgb1[1] * (1 - ratio) + rgb2[1] * ratio) * 255);
          const b = Math.round((rgb1[2] * (1 - ratio) + rgb2[2] * ratio) * 255);

          // Center-softened, edge-vibrant linear gradient
          const grad = ctx2d.createLinearGradient(0, 0, w, 0);
          grad.addColorStop(0, `rgba(${r}, ${g}, ${b}, 0)`);
          grad.addColorStop(0.12, `rgba(${r}, ${g}, ${b}, ${0.75 * opacity * brightness})`);
          grad.addColorStop(0.35, `rgba(${r}, ${g}, ${b}, ${0.50 * opacity * brightness})`);
          grad.addColorStop(0.5, `rgba(${r}, ${g}, ${b}, ${0.38 * opacity * brightness})`);
          grad.addColorStop(0.65, `rgba(${r}, ${g}, ${b}, ${0.50 * opacity * brightness})`);
          grad.addColorStop(0.88, `rgba(${r}, ${g}, ${b}, ${0.75 * opacity * brightness})`);
          grad.addColorStop(1, `rgba(${r}, ${g}, ${b}, 0)`);

          ctx2d.beginPath();
          ctx2d.lineWidth = thickness * (1.5 + (1 - ratio) * 0.7);
          ctx2d.strokeStyle = grad;
          ctx2d.shadowColor = color3;
          ctx2d.shadowBlur = glow * 900 * 0.6;

          const steps = 140;
          const inclination = (i - (n - 1) * 0.5) * 0.06 * h;

          for (let step = 0; step <= steps; step++) {
            const xNorm = step / steps;
            const x = xNorm * w;
            const sideWeight = 0.55 + 0.45 * Math.pow(Math.abs(xNorm - 0.5) * 2.0, 1.2);
            const spreadDx = spread * h * 1.15 * sideWeight * (0.85 + i * taper * 0.2);
            const phase = time + (i * Math.PI * 2) / n;

            // Multi-harmonic orbital arc
            const wave = Math.sin(xNorm * frequency * Math.PI * 0.85 + phase) * 0.75 + Math.sin(xNorm * frequency * Math.PI * 1.6 + phase * 1.1 + i * 0.5) * 0.25;
            const y = yCenter + wave * spreadDx + inclination * (xNorm - 0.5);

            if (step === 0) ctx2d.moveTo(x, y);
            else ctx2d.lineTo(x, y);
          }
          ctx2d.stroke();
        }

        ctx2d.restore();
        fallbackRaf = requestAnimationFrame(loop2d);
      };
      fallbackRaf = requestAnimationFrame(loop2d);

      return () => {
        cancelAnimationFrame(fallbackRaf);
        ro2d.disconnect();
        if (canvas && canvas.parentNode === container) {
          container.removeChild(canvas);
        }
      };
    }

    // Hardware Accelerated WebGL Path
    gl.clearColor(0, 0, 0, 0);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

    canvas = gl.canvas as HTMLCanvasElement;
    canvas.style.position = "absolute";
    canvas.style.inset = "0";
    canvas.style.width = "100%";
    canvas.style.height = "100%";
    canvas.style.display = "block";
    container.appendChild(canvas);

    const isWebGL2 = typeof WebGL2RenderingContext !== "undefined" && gl instanceof WebGL2RenderingContext;
    const vertex = isWebGL2 ? vertex300 : vertex100;
    const fragment = isWebGL2 ? fragment300 : fragment100;

    const geometry = new Triangle(gl);
    const program = new Program(gl, {
      vertex,
      fragment,
      uniforms: {
        iTime: { value: 0 },
        iResolution: { value: new Float32Array([1, 1]) },
        uSpeed: { value: speed },
        uThreadCount: { value: threadCount },
        uFrequency: { value: frequency },
        uSpread: { value: spread },
        uTaper: { value: taper },
        uPosition: { value: position },
        uFanMode: { value: FAN_MODE[fanMode] ?? 0 },
        uGlow: { value: glow },
        uFalloff: { value: falloff },
        uThickness: { value: thickness },
        uBrightness: { value: brightness },
        uOpacity: { value: opacity },
        uMirror: { value: mirror ? 1.0 : 0.0 },
        uShimmer: { value: shimmer ? 1.0 : 0.0 },
        uGrain: { value: grain ? 1.0 : 0.0 },
        uGrainIntensity: { value: grainIntensity },
        uColor1: { value: new Float32Array(hexToRgb(color1)) },
        uColor2: { value: new Float32Array(hexToRgb(color2)) },
        uColor3: { value: new Float32Array(hexToRgb(color3)) },
        uMouse: { value: new Float32Array([0.5, 0.5]) },
        uMouseStrength: { value: mouseStrength },
        uEnableMouse: { value: mouseInteraction ? 1.0 : 0.0 },
        uMouseActive: { value: 0 },
      },
    });

    const mesh = new Mesh(gl, { geometry, program });
    ctxMap.set(container, { renderer, program, mesh });

    const setSize = () => {
      const rect = container.getBoundingClientRect();
      const w = Math.max(1, Math.floor(rect.width));
      const h = Math.max(1, Math.floor(rect.height));
      renderer.setSize(w, h);
      const res = (program.uniforms.iResolution as { value: Float32Array }).value;
      res[0] = gl.drawingBufferWidth;
      res[1] = gl.drawingBufferHeight;
      renderer.render({ scene: mesh });
    };

    const ro = new ResizeObserver(setSize);
    ro.observe(container);
    setSize();

    const currentMouse: [number, number] = [0.5, 0.5];
    const targetMouse: [number, number] = [0.5, 0.5];
    let currentActive = 0;
    let targetActive = 0;

    const onMouseMove = (e: MouseEvent) => {
      const rect = container.getBoundingClientRect();
      if (rect.width > 0 && rect.height > 0) {
        targetMouse[0] = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
        targetMouse[1] = Math.max(0, Math.min(1, 1.0 - (e.clientY - rect.top) / rect.height));
        targetActive = 1;
      }
    };
    const onMouseEnter = () => {
      targetActive = 1;
    };
    const onMouseLeave = () => {
      targetActive = 0;
    };
    window.addEventListener("mousemove", onMouseMove);
    container.addEventListener("mouseenter", onMouseEnter);
    container.addEventListener("mouseleave", onMouseLeave);

    let raf = 0;
    let isVisible = true;
    let isPageVisible = !document.hidden;
    const t0 = performance.now();

    const loop = (t: number) => {
      (program.uniforms.iTime as { value: number }).value = (t - t0) * 0.001;
      currentMouse[0] += 0.05 * (targetMouse[0] - currentMouse[0]);
      currentMouse[1] += 0.05 * (targetMouse[1] - currentMouse[1]);
      currentActive += 0.05 * (targetActive - currentActive);
      const mouse = (program.uniforms.uMouse as { value: Float32Array }).value;
      mouse[0] = currentMouse[0];
      mouse[1] = currentMouse[1];
      (program.uniforms.uMouseActive as { value: number }).value = currentActive;
      (program.uniforms.uEnableMouse as { value: number }).value = mouseRef.current.enabled
        ? 1.0
        : 0.0;
      (program.uniforms.uMouseStrength as { value: number }).value = mouseRef.current.strength;
      renderer.render({ scene: mesh });
      raf = requestAnimationFrame(loop);
    };

    const tryStart = () => {
      if (isVisible && isPageVisible && raf === 0) raf = requestAnimationFrame(loop);
    };
    const tryStop = () => {
      if (raf !== 0) {
        cancelAnimationFrame(raf);
        raf = 0;
      }
    };

    const io = new IntersectionObserver(
      ([entry]) => {
        isVisible = entry.isIntersecting;
        isVisible ? tryStart() : tryStop();
      },
      { threshold: 0 }
    );
    io.observe(container);

    const onVisibility = () => {
      isPageVisible = !document.hidden;
      isPageVisible ? tryStart() : tryStop();
    };
    document.addEventListener("visibilitychange", onVisibility);

    tryStart();

    return () => {
      tryStop();
      ro.disconnect();
      io.disconnect();
      document.removeEventListener("visibilitychange", onVisibility);
      window.removeEventListener("mousemove", onMouseMove);
      container.removeEventListener("mouseenter", onMouseEnter);
      container.removeEventListener("mouseleave", onMouseLeave);
      ctxMap.delete(container);
      try {
        if (canvas && canvas.parentNode === container) {
          container.removeChild(canvas);
        }
      } catch {}
      gl.getExtension("WEBGL_lose_context")?.loseContext();
    };
  }, []);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const ctx = ctxMap.get(container);
    if (!ctx) return;
    const { program } = ctx;
    const u = program.uniforms as Record<string, { value: any }>;

    u.uSpeed.value = speed;
    u.uThreadCount.value = Math.round(threadCount);
    u.uFrequency.value = frequency;
    u.uSpread.value = spread;
    u.uTaper.value = taper;
    u.uPosition.value = position;
    u.uFanMode.value = FAN_MODE[fanMode] ?? 0;
    u.uGlow.value = glow;
    u.uFalloff.value = falloff;
    u.uThickness.value = thickness;
    u.uBrightness.value = brightness;
    u.uOpacity.value = opacity;
    u.uMirror.value = mirror ? 1.0 : 0.0;
    u.uShimmer.value = shimmer ? 1.0 : 0.0;
    u.uGrain.value = grain ? 1.0 : 0.0;
    u.uGrainIntensity.value = grainIntensity;
    const c1 = u.uColor1.value as Float32Array;
    const rgb1 = hexToRgb(color1);
    c1[0] = rgb1[0];
    c1[1] = rgb1[1];
    c1[2] = rgb1[2];
    const c2 = u.uColor2.value as Float32Array;
    const rgb2 = hexToRgb(color2);
    c2[0] = rgb2[0];
    c2[1] = rgb2[1];
    c2[2] = rgb2[2];
    const c3 = u.uColor3.value as Float32Array;
    const rgb3 = hexToRgb(color3);
    c3[0] = rgb3[0];
    c3[1] = rgb3[1];
    c3[2] = rgb3[2];
    u.uMouseStrength.value = mouseStrength;
    u.uEnableMouse.value = mouseInteraction ? 1.0 : 0.0;
    mouseRef.current.enabled = mouseInteraction;
    mouseRef.current.strength = mouseStrength;
  }, [
    color1,
    color2,
    color3,
    speed,
    threadCount,
    frequency,
    spread,
    taper,
    position,
    fanMode,
    glow,
    falloff,
    thickness,
    brightness,
    opacity,
    mirror,
    shimmer,
    grain,
    grainIntensity,
    mouseInteraction,
    mouseStrength,
  ]);

  return <div ref={containerRef} className={`web-threads-container ${className}`.trim()} />;
};

export default WebThreads;
