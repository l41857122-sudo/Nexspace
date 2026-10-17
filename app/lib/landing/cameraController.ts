/**
 * lib/landing/cameraController.ts
 * Master single source of truth for the cinematic hero scroll timeline.
 * Ensures continuous transitions with ZERO black gaps or orphaned elements.
 */

import { lerp, smoothstep, clamp } from "./animationUtils";

export interface CameraTimelineState {
  progress: number;
  // Camera & Earth transforms
  camZ: number;
  camX: number;
  camY: number;
  earthScale: number;
  earthAlpha: number;
  // Surface imagery transition
  imageryAlpha: number;
  imageryScale: number;
  gridAlpha: number;
  scanBeamProgress: number;
  analysisReadyAlpha: number;
  // Hero text & HUD opacity
  heroTextAlpha: number;
  hudAlpha: number;
  // Satellite trajectory state
  satelliteOrbitProgress: number;
  satelliteAlpha: number;
  // AI Scanning activation
  scanProgress: number;
}

/**
 * Computes the unified cinematic state across the master scroll progression p in [0, 1].
 */
export function getCameraTimelineState(progress: number): CameraTimelineState {
  const p = clamp(progress, 0, 1);

  // 1. Hero Content & HUD Opacity:
  // 0 to 32%: fully legible
  // 32% to 52%: smooth graceful fade
  // 52%+: faded
  let heroTextAlpha = 1.0;
  if (p > 0.32) {
    heroTextAlpha = clamp(1.0 - (p - 0.32) / 0.20, 0, 1);
  }
  const hudAlpha = heroTextAlpha;

  // 2. Camera Physical Dive & Earth Scale Expansion:
  // Camera smoothly plunges toward Earth's surface as user scrolls past 32%
  const zoomT = smoothstep(0.32, 0.72, p);
  const camZ = lerp(4.6, 1.95, zoomT);
  const camX = lerp(0.0, 0.42, zoomT);
  const camY = lerp(0.0, 0.06, zoomT);

  // Earth scale expands smoothly (1.0 to 1.40) - curvature dominates viewport without sphere clipping
  const earthScale = lerp(1.0, 1.40, zoomT);

  // 3. Continuous Earth Alpha (Crossfade seamlessly into real imagery, ZERO black gap):
  // 0% to 60%: solid (1.0)
  // 60% to 78%: gentle cross-fade only WHILE real satellite imagery is dissolving in
  // 78%+: 0.0 (imagery is fully opaque)
  let earthAlpha = 1.0;
  if (p > 0.60) {
    earthAlpha = clamp(1.0 - (p - 0.60) / 0.18, 0, 1);
  }

  // 4. Real Satellite Imagery Reveal:
  // 0% to 56%: 0.0
  // 56% to 76%: smooth dissolve in over the expanding Earth surface
  // 76% to 100%: 1.0 (fully dominant)
  const imageryAlpha = smoothstep(0.56, 0.76, p);
  // Deliberate cinematic settling without excessive zoom crop (1.04 to 1.00)
  const imageryScale = lerp(1.04, 1.0, smoothstep(0.56, 0.85, p));

  // Optical coordinate grid fades in gradually
  const gridAlpha = smoothstep(0.68, 0.82, p);

  // Subtle scan beam sweeps across the imagery (0.0 to 1.0 between 72% and 88%)
  const scanBeamProgress = clamp((p - 0.72) / 0.16, 0, 1);

  // "IMAGERY ACQUIRED // ANALYSIS READY" status activates once scan sweeps
  const analysisReadyAlpha = smoothstep(0.86, 0.94, p);

  // 5. 3D Satellite Spacecraft:
  // Visible during orbital trajectory and camera pass, exits primary frame naturally
  const satelliteOrbitProgress = p;
  let satelliteAlpha = 1.0;
  if (p > 0.78) {
    satelliteAlpha = clamp(1.0 - (p - 0.78) / 0.14, 0, 1);
  }

  // 6. AI Scanning Transition:
  const scanProgress = smoothstep(0.88, 1.0, p);

  return {
    progress: p,
    camZ,
    camX,
    camY,
    earthScale,
    earthAlpha,
    imageryAlpha,
    imageryScale,
    gridAlpha,
    scanBeamProgress,
    analysisReadyAlpha,
    heroTextAlpha,
    hudAlpha,
    satelliteOrbitProgress,
    satelliteAlpha,
    scanProgress,
  };
}
