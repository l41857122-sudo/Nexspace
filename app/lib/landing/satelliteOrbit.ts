/**
 * lib/landing/satelliteOrbit.ts
 * Computes physically believable curved 3D orbital trajectory, banking angles,
 * and depth scaling for the satellite across the scroll progression.
 * Synchronized with the visible orbital SVG arc.
 */

export interface SatelliteOrbitState {
  screenX: number; // percentage (0 to 100)
  screenY: number; // percentage (0 to 100)
  rotationDeg: number; // banking/tilt in degrees
  scale: number; // 3D depth perspective factor
  solarGlazeIntensity: number;
}

function cubicBezier(t: number, p0: number, p1: number, p2: number, p3: number): number {
  const oneMinusT = 1 - t;
  return (
    Math.pow(oneMinusT, 3) * p0 +
    3 * Math.pow(oneMinusT, 2) * t * p1 +
    3 * oneMinusT * Math.pow(t, 2) * p2 +
    Math.pow(t, 3) * p3
  );
}

function cubicBezierDerivative(t: number, p0: number, p1: number, p2: number, p3: number): number {
  const oneMinusT = 1 - t;
  return (
    3 * Math.pow(oneMinusT, 2) * (p1 - p0) +
    6 * oneMinusT * t * (p2 - p1) +
    3 * Math.pow(t, 2) * (p3 - p2)
  );
}

/**
 * Calculates the satellite's 3D position and orientation along its orbit
 * based on normalized master scroll progress t in [0, 1].
 */
export function getSatelliteOrbitState(t: number): SatelliteOrbitState {
  const clampedT = Math.min(1, Math.max(0, t));

  // The orbital SVG path is: M 450,720 C 650,560 840,360 980,160 (1200 x 800)
  // At hero rest (t = 0), satellite sits in foreground right of Earth at u = 0.68 (~69% X, ~44% Y)
  // As user scrolls, it travels smoothly along the orbit to u = 0.96 (~80% X, ~23% Y)
  const u = 0.68 + clampedT * 0.28;

  const rawX = cubicBezier(u, 450, 650, 840, 980);
  const rawY = cubicBezier(u, 720, 560, 360, 160);

  const screenX = (rawX / 1200) * 100;
  const screenY = (rawY / 800) * 100;

  // Tangent angle along the trajectory curve for realistic banking
  const dx = cubicBezierDerivative(u, 450, 650, 840, 980);
  const dy = cubicBezierDerivative(u, 720, 560, 360, 160);
  const tangentDeg = (Math.atan2(dy, dx) * 180) / Math.PI; // around -51 to -54 deg

  // Adjust base angle to align satellite body toward flight direction with sensor tilted Earthward
  const rotationDeg = -14 + (tangentDeg + 51.5) * 0.8;

  // Depth scaling: satellite gently scales from 1.08 to 0.90 as it ascends along orbit
  const scale = 1.08 - clampedT * 0.18;

  // Solar panel gleam peaks at t ~ 0.35 when passing solar reflection angle
  const solarAngleDelta = Math.abs(clampedT - 0.35);
  const solarGlazeIntensity = Math.max(0, 1 - solarAngleDelta * 5);

  return {
    screenX,
    screenY,
    rotationDeg,
    scale,
    solarGlazeIntensity,
  };
}
