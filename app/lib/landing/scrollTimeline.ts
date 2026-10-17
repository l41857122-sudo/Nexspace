/**
 * lib/landing/scrollTimeline.ts
 * Manages section IDs, navigation items, and scroll coordinates for the landing journey.
 */

export interface NavSection {
  id: string;
  num: string;
  label: string;
  title: string;
  minPct: number;
  maxPct: number;
}

export const LANDING_SECTIONS: NavSection[] = [
  { id: "hero", num: "01", label: "HERO", title: "Autonomous Spatial AI", minPct: 0.00, maxPct: 0.12 },
  { id: "problem", num: "02", label: "PROBLEM", title: "The Problem", minPct: 0.12, maxPct: 0.24 },
  { id: "investigation", num: "03", label: "SCALES", title: "Live Investigation", minPct: 0.24, maxPct: 0.36 },
  { id: "pixels", num: "04", label: "PIXELS", title: "Pixels to Intelligence", minPct: 0.36, maxPct: 0.48 },
  { id: "pipeline", num: "05", label: "PIPELINE", title: "Investigation Pipeline", minPct: 0.48, maxPct: 0.60 },
  { id: "realworld", num: "06", label: "SHOWCASE", title: "Real-World Intelligence", minPct: 0.60, maxPct: 0.72 },
  { id: "evidence", num: "07", label: "EVIDENCE", title: "Evidence Verification", minPct: 0.72, maxPct: 0.84 },
  { id: "about", num: "08", label: "ABOUT", title: "People Behind NexSpace", minPct: 0.84, maxPct: 0.94 },
  { id: "brand", num: "09", label: "NEXSPACE", title: "Verified Intelligence", minPct: 0.94, maxPct: 1.00 },
];
