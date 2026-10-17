"use client";

import { useState, useEffect, useRef, useMemo, useCallback, memo } from "react";
import Link from "next/link";
import {
  Search,
  Sparkles,
  ArrowRight,
  ShieldCheck,
  Clock,
  Layers,
  Upload,
  Info,
  Target,
  RotateCcw,
  Building2,
  Ship,
  Waves,
  ScanLine,
  Flame,
  Leaf,
  MapPin,
  AlertTriangle,
  ChevronDown,
  ChevronUp,
  Image as ImageIcon,
  Check,
  Terminal,
  SlidersHorizontal,
  Eye,
  EyeOff,
  Maximize2,
  ZoomIn,
  ZoomOut,
  Compass,
  Filter,
  ArrowUpDown,
  ChevronLeft,
  ChevronRight,
  Globe,
  ExternalLink,
  FileText,
  CheckCircle2,
  HelpCircle,
  AlertCircle,
} from "lucide-react";
import Sidebar from "./Sidebar";
import GoogleMapView from "./GoogleMapView";
import SatelliteDiscoveryModal from "./SatelliteDiscoveryModal";
import type {
  NexSpaceQueryResponse,
  CanonicalSourceImage,
  CanonicalInvestigationState,
  DiscoveredSatelliteScene,
} from "../types/nexspace";
import {
  synthesizeInvestigationResult,
  InvestigationResult,
  EnrichedIntelligenceDetection,
} from "../types/intelligence";
import {
  SAMPLE_OPTICAL_PORT,
  SAMPLE_OPTICAL_URBAN,
  SAMPLE_SAR_RADAR,
  DEMO_IMAGE_CATALOG,
} from "../utils/sampleImages";
import {
  getActiveSourceImage,
  setActiveSourceImage,
  getCurrentInvestigation,
  setCurrentInvestigation,
  clearCurrentInvestigation,
  DEFAULT_DEMO_SOURCE,
} from "../utils/investigationStorage";

interface CapabilitiesRecord {
  captioning?: string;
  grounding?: string;
  vqa?: string;
  change_analysis?: string;
  anomaly_extraction?: string;
  optical_sar_fusion?: string;
  geospatial?: string;
  [key: string]: string | undefined;
}

// ----------------------------------------------------------------
// Helper: Confidence Semantics Mapping
// ----------------------------------------------------------------
function getConfidenceInfo(
  score: number | null | undefined,
  confType?: string | null
): {
  label: string;
  color: string;
  bg: string;
  border: string;
} {
  if (confType === "generation_failure" || confType === "invalid_generation") {
    return { label: "Unverified (Quality Filter Rejected)", color: "text-rose-400", bg: "bg-rose-500/15", border: "border-rose-500/30" };
  }
  if (confType === "unavailable") {
    return { label: "Unavailable", color: "text-slate-400", bg: "bg-slate-800/60", border: "border-slate-700/60" };
  }
  if (score === null || score === undefined || isNaN(score)) {
    return { label: "Uncalibrated", color: "text-slate-400", bg: "bg-slate-800/60", border: "border-slate-700/60" };
  }
  if (score < 0.40) {
    return { label: "Low", color: "text-amber-400", bg: "bg-amber-500/15", border: "border-amber-500/30" };
  }
  if (score < 0.70) {
    return { label: "Moderate", color: "text-cyan-300", bg: "bg-cyan-500/15", border: "border-cyan-500/30" };
  }
  if (score < 0.90) {
    return { label: "High", color: "text-emerald-400", bg: "bg-emerald-500/15", border: "border-emerald-500/30" };
  }
  return { label: "Very High", color: "text-emerald-300", bg: "bg-emerald-500/25", border: "border-emerald-500/50" };
}

// ----------------------------------------------------------------
// Helper: Robust Grounding Bounding Box Normalizer
// Always returns canonical [xmin, ymin, xmax, ymax] normalized 0..1000
// ----------------------------------------------------------------
export function normalizeBox(
  det: unknown,
  imgWidth = 512,
  imgHeight = 512
): [number, number, number, number] | null {
  if (!det || typeof det !== "object") return null;

  const d = det as Record<string, unknown>;
  const raw = d.bbox_normalized || d.box_2d || d.bbox_pixel || d.box || d.bbox;
  if (!Array.isArray(raw) || raw.length !== 4) return null;

  const numeric = raw.map((val) => {
    const n = Number(val);
    return typeof n === "number" && Number.isFinite(n) ? n : NaN;
  });

  if (numeric.some((n) => isNaN(n))) return null;

  // In computer vision (Grounding DINO, PaliGemma, YOLO, etc),
  // standard bbox arrays are [ymin, xmin, ymax, xmax].
  // If box or bbox_pixel/bbox_normalized is supplied:
  let ymin = numeric[0];
  let xmin = numeric[1];
  let ymax = numeric[2];
  let xmax = numeric[3];

  // Scale if normalized 0..1 float
  if (Math.max(xmin, ymin, xmax, ymax) <= 1.05) {
    xmin *= 1000;
    ymin *= 1000;
    xmax *= 1000;
    ymax *= 1000;
  } else if (!d.bbox_normalized && !d.box_2d && imgWidth > 0 && imgHeight > 0) {
    // Convert raw pixel dimensions to 0..1000
    xmin = (xmin / imgWidth) * 1000;
    ymin = (ymin / imgHeight) * 1000;
    xmax = (xmax / imgWidth) * 1000;
    ymax = (ymax / imgHeight) * 1000;
  }

  // Ensure coordinate bounds ordering
  if (xmin > xmax) [xmin, xmax] = [xmax, xmin];
  if (ymin > ymax) [ymin, ymax] = [ymax, ymin];

  xmin = Math.round(Math.max(0, Math.min(1000, xmin)));
  ymin = Math.round(Math.max(0, Math.min(1000, ymin)));
  xmax = Math.round(Math.max(0, Math.min(1000, xmax)));
  ymax = Math.round(Math.max(0, Math.min(1000, ymax)));

  // Must have non-zero dimension
  if (xmax <= xmin || ymax <= ymin) return null;

  return [xmin, ymin, xmax, ymax];
}

// ----------------------------------------------------------------
// Helper: Clean Canonical Entity Label Formatter
// ----------------------------------------------------------------
export function formatCanonicalLabel(rawLabel?: string, fallback = "Entity"): string {
  if (!rawLabel) return fallback;
  let clean = rawLabel.trim();

  // Strip prompt token delimiters if present (e.g. "ship . boat" or "building · rooftop")
  if (clean.includes(" . ") || clean.includes(" · ") || clean.includes(" , ") || clean.includes(",")) {
    const parts = clean.split(/[.·,]/).map((p) => p.trim().toLowerCase()).filter(Boolean);
    if (parts.length > 1) {
      if (parts.some((p) => p.includes("water") || p.includes("lake") || p.includes("river") || p.includes("sea") || p.includes("ocean"))) {
        return "Water Body";
      }
      if (parts.some((p) => p.includes("ship") || p.includes("boat") || p.includes("vessel"))) {
        return "Vessel";
      }
      if (parts.some((p) => p.includes("car") || p.includes("truck") || p.includes("vehicle"))) {
        return "Vehicle";
      }
      if (parts.some((p) => p.includes("building") || p.includes("structure") || p.includes("house") || p.includes("rooftop"))) {
        return "Building";
      }
    }
    clean = parts[0] || clean;
  }

  // Strip conversational wrappers and trailing conditionals
  clean = clean.replace(/^(is there|are there|can you see|do you see|tell me if there is|tell me if there are)\s+(a\s+|an\s+|any\s+)?/i, "").trim();
  clean = clean.replace(/\bif\s+(yes|so|present|visible|any|there is|there are)\b.*$/i, "").trim();
  clean = clean.replace(/\bthen\s+(locate|find|detect|pinpoint|show)\s+(them|it|all)\b.*$/i, "").trim();
  clean = clean.replace(/\b(locate|find|detect|pinpoint|show)\s+(it|them|all)\b.*$/i, "").trim();
  clean = clean.replace(/\b(in|on|across|within)\s+(the\s+|this\s+)?(image|scene|photo|picture|overhead|raster)\b.*$/i, "").trim();
  clean = clean.replace(/[?!.,:;]/g, "").trim();

  // Strip leading action verbs if present (e.g. "find ships and boats" -> "ships and boats")
  clean = clean.replace(/^(find|locate|detect|identify|show|extract)\s+(the\s+|all\s+)?/i, "").trim();
  clean = clean.toLowerCase();

  // Multi-target compound queries without single subtype classification
  if (
    clean.includes("ships and boats") ||
    clean.includes("boats and ships") ||
    clean.includes("ships or boats") ||
    clean.includes("boats or ships") ||
    clean.includes("ship and boat") ||
    clean.includes("boat and ship") ||
    clean.includes("ships & boats") ||
    clean.includes("boats & ships") ||
    (clean.includes("ship") && clean.includes("boat"))
  ) {
    return "Vessel";
  }

  if (
    clean.includes("cars and trucks") ||
    clean.includes("trucks and cars") ||
    clean.includes("cars or trucks") ||
    clean.includes("trucks or cars") ||
    clean.includes("vehicles and cars") ||
    (clean.includes("car") && clean.includes("truck"))
  ) {
    return "Vehicle";
  }

  if (
    clean.includes("buildings and houses") ||
    clean.includes("houses and buildings") ||
    clean.includes("buildings or houses") ||
    (clean.includes("building") && clean.includes("house"))
  ) {
    return "Building";
  }

  // Exact single-target or verified subtype mappings
  const singularMap: Record<string, string> = {
    "water body": "Water Body",
    "water bodies": "Water Body",
    "body of water": "Water Body",
    "bodies of water": "Water Body",
    water: "Water Body",
    ocean: "Ocean",
    sea: "Sea",
    lake: "Lake",
    lakes: "Lake",
    river: "River",
    rivers: "River",
    waterway: "Waterway",
    waterways: "Waterway",
    reservoir: "Reservoir",
    reservoirs: "Reservoir",
    vegetation: "Vegetation",
    trees: "Tree",
    tree: "Tree",
    forests: "Forest",
    forest: "Forest",
    woodland: "Woodland",
    farmland: "Farmland",
    farmlands: "Farmland",
    agriculture: "Agricultural Field",
    "agricultural field": "Agricultural Field",
    "agricultural fields": "Agricultural Field",
    "construction site": "Construction Site",
    "construction sites": "Construction Site",
    construction: "Construction Site",
    "industrial area": "Industrial Area",
    "industrial areas": "Industrial Area",
    buildings: "Building",
    building: "Building",
    structures: "Structure",
    structure: "Structure",
    houses: "House",
    house: "House",
    rooftops: "Rooftop",
    rooftop: "Rooftop",
    warehouses: "Warehouse",
    warehouse: "Warehouse",
    hangars: "Hangar",
    hangar: "Hangar",
    ships: "Ship",
    ship: "Ship",
    boats: "Boat",
    boat: "Boat",
    vessels: "Vessel",
    vessel: "Vessel",
    maritime_vessels: "Maritime Vessel",
    maritime_vessel: "Maritime Vessel",
    "maritime vessels": "Maritime Vessel",
    "maritime vessel": "Maritime Vessel",
    bridges: "Bridge",
    bridge: "Bridge",
    roads: "Road",
    road: "Road",
    highways: "Highway",
    highway: "Highway",
    over: "Road",
    pass: "Road",
    overpass: "Road",
    overpasses: "Road",
    street: "Road",
    streets: "Road",
    freeway: "Road",
    freeways: "Road",
    motorway: "Road",
    motorways: "Road",
    intersections: "Intersection",
    intersection: "Intersection",
    cars: "Car",
    car: "Car",
    trucks: "Truck",
    truck: "Truck",
    vehicles: "Vehicle",
    vehicle: "Vehicle",
    airplanes: "Airplane",
    airplane: "Airplane",
    planes: "Plane",
    plane: "Plane",
    aircraft: "Aircraft",
    aircrafts: "Aircraft",
    docks: "Dock",
    dock: "Dock",
    ports: "Port",
    port: "Port",
    piers: "Pier",
    pier: "Pier",
    containers: "Container",
    container: "Container",
    towers: "Tower",
    tower: "Tower",
  };

  if (singularMap[clean]) {
    return singularMap[clean];
  }
  if (!clean || clean.length < 2) return fallback;
  return clean.charAt(0).toUpperCase() + clean.slice(1);
}

// ----------------------------------------------------------------
// Inner top bar with Live Capabilities
// ----------------------------------------------------------------
function PageHeader({ capabilities }: { capabilities: CapabilitiesRecord | null }) {
  const caps = capabilities || {
    captioning: "available",
    grounding: "available",
    vqa: "adapter_available",
    change_analysis: "available",
  };

  return (
    <header className="px-4 sm:px-6 lg:px-8 py-4 border-b border-slate-800/80 bg-[#09131f]/60 backdrop-blur-md flex flex-col sm:flex-row sm:items-center justify-between gap-3">
      <div>
        <div className="flex items-center gap-2">
          <div className="p-1.5 rounded-lg bg-cyan-500/10 border border-cyan-500/20 text-cyan-400">
            <Sparkles size={15} />
          </div>
          <h1 className="text-sm font-semibold text-white tracking-tight">
            NexSpace Natural Language Intelligence
          </h1>
          <span className="text-[10px] font-mono text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded-full border border-emerald-500/20">
            Live ML Engine
          </span>
        </div>
        <p className="text-[11px] text-slate-400 mt-0.5 font-mono">
          Multi-specialist neural vision analysis for satellite and aerial imagery
        </p>
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <div className="flex items-center gap-1.5 bg-slate-900/80 border border-slate-800 rounded-lg px-2.5 py-1 text-[10px] font-mono text-slate-300">
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
          <span>BLIP Captioning: <strong>{caps.captioning}</strong></span>
        </div>
        <div className="flex items-center gap-1.5 bg-slate-900/80 border border-slate-800 rounded-lg px-2.5 py-1 text-[10px] font-mono text-slate-300">
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
          <span>Grounding DINO: <strong>{caps.grounding}</strong></span>
        </div>
      </div>
    </header>
  );
}

// ----------------------------------------------------------------
// Search & Execute Bar
// ----------------------------------------------------------------
// Search & Execute Bar
// ----------------------------------------------------------------
function QueryBar({
  value,
  onChange,
  onExecute,
  loading,
  loadingPhase,
}: {
  value: string;
  onChange: (v: string) => void;
  onExecute: () => void;
  loading: boolean;
  loadingPhase: string;
}) {
  return (
    <div className="w-full space-y-2">
      <div className="relative flex items-center">
        <div className="absolute left-4 text-cyan-400">
          <Search size={16} />
        </div>
        <input
          type="text"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !loading && value.trim()) {
              onExecute();
            }
          }}
          disabled={loading}
          placeholder="Search / ask anything about your imagery (e.g. 'Locate buildings', 'Find ships and boats', 'Describe scene')..."
          className="w-full bg-[#0c1624] border border-slate-800/90 rounded-xl pl-11 pr-32 py-3.5 text-sm text-slate-200 placeholder:text-slate-500 focus:outline-none focus:border-cyan-500/60 focus:ring-1 focus:ring-cyan-500/40 transition-all font-sans disabled:opacity-60"
        />
        <div className="absolute right-2.5">
          <button
            onClick={onExecute}
            disabled={loading || !value.trim()}
            className="flex items-center gap-1.5 bg-gradient-to-r from-cyan-500 to-sky-500 hover:from-cyan-400 hover:to-sky-400 text-slate-950 font-semibold px-4 py-2 rounded-lg text-xs transition-all shadow-[0_0_12px_rgba(6,182,212,0.3)] disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer"
          >
            <span>{loading ? "Analyzing..." : "Analyze"}</span>
            <ArrowRight size={13} />
          </button>
        </div>
      </div>

      {loading && (
        <div className="flex items-center gap-2 text-xs text-cyan-300 font-mono bg-cyan-500/10 border border-cyan-500/25 px-3 py-2 rounded-lg">
          <Clock size={13} className="animate-spin text-cyan-400 shrink-0" />
          <span>{loadingPhase}</span>
        </div>
      )}
    </div>
  );
}

// ----------------------------------------------------------------
// Example Query Chips (Non-Restrictive Examples)
// ----------------------------------------------------------------
function ExampleQueryChips({ onSelect }: { onSelect: (s: string) => void }) {
  const exampleQueries = [
    "Locate buildings",
    "Find ships and boats",
    "Detect construction activity",
    "Compare these images",
  ];

  return (
    <div className="flex flex-wrap items-center gap-2 pt-1">
      <span className="text-[11px] text-slate-500 font-mono flex items-center gap-1">
        <Sparkles size={11} className="text-cyan-400/70" />
        <span>Example queries:</span>
      </span>
      {exampleQueries.map((example, idx) => (
        <button
          key={idx}
          type="button"
          onClick={() => onSelect(example)}
          className="text-xs bg-[#091522] hover:bg-cyan-500/15 text-slate-300 hover:text-cyan-300 border border-slate-800 hover:border-cyan-500/40 px-3 py-1 rounded-lg transition-all font-mono cursor-pointer flex items-center gap-1.5 group"
        >
          <span>{example}</span>
          <ArrowRight size={10} className="text-slate-600 group-hover:text-cyan-400 transition-colors" />
        </button>
      ))}
    </div>
  );
}

import { validateAndProcessImageFile, ACCEPT_FILE_ATTR } from "../utils/imageValidation";

// ----------------------------------------------------------------
// Raster Image Selector / Upload Panel
// ----------------------------------------------------------------
const ImageSelector = memo(function ImageSelector({
  canonicalSource,
  onSelectSource,
  sarImage,
  onSelectSarImage,
  onOpenDiscovery,
}: {
  canonicalSource: CanonicalSourceImage;
  onSelectSource: (src: CanonicalSourceImage) => void;
  sarImage: string | null;
  onSelectSarImage: (b64: string | null) => void;
  onOpenDiscovery?: () => void;
}) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [locationSearch, setLocationSearch] = useState("");
  const [isAcquiring, setIsAcquiring] = useState(false);
  const [acquisitionSuccess, setAcquisitionSuccess] = useState<{
    name: string;
    satellite: string;
    resolution: number;
    cloudCover: number;
    crs: string;
    coords: string;
  } | null>(null);

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    setUploadError(null);
    const file = e.target.files?.[0];
    if (!file) return;

    const result = await validateAndProcessImageFile(file);
    if (!result.valid || !result.source) {
      setUploadError(result.error || "Failed to process the uploaded image.");
      if (fileInputRef.current) fileInputRef.current.value = "";
      return;
    }

    onSelectSource(result.source);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const handleAcquireLocation = async (targetLocation: string) => {
    if (!targetLocation.trim() || isAcquiring) return;
    setIsAcquiring(true);
    setUploadError(null);
    try {
      const res = await fetch("/api/satellite/acquire", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: targetLocation.trim(), forceRefresh: true })
      });
      const data = await res.json();
      if (res.ok && data.sourceImage) {
        onSelectSource(data.sourceImage);
        setAcquisitionSuccess({
          name: data.location.name,
          satellite: data.metadata.satellite,
          resolution: data.metadata.spatialResolutionMeters,
          cloudCover: data.metadata.cloudCoverPercentage,
          crs: data.metadata.crs,
          coords: `${data.location.latitude.toFixed(4)}°N, ${data.location.longitude.toFixed(4)}°E`
        });
      } else {
        setUploadError(data.message || data.error || "Could not acquire satellite tile for this location.");
      }
    } catch {
      setUploadError("Failed to acquire satellite imagery from Copernicus / Landsat datasets.");
    } finally {
      setIsAcquiring(false);
    }
  };

  const POPULAR_LOCATIONS = [
    { label: "📍 Gomti Nagar, Lucknow", query: "Gomti Nagar, Lucknow" },
    { label: "🌊 Marine Drive, Mumbai", query: "Marine Drive, Mumbai" },
    { label: "🏛️ Connaught Place, Delhi", query: "Connaught Place, New Delhi" },
    { label: "🕌 Taj Mahal, Agra", query: "Taj Mahal, Agra" },
    { label: "🏢 Bengaluru", query: "Bengaluru, Karnataka" },
    { label: "🌿 Wayanad", query: "Wayanad, Kerala" },
    { label: "🛕 Ayodhya", query: "Ayodhya, Uttar Pradesh" },
    { label: "🏙️ Dubai", query: "Dubai" },
  ];

  return (
    <div className="border border-slate-800/90 bg-[#0c1624]/60 backdrop-blur-md rounded-xl p-4 sm:p-5 shadow-sm space-y-4">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
        <div>
          <h3 className="text-sm font-semibold text-white tracking-tight flex items-center gap-2">
            <Layers size={14} className="text-cyan-400" />
            Input Imagery Selection
          </h3>
          <p className="text-[11px] text-slate-400 font-mono mt-0.5">
            Supported: JPG, PNG, TIFF, GeoTIFF, Sentinel-2 L2A STAC · Max 25 MB
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <input
            ref={fileInputRef}
            type="file"
            accept={ACCEPT_FILE_ATTR}
            onChange={handleFileUpload}
            className="hidden"
          />
          <button
            onClick={() => fileInputRef.current?.click()}
            className="flex items-center gap-1.5 bg-slate-900/80 hover:bg-slate-800 text-slate-300 border border-slate-700 text-xs px-3 py-1.5 rounded-lg transition-all cursor-pointer font-mono"
          >
            <Upload size={12} className="text-cyan-400" />
            <span>Upload Imagery</span>
          </button>
          <span className="text-slate-500 text-xs font-mono">OR</span>
          <button
            onClick={onOpenDiscovery}
            className="flex items-center gap-1.5 bg-gradient-to-r from-cyan-500/20 to-blue-500/20 hover:from-cyan-500/30 hover:to-blue-500/30 text-cyan-300 border border-cyan-400/50 text-xs px-3.5 py-1.5 rounded-lg transition-all cursor-pointer font-bold shadow-[0_0_12px_rgba(6,182,212,0.2)]"
          >
            <Compass size={13} className="text-cyan-400 animate-pulse" />
            <span>Find Satellite Imagery</span>
          </button>
        </div>
      </div>

      {uploadError && (
        <div className="bg-rose-500/10 border border-rose-500/30 px-3 py-1.5 rounded-lg text-rose-300 text-xs font-mono flex items-center gap-2">
          <AlertTriangle size={13} className="text-rose-400 shrink-0" />
          <span>{uploadError}</span>
        </div>
      )}

      {/* Visual Image Catalog Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-3">
        {DEMO_IMAGE_CATALOG.map((item) => {
          const isSelected = canonicalSource.dataUrl === item.base64 || (item.category === "sar" && canonicalSource.filename === "sar.png");

          return (
            <button
              key={item.id}
              onClick={() => {
                const demoSource: CanonicalSourceImage = {
                  id: `src-demo-${item.id}`,
                  filename: `${item.id}.png`,
                  mediaType: "image/png",
                  dataUrl: item.base64,
                  source: "demo",
                };
                onSelectSource(demoSource);
                if (item.category === "sar") {
                  onSelectSarImage(item.base64);
                } else {
                  onSelectSarImage(null);
                }
              }}
              className={`group relative flex flex-col rounded-lg border overflow-hidden transition-all text-left cursor-pointer ${isSelected
                ? "border-cyan-400 bg-cyan-500/15 shadow-[0_0_12px_rgba(6,182,212,0.3)] ring-1 ring-cyan-400/50"
                : "border-slate-800 bg-slate-950/80 hover:border-slate-700 hover:bg-slate-900/60"
                }`}
            >
              <div className="relative aspect-square w-full bg-slate-900 overflow-hidden">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={item.url}
                  alt={item.name}
                  loading="lazy"
                  decoding="async"
                  className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-200"
                />
                <span
                  className={`absolute top-1.5 right-1.5 text-[9px] font-mono px-1.5 py-0.5 rounded backdrop-blur-md ${item.category === "sar"
                    ? "bg-purple-900/80 text-purple-300 border border-purple-500/30"
                    : "bg-slate-900/80 text-cyan-300 border border-cyan-500/30"
                    }`}
                >
                  {item.category.toUpperCase()}
                </span>
                {isSelected && (
                  <span className="absolute bottom-1.5 left-1.5 bg-cyan-500 text-slate-950 p-0.5 rounded-full shadow">
                    <Check size={10} className="stroke-[3]" />
                  </span>
                )}
              </div>
              <div className="p-2 space-y-0.5">
                <div className="text-xs font-semibold text-slate-200 group-hover:text-white truncate">
                  {item.name}
                </div>
                <div className="text-[10px] text-slate-400 font-mono truncate">
                  {item.dimensions}
                </div>
              </div>
            </button>
          );
        })}
      </div>

      {/* Active Selection Details Banner */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between text-xs font-mono text-slate-400 pt-2 border-t border-slate-800/60 gap-2">
        <div className="flex items-center gap-3 flex-wrap">
          <span className="flex items-center gap-1.5">
            <ImageIcon size={12} className="text-cyan-400" />
            <span>Active Source:</span>
            <strong className="text-slate-200">{canonicalSource.filename}</strong>
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-slate-800 text-cyan-300 border border-slate-700">
              {canonicalSource.source === "upload" ? "Uploaded File" : "Sample Raster"}
            </span>
          </span>
          {sarImage && (
            <span className="text-purple-300 bg-purple-950/60 border border-purple-500/30 px-2 py-0.5 rounded text-[10px]">
              + Dual SAR Channel Active
            </span>
          )}
        </div>
        {sarImage && (
          <button
            onClick={() => onSelectSarImage(null)}
            className="text-slate-500 hover:text-rose-400 transition-colors cursor-pointer text-[11px]"
          >
            Remove SAR Layer
          </button>
        )}
      </div>
    </div>
  );
});

// ----------------------------------------------------------------
// Grounding Visual Bounding Box Overlay Component (Expansive & Border-Touching)
// ----------------------------------------------------------------
const GroundingVisualOverlay = memo(function GroundingVisualOverlay({
  imageSrc,
  detections,
}: {
  imageSrc: string;
  detections?: unknown[];
}) {
  const [zoom, setZoom] = useState(1);
  const [isFullFit, setIsFullFit] = useState(false);

  const handleZoomIn = () => setZoom((z) => Math.min(3, +(z + 0.25).toFixed(2)));
  const handleZoomOut = () => setZoom((z) => Math.max(1, +(z - 0.25).toFixed(2)));
  const handleReset = () => {
    setZoom(1);
    setIsFullFit(false);
  };

  return (
    <div className="relative w-full rounded-xl overflow-hidden border border-cyan-500/25 bg-[#08121e] flex items-center justify-center min-h-[260px] max-h-[380px]">
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={imageSrc}
        alt="Satellite Target Raster"
        className="w-full h-auto max-h-[380px] object-contain mx-auto block select-none pointer-events-none"
      />

      {/* Real Grounding Bounding Boxes (Exact Sub-Pixel SVG Alignment with preserveAspectRatio="xMidYMid meet") */}
      <svg
        viewBox="0 0 1000 1000"
        preserveAspectRatio="xMidYMid meet"
        className="absolute inset-0 w-full h-full pointer-events-none"
      >
        {detections?.map((rawDet, idx) => {
          const box = normalizeBox(rawDet);
          if (!box) return null;

          const [xmin, ymin, xmax, ymax] = box;
          const width = Math.max(2, xmax - xmin);
          const height = Math.max(2, ymax - ymin);

          const detObj = rawDet as {
            label?: string;
            score?: number;
            has_polygon?: boolean;
            polygon?: [number, number][];
            polygon_pixels?: [number, number][];
          };
          const label = detObj.label || "Detected Object";
          const score = typeof detObj.score === "number" ? detObj.score : null;
          const labelText = score !== null ? `${label} · ${Math.round(score * 100)}%` : label;
          const labelWidth = Math.max(90, labelText.length * 8.5 + 16);

          let polyPoints: string | null = null;
          if (Array.isArray(detObj.polygon) && detObj.polygon.length >= 3) {
            polyPoints = detObj.polygon.map(([px, py]) => `${px},${py}`).join(" ");
          }

          return (
            <g key={idx}>
              {polyPoints ? (
                <polygon
                  points={polyPoints}
                  fill="rgba(0, 255, 127, 0.45)"
                  stroke="#00ff7f"
                  strokeWidth="2.5"
                  strokeLinejoin="round"
                  vectorEffect="non-scaling-stroke"
                />
              ) : (
                <rect
                  x={xmin}
                  y={ymin}
                  width={width}
                  height={height}
                  fill="rgba(0, 255, 127, 0.25)"
                  stroke="#00ff7f"
                  strokeWidth="2.5"
                  vectorEffect="non-scaling-stroke"
                />
              )}
              <rect
                x={xmin}
                y={Math.max(0, ymin - 22)}
                width={labelWidth}
                height={20}
                fill="rgba(15, 23, 42, 0.92)"
                stroke="rgba(0, 255, 127, 0.45)"
                strokeWidth="1"
                rx="3"
              />
              <text
                x={xmin + 6}
                y={Math.max(14, ymin - 7)}
                fill="#6ee7b7"
                fontSize="12"
                fontFamily="ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace"
                fontWeight="600"
              >
                {labelText}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
});

// ----------------------------------------------------------------
// Scan Results Panel (Full-Screen Spatial Intelligence Workspace)
// ----------------------------------------------------------------
interface EnrichedDetection {
  index: number;
  tag: string;
  rawLabel?: string;
  cleanLabel: string;
  score: number;
  scorePercent: number;
  box: [number, number, number, number];
  width: number;
  height: number;
  areaPx: number;
  areaEstimate?: string;
  confCategory: "high" | "medium" | "low";
  geometryType?: "polygon" | "box" | "line";
  polygon?: [number, number][];
  coordinates?: {
    lat: number;
    lon: number;
    formatted: string;
  };
  gmapsUrl?: string;
}

const ScanResultsPanel = memo(function ScanResultsPanel({
  result,
  sourceImage,
  queryText,
}: {
  result: NexSpaceQueryResponse;
  sourceImage: CanonicalSourceImage;
  queryText?: string;
}) {
  const [showTechnical, setShowTechnical] = useState(false);
  const [selectedDetIndex, setSelectedDetIndex] = useState<number | null>(null);
  const [confidenceFilter, setConfidenceFilter] = useState<"all" | "high" | "medium" | "low">("all");
  const [selectedCategory, setSelectedCategory] = useState<string>("all");
  const [sortBy, setSortBy] = useState<"confidence_desc" | "confidence_asc" | "area_desc" | "id_asc">("confidence_desc");
  const [currentPage, setCurrentPage] = useState<number>(1);
  const [zoomLevel, setZoomLevel] = useState<number>(1);
  const [showBboxes, setShowBboxes] = useState<boolean>(true);
  const [viewMode, setViewMode] = useState<"evidence" | "map">("evidence");

  if (!result) return null;

  const grounding = result.grounding;
  const rawDetections = Array.isArray(grounding?.detections)
    ? grounding.detections
    : (Array.isArray(result.detections) ? result.detections : []);
  const vqaResults = Array.isArray(result.vqa_results) ? result.vqa_results : [];
  const report = result.investigation_report;
  const trace = Array.isArray(result.execution_trace) ? result.execution_trace : [];
  const limitations = result.limitations || report?.limitations || [];

  const overallConf = getConfidenceInfo(result.confidence, result.confidence_type);

  const isOffline = result.backend_status === "offline_fallback";
  const isDemo = sourceImage.source === "demo";
  const hasFallbackTool = isOffline || (result.confidence_type === "heuristic" && (!result.detections || result.detections.length === 0) && result.selected_tools?.length === 1 && result.selected_tools[0] === "VQA");

  const imgW = sourceImage.width || 1024;
  const imgH = sourceImage.height || 1024;

  // Preserve user query as the investigation target
  const effectiveQuery = queryText || result.query || ((result as unknown as Record<string, unknown>).raw_input as string) || "Spatial Intelligence Investigation";

  // Check if query or result represents a pure scene captioning or counting task
  const effectiveTask = (result.task || result.task_type || result.plan?.task_type || "").toUpperCase();
  const isCountingTask = effectiveTask === "COUNTING" || Boolean(result.count_summary) || Boolean((result.parameters as Record<string, unknown> | undefined)?.is_counting);
  const isPresenceTask = effectiveTask === "PRESENCE_CHECK";
  const hasCaptionTool = Boolean(result.selected_tools?.includes("Optical_Caption") || result.selected_tools?.includes("SAR_Caption") || result.optical_caption);
  const hasGroundingTool = Boolean(result.selected_tools?.includes("Grounding") || result.grounding);

  const isCaptionOnly = (effectiveTask === "CAPTIONING") || (hasCaptionTool && !hasGroundingTool && rawDetections.length === 0 && !isCountingTask);
  const isMultiTaskWithCaption = hasCaptionTool && hasGroundingTool;

  // Derive target label from detections, falling back to query target (NEVER action words)
  const targetPhraseParam = (result.target || (result.parameters as Record<string, unknown> | undefined)?.target_phrase) as string | undefined;
  const primaryTargetLabel = isCaptionOnly
    ? "Scene Description"
    : targetPhraseParam
      ? formatCanonicalLabel(targetPhraseParam, "Entity")
      : rawDetections.length > 0
        ? formatCanonicalLabel((rawDetections[0] as { label?: string }).label, "Entity")
        : "Targets";

  const detectionStatus = (result.detection_status || (result.parameters as Record<string, unknown> | undefined)?.detection_status || (rawDetections.length > 0 ? "DETECTIONS_FOUND" : "GENUINE_ZERO")) as string;
  const countSummary = result.count_summary || ((result.parameters as Record<string, unknown> | undefined)?.count_summary as { count: number; target: string; reliability: string; message: string; detections_count: number } | undefined);

  const primaryPlural = primaryTargetLabel.endsWith("s") ? primaryTargetLabel : `${primaryTargetLabel}s`;

  const intel: InvestigationResult = useMemo(
    () => synthesizeInvestigationResult(effectiveQuery, result, sourceImage),
    [effectiveQuery, result, sourceImage]
  );

  const enrichedDetections: EnrichedDetection[] = useMemo(() => {
    const list: EnrichedDetection[] = [];
    rawDetections.forEach((d: unknown, idx: number) => {
      const box = normalizeBox(d, imgW, imgH);
      if (!box) return;
      const [xmin, ymin, xmax, ymax] = box;
      const width = Math.max(2, xmax - xmin);
      const height = Math.max(2, ymax - ymin);
      const detObj = d as {
        label?: string;
        score?: number;
        confidence?: number;
        geometry_type?: string;
        has_polygon?: boolean;
        polygon?: [number, number][];
        polygon_pixels?: [number, number][];
        geometry?: { type?: string; coordinates?: [number, number][][] };
      };
      const score = typeof detObj.score === "number" && !isNaN(detObj.score)
        ? detObj.score
        : (typeof detObj.confidence === "number" && !isNaN(detObj.confidence) ? detObj.confidence : 0.62);
      const scorePercent = Math.round(score * 100);
      const cleanLabel = formatCanonicalLabel(detObj.label, primaryTargetLabel);
      const tag = String(idx + 1).padStart(2, "0");
      const areaPx = Math.round((width / 1000 * imgW) * (height / 1000 * imgH));

      let polyCoords: [number, number][] | undefined = undefined;
      if (Array.isArray(detObj.polygon) && detObj.polygon.length >= 3) {
        polyCoords = detObj.polygon;
      } else if (Array.isArray(detObj.polygon_pixels) && detObj.polygon_pixels.length >= 3 && imgW > 0 && imgH > 0) {
        polyCoords = detObj.polygon_pixels.map(([px, py]) => [
          Math.round((px / imgW) * 1000),
          Math.round((py / imgH) * 1000)
        ]);
      } else if (detObj.geometry?.type === "Polygon" && Array.isArray(detObj.geometry.coordinates?.[0])) {
        polyCoords = detObj.geometry.coordinates[0];
      }

      let confCategory: "high" | "medium" | "low" = "medium";
      if (score >= 0.70) {
        confCategory = "high";
      } else if (score < 0.40) {
        confCategory = "low";
      } else {
        confCategory = "medium";
      }

      // Geospatial coordinates resolution
      let coordinates: { lat: number; lon: number; formatted: string } | undefined;
      let gmapsUrl: string | undefined;
      const feat = result.geojson?.features?.find(
        (f: any) =>
          f.properties?.evidence_id === `det_${idx}` ||
          f.id === `det_${idx}` ||
          f.id === String(idx) ||
          f.properties?.evidence_id === tag
      ) || result.geojson?.features?.[idx];
      const ring =
        feat?.geometry?.type === "Polygon" && Array.isArray(feat.geometry.coordinates)
          ? (feat.geometry.coordinates[0] as number[][])
          : null;
      const latLon = ring && ring.length > 0 ? { lon: ring[0][0], lat: ring[0][1] } : null;
      if (latLon) {
        const lat = latLon.lat;
        const lon = latLon.lon;
        const latStr = `${Math.abs(lat).toFixed(4)}° ${lat >= 0 ? "N" : "S"}`;
        const lonStr = `${Math.abs(lon).toFixed(4)}° ${lon >= 0 ? "E" : "W"}`;
        coordinates = { lat, lon, formatted: `${latStr}, ${lonStr}` };
        gmapsUrl = `https://www.google.com/maps?q=${lat.toFixed(6)},${lon.toFixed(6)}`;
      }

      list.push({
        index: idx,
        tag,
        rawLabel: detObj.label,
        cleanLabel,
        score,
        scorePercent,
        box,
        width,
        height,
        areaPx,
        areaEstimate: areaPx > 3000 ? `${(areaPx * 0.25).toLocaleString()} m² (est.)` : undefined,
        confCategory,
        geometryType: polyCoords ? "polygon" : "box",
        polygon: polyCoords,
        coordinates,
        gmapsUrl,
      });
    });
    return list;
  }, [rawDetections, imgW, imgH, primaryTargetLabel, result.geojson]);

  const targetLabelDisplay = enrichedDetections.length === 1 ? primaryTargetLabel : primaryPlural;

  // Categories present in detections for multi-target queries
  const detectedCategories = useMemo(() => {
    const map = new Map<string, number>();
    enrichedDetections.forEach((d) => {
      const cat = d.cleanLabel;
      map.set(cat, (map.get(cat) || 0) + 1);
    });
    return Array.from(map.entries()).map(([label, count]) => ({ label, count }));
  }, [enrichedDetections]);

  // Confidence Category Counts
  const allCount = enrichedDetections.length;
  const highCount = enrichedDetections.filter((d) => d.confCategory === "high").length;
  const medCount = enrichedDetections.filter((d) => d.confCategory === "medium").length;
  const lowCount = enrichedDetections.filter((d) => d.confCategory === "low").length;

  // Filter & Sort
  const filteredDetections = useMemo(() => {
    let list = enrichedDetections;
    if (selectedCategory !== "all") {
      list = list.filter((d) => d.cleanLabel.toLowerCase() === selectedCategory.toLowerCase());
    }
    if (confidenceFilter === "all") return list;
    return list.filter((d) => d.confCategory === confidenceFilter);
  }, [enrichedDetections, confidenceFilter, selectedCategory]);

  const sortedDetections = useMemo(() => {
    const copy = [...filteredDetections];
    if (sortBy === "confidence_desc") {
      copy.sort((a, b) => b.score - a.score);
    } else if (sortBy === "confidence_asc") {
      copy.sort((a, b) => a.score - b.score);
    } else if (sortBy === "area_desc") {
      copy.sort((a, b) => b.areaPx - a.areaPx);
    } else if (sortBy === "id_asc") {
      copy.sort((a, b) => a.index - b.index);
    }
    return copy;
  }, [filteredDetections, sortBy]);

  // Pagination (6 rich cards per page)
  const PAGE_SIZE = 6;
  const totalPages = Math.max(1, Math.ceil(sortedDetections.length / PAGE_SIZE));
  const activePage = Math.min(currentPage, totalPages);
  const paginatedDetections = sortedDetections.slice((activePage - 1) * PAGE_SIZE, activePage * PAGE_SIZE);
  const startIdx = sortedDetections.length > 0 ? (activePage - 1) * PAGE_SIZE + 1 : 0;
  const endIdx = Math.min(activePage * PAGE_SIZE, sortedDetections.length);

  // Bidirectional Selection Handler
  const handleSelectDetection = (index: number | null) => {
    if (index === null || selectedDetIndex === index) {
      setSelectedDetIndex(null);
      return;
    }
    setSelectedDetIndex(index);

    // If selected detection is not in currently visible page, adjust filter & page
    const foundInSorted = sortedDetections.findIndex((d) => d.index === index);
    if (foundInSorted >= 0) {
      const targetPage = Math.floor(foundInSorted / PAGE_SIZE) + 1;
      if (targetPage !== activePage) {
        setCurrentPage(targetPage);
      }
    } else {
      // It's hidden by filter, reset filter to 'all' so card is accessible
      setConfidenceFilter("all");
      const foundInAll = enrichedDetections.findIndex((d) => d.index === index);
      if (foundInAll >= 0) {
        const targetPage = Math.floor(foundInAll / PAGE_SIZE) + 1;
        setCurrentPage(targetPage);
      }
    }
  };

  return (
    <div className="w-full border border-cyan-500/30 bg-[#091524] rounded-xl p-4 sm:p-5 shadow-[0_0_24px_rgba(6,182,212,0.12)] space-y-5">
      {/* 1. Header with Source Info, User Query & Overall Confidence */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-800 pb-4">
        <div>
          <div className="flex flex-wrap items-center gap-2">
            <ShieldCheck className="text-cyan-400" size={18} />
            <h2 className="text-base font-semibold text-white tracking-tight">
              Investigation Findings
            </h2>
            {isOffline ? (
              <span className="text-[10px] font-mono text-rose-300 bg-rose-500/20 border border-rose-500/40 px-2 py-0.5 rounded-full font-bold">
                🔴 AI BACKEND OFFLINE
              </span>
            ) : isSatellite ? (
              <span className="text-[10px] font-mono text-cyan-300 bg-cyan-500/20 border border-cyan-500/40 px-2 py-0.5 rounded-full font-bold">
                🟢 LIVE COPERNICUS SATELLITE ANALYSIS
              </span>
            ) : isUpload ? (
              <span className="text-[10px] font-mono text-emerald-300 bg-emerald-500/20 border border-emerald-500/40 px-2 py-0.5 rounded-full font-bold">
                🟢 UPLOADED RASTER ANALYSIS
              </span>
            ) : hasFallbackTool ? (
              <span className="text-[10px] font-mono text-amber-300 bg-amber-500/20 border border-amber-500/40 px-2 py-0.5 rounded-full font-bold">
                🟡 FALLBACK ANALYSIS
              </span>
            ) : (
              <span className="text-[10px] font-mono text-blue-300 bg-blue-500/20 border border-blue-500/40 px-2 py-0.5 rounded-full font-bold">
                🔵 DEMO CATALOG SCENE
              </span>
            )}
          </div>
          <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs font-mono mt-1 text-slate-400">
            <span>
              Investigation Task: <strong className="text-white font-semibold">{isCaptionOnly ? "Scene Description" : isCountingTask ? "Object Counting" : isPresenceTask ? "Presence Verification" : effectiveTask || "Spatial Intelligence"}</strong>
            </span>
            {targetPhraseParam && (
              <>
                <span className="text-slate-600">·</span>
                <span>
                  Target: <strong className="text-cyan-300 font-semibold">{primaryTargetLabel}</strong>
                </span>
              </>
            )}
            <span className="text-slate-600">·</span>
            <span>
              Query: <strong className="text-slate-300 font-semibold">"{effectiveQuery}"</strong>
            </span>
            <span className="text-slate-600">·</span>
            <span>
              Source: <strong className="text-cyan-300">{sourceImage.filename}</strong> · {sourceImage.source === "upload" ? "Uploaded by user" : "Verified demo tile"}
            </span>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <div className={`flex items-center gap-1.5 px-3 py-1 rounded-lg border text-xs font-mono ${overallConf.bg} ${overallConf.border} ${overallConf.color}`}>
            <span>Confidence:</span>
            <strong>{isOffline ? "Not available" : overallConf.label}</strong>
          </div>
        </div>
      </div>

      {/* 2. Full-Screen Spatial Intelligence Workspace (Left ~62-65%, Right ~35-38%) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-stretch min-h-[560px] lg:min-h-[620px]">
        {/* Left Column: Satellite Evidence Viewer (Col-span-7 / 62-65%) */}
        <div className="lg:col-span-7 xl:col-span-8 flex flex-col bg-[#060e18] rounded-xl border border-slate-800 relative overflow-hidden shadow-2xl">
          {/* Viewer HUD Control Bar */}
          <div className="flex items-center justify-between px-3.5 py-2.5 bg-slate-950/90 border-b border-slate-800/80 text-xs font-mono">
            <div className="flex items-center gap-2">
              <span className="flex h-2 w-2 relative">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-cyan-500"></span>
              </span>
              <span className="text-cyan-300 font-bold tracking-wider uppercase text-[11px]">
                {result.selected_scene ? "Sentinel-2 Satellite Imagery" : "Satellite Evidence Viewer"}
              </span>
              <span className="text-[10px] text-slate-400 bg-slate-900 border border-slate-800 px-2 py-0.5 rounded font-mono">
                {result.selected_scene ? `${result.selected_scene.platform || "Sentinel-2"} · 10m Optical GSD` : "1000×1000 SYNC"}
              </span>
            </div>

            <div className="flex items-center gap-2">
              <div className="flex items-center bg-slate-900 border border-slate-800 rounded-lg p-0.5 text-[11px]">
                <button
                  onClick={() => setViewMode("evidence")}
                  className={`px-2.5 py-1 rounded transition-colors cursor-pointer ${viewMode === "evidence"
                    ? "bg-cyan-500 text-slate-950 font-bold shadow-sm"
                    : "text-slate-400 hover:text-white"
                    }`}
                >
                  🛰️ Evidence
                </button>
                <button
                  onClick={() => setViewMode("map")}
                  className={`px-2.5 py-1 rounded transition-colors cursor-pointer ${viewMode === "map"
                    ? "bg-cyan-500 text-slate-950 font-bold shadow-sm"
                    : "text-slate-400 hover:text-white"
                    }`}
                >
                  🗺️ Google Maps
                </button>
              </div>

              {viewMode === "evidence" && (
                <>
                  <button
                    onClick={() => setShowBboxes(!showBboxes)}
                    className={`flex items-center gap-1 px-2.5 py-1 rounded text-[11px] border transition-colors cursor-pointer ${showBboxes
                      ? "bg-cyan-500/20 border-cyan-500/40 text-cyan-300 font-medium"
                      : "bg-slate-900 border-slate-800 text-slate-400"
                      }`}
                    title="Toggle Spatial Bounding Boxes"
                  >
                    {showBboxes ? <Eye size={12} /> : <EyeOff size={12} />}
                    <span>{showBboxes ? "Overlay On" : "Overlay Off"}</span>
                  </button>
                  <div className="flex items-center bg-slate-900 border border-slate-800 rounded p-0.5 text-[11px] text-slate-300">
                    <button
                      onClick={() => setZoomLevel((z) => Math.max(1, z - 0.25))}
                      className="px-1.5 py-0.5 hover:text-cyan-300 transition-colors cursor-pointer font-bold"
                      title="Zoom Out"
                    >
                      −
                    </button>
                    <span className="px-1.5 font-mono text-[10px] text-slate-400">{Math.round(zoomLevel * 100)}%</span>
                    <button
                      onClick={() => setZoomLevel((z) => Math.min(2.5, z + 0.25))}
                      className="px-1.5 py-0.5 hover:text-cyan-300 transition-colors cursor-pointer font-bold"
                      title="Zoom In"
                    >
                      +
                    </button>
                    {zoomLevel !== 1 && (
                      <button
                        onClick={() => setZoomLevel(1)}
                        className="px-1.5 text-[10px] text-cyan-400 hover:underline cursor-pointer font-bold"
                      >
                        Reset
                      </button>
                    )}
                  </div>
                </>
              )}
            </div>
          </div>

          {/* Viewport Canvas with Synchronized SVG Overlay OR Google Maps */}
          {viewMode === "map" ? (
            <div className="flex-1 p-2 bg-[#030811] min-h-[460px]">
              <GoogleMapView
                geoMeta={result.geospatial_metadata || result.geospatial}
                geojson={result.geojson}
                evidence={result.evidence as any}
                selectedDetectionId={
                  selectedDetIndex !== null
                    ? String((rawDetections[selectedDetIndex] as any)?.evidence_id || `det_${selectedDetIndex}`)
                    : null
                }
                onSelectDetection={(id) => {
                  const idx = enrichedDetections.findIndex(
                    (d) => d.tag === id || `det_${d.index}` === id || `feat_${d.index}` === id
                  );
                  if (idx >= 0) handleSelectDetection(enrichedDetections[idx].index);
                }}
                aoiBbox={
                  result.geospatial_metadata?.bounds_world
                    ? [
                      result.geospatial_metadata.bounds_world.min_x,
                      result.geospatial_metadata.bounds_world.min_y,
                      result.geospatial_metadata.bounds_world.max_x,
                      result.geospatial_metadata.bounds_world.max_y,
                    ]
                    : (result.google_maps_overlay?.bounds
                      ? [
                        result.google_maps_overlay.bounds.west,
                        result.google_maps_overlay.bounds.south,
                        result.google_maps_overlay.bounds.east,
                        result.google_maps_overlay.bounds.north,
                      ]
                      : null)
                }
                scene={result.selected_scene || getCurrentInvestigation()?.selectedSatelliteScene || null}
                location={((result as any)?.geospatial?.resolved_location || (result as any)?.resolved_location) ?? null}
                overlayUrl={result.satellite_image_url || result.google_maps_overlay?.image_url || null}
                overlayBounds={result.google_maps_overlay?.bounds || null}
                height="500px"
              />
            </div>
          ) : (
            <div className="flex-1 relative flex items-center justify-center p-3 bg-[#030811] overflow-hidden min-h-[380px] lg:min-h-[460px]">
              <div
                className="relative w-full h-full flex items-center justify-center transition-transform duration-200"
                style={{ transform: `scale(${zoomLevel})` }}
              >
                <svg
                  viewBox="0 0 1000 1000"
                  preserveAspectRatio="xMidYMid meet"
                  className="w-full h-full max-h-[520px] select-none"
                >
                  {/* Base Raster Layer: Displays actual retrieved Sentinel-2 RGB raster */}
                  <image
                    href={
                      result.satellite_image_url ||
                      result.google_maps_overlay?.image_url ||
                      result.satellite_image_base64 ||
                      result.google_maps_overlay?.image_base64 ||
                      sourceImage.dataUrl
                    }
                    x="0"
                    y="0"
                    width="1000"
                    height="1000"
                    preserveAspectRatio="none"
                  />

                  {/* Detection Overlays with Canonical Numbered Tags */}
                  {showBboxes &&
                    enrichedDetections.map((det) => {
                      const isSelected = selectedDetIndex === det.index;
                      const [xmin, ymin, xmax, ymax] = det.box;
                      const labelTag = `#${det.tag} ${det.cleanLabel}`;
                      const tagWidth = Math.max(68, labelTag.length * 7.2 + 8);

                      return (
                        <g
                          key={det.index}
                          onClick={(e) => {
                            e.stopPropagation();
                            handleSelectDetection(det.index);
                          }}
                          className="cursor-pointer pointer-events-auto"
                        >
                          {/* Polygon Region Contour if available */}
                          {det.polygon && det.polygon.length >= 3 ? (
                            <polygon
                              points={det.polygon.map(([px, py]) => `${px},${py}`).join(" ")}
                              fill={isSelected ? "rgba(0, 255, 127, 0.65)" : "rgba(0, 255, 127, 0.45)"}
                              stroke={isSelected ? "#5fffd7" : "#00ff7f"}
                              strokeWidth={isSelected ? "3" : "2"}
                              strokeLinejoin="round"
                              vectorEffect="non-scaling-stroke"
                              className="transition-all duration-150"
                            />
                          ) : (
                            /* Bounding Box Rect Fallback */
                            <rect
                              x={xmin}
                              y={ymin}
                              width={det.width}
                              height={det.height}
                              fill={isSelected ? "rgba(0, 255, 127, 0.40)" : "rgba(0, 255, 127, 0.20)"}
                              stroke={isSelected ? "#5fffd7" : "#00ff7f"}
                              strokeWidth={isSelected ? "3" : "1.75"}
                              vectorEffect="non-scaling-stroke"
                              className="transition-all duration-150"
                            />
                          )}
                          {/* Floating Label Badge */}
                          <rect
                            x={xmin}
                            y={Math.max(0, ymin - 18)}
                            width={tagWidth}
                            height={16}
                            fill={isSelected ? "rgba(0, 255, 127, 0.95)" : "rgba(10, 19, 34, 0.88)"}
                            stroke={isSelected ? "#5fffd7" : "rgba(0, 255, 127, 0.5)"}
                            strokeWidth="1"
                            rx="2"
                          />
                          <text
                            x={xmin + 4}
                            y={Math.max(12, ymin - 6)}
                            fill={isSelected ? "#020617" : "#a7f3d0"}
                            fontSize="10"
                            fontFamily="ui-monospace, monospace"
                            fontWeight="bold"
                          >
                            {labelTag}
                          </text>
                        </g>
                      );
                    })}
                </svg>
              </div>
            </div>
          )}

          {/* Bottom Metadata & Telemetry Strip */}
          <div className="px-3.5 py-2.5 bg-slate-950/90 border-t border-slate-800/80 flex flex-wrap items-center justify-between gap-2 text-[10px] font-mono text-slate-400">
            <div className="flex flex-wrap items-center gap-3">
              {result.selected_scene ? (
                <>
                  <span>Sensor: <strong className="text-slate-300">{result.selected_scene.platform || "Sentinel-2"} (MSI)</strong></span>
                  <span>GSD: <strong className="text-cyan-300">{result.selected_scene.spatial_resolution_m || 10}m Optical</strong></span>
                  <span>Cloud: <strong className="text-slate-300">{result.selected_scene.cloud_cover !== undefined ? `${result.selected_scene.cloud_cover.toFixed(1)}%` : "0.0%"}</strong></span>
                  <span>CRS: <strong className="text-slate-300">{result.google_maps_overlay?.crs || result.selected_scene.crs || "EPSG:4326"}</strong></span>
                  <span>Scene: <strong className="text-cyan-300 truncate max-w-[200px]" title={result.selected_scene.scene_id}>{result.selected_scene.scene_id}</strong></span>
                </>
              ) : (
                <>
                  <span>Sensor: <strong className="text-slate-300">{(sourceImage as unknown as Record<string, unknown>).sensor as string || (result.selected_scene ? "Satellite MSI" : "Standard RGB")}</strong></span>
                  <span>GSD: <strong className="text-cyan-300">{(sourceImage as unknown as Record<string, unknown>).resolution as string || (result.geospatial?.geospatial_available ? `${result.geospatial.gsd}m` : "Unavailable (Non-georeferenced)")}</strong></span>
                  <span>CRS: <strong className="text-slate-300">{result.spatial_summary?.crs || (result.geospatial?.geospatial_available ? result.geospatial.crs : undefined) || ((sourceImage as unknown as Record<string, unknown>).crs as string) || "None (Pixel Coordinates)"}</strong></span>
                  <span>Dim: <strong className="text-slate-300">{imgW}×{imgH} px</strong></span>
                </>
              )}
            </div>
            <div className="flex items-center gap-2">
              {result.geospatial?.geospatial_available || result.selected_scene ? (
                <span className="text-emerald-400 font-semibold">● Spatial Sync Active</span>
              ) : (
                <span className="text-slate-500 font-medium">○ Pixel Frame (Unreferenced)</span>
              )}
            </div>
          </div>
        </div>

        {/* Right Column: Intelligence Result Panel (Col-span-5 / ~35%) */}
        <div className="lg:col-span-5 xl:col-span-4 flex flex-col bg-slate-900/95 rounded-xl border border-slate-800 p-4 shadow-xl h-full space-y-3.5">
          {/* 1. Hero Intelligence Conclusion & Natural-Language Answer */}
          <div className="p-3.5 rounded-xl bg-gradient-to-br from-slate-950 via-[#061424] to-slate-950 border border-cyan-500/30 shadow-[0_0_20px_rgba(6,182,212,0.14)] space-y-2">
            <div className="flex items-center justify-between gap-2">
              <div className="flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse shrink-0" />
                <h3 className="text-xs font-mono font-bold text-cyan-300 uppercase tracking-wider">
                  {intel.headline}
                </h3>
              </div>
              <span
                className={`text-[10px] font-mono font-bold px-2.5 py-0.5 rounded-full shrink-0 border ${intel.detectionState === "detections_found"
                  ? "text-emerald-300 bg-emerald-500/15 border-emerald-500/30"
                  : intel.detectionState === "not_applicable"
                    ? "text-cyan-300 bg-cyan-500/15 border-cyan-500/30"
                    : "text-slate-300 bg-slate-800 border-slate-700"
                  }`}
              >
                {intel.detectionState === "detections_found"
                  ? `${intel.targetCount} Verified Instances`
                  : intel.detectionState === "not_applicable"
                    ? "Scene Description"
                    : "No Matches"}
              </span>
            </div>

            {/* Human-Readable Synthesized Natural Language Answer */}
            <p className="text-xs sm:text-sm font-medium text-slate-100 leading-snug font-sans">
              "{intel.naturalLanguageAnswer}"
            </p>

            {/* Location & Quick Map Toggle */}
            <div className="flex items-center justify-between pt-1 border-t border-slate-800/80 text-[11px] font-mono text-slate-400">
              <span className="flex items-center gap-1.5 truncate">
                <MapPin size={11} className="text-cyan-400 shrink-0" />
                <span className="text-slate-300 truncate font-sans">
                  {intel.locationInfo.resolvedName}
                </span>
              </span>
              <button
                onClick={() => setViewMode(viewMode === "map" ? "evidence" : "map")}
                className="inline-flex items-center gap-1 text-cyan-400 hover:text-cyan-200 transition-colors cursor-pointer shrink-0 ml-2 font-bold"
              >
                <span>{viewMode === "map" ? "🛰️ View Image" : "🗺️ View on Map"}</span>
              </button>
            </div>
          </div>

          {/* Dedicated Satellite Scene Card (Section 4, 11, 13) */}
          {result.selected_scene && (
            <div className="p-3.5 rounded-xl bg-slate-950/90 border border-cyan-500/30 space-y-2.5 font-mono text-xs shadow-inner">
              <div className="flex items-center justify-between pb-1.5 border-b border-slate-800/80">
                <div className="flex items-center gap-2">
                  <Globe size={13} className="text-cyan-400" />
                  <span className="text-[11px] font-bold text-white uppercase tracking-wider">
                    Selected Scene
                  </span>
                </div>
                <span className="text-[10px] text-emerald-300 bg-emerald-500/15 border border-emerald-500/30 px-2 py-0.5 rounded-full font-bold">
                  {result.selected_scene.processing_level || "Sentinel-2 L2A"}
                </span>
              </div>

              <div className="grid grid-cols-2 gap-2 text-[11px]">
                <div className="bg-slate-900/90 border border-slate-800/80 p-2 rounded">
                  <span className="text-slate-500 block text-[9px] uppercase tracking-wider">Scene ID</span>
                  <span className="text-cyan-300 font-semibold truncate block" title={result.selected_scene.scene_id}>
                    {result.selected_scene.scene_id}
                  </span>
                </div>
                <div className="bg-slate-900/90 border border-slate-800/80 p-2 rounded">
                  <span className="text-slate-500 block text-[9px] uppercase tracking-wider">Acquisition Date</span>
                  <span className="text-slate-200 font-semibold block">
                    {result.selected_scene.acquisition_date ? new Date(result.selected_scene.acquisition_date).toLocaleDateString() : "Recent"}
                  </span>
                </div>
                <div className="bg-slate-900/90 border border-slate-800/80 p-2 rounded">
                  <span className="text-slate-500 block text-[9px] uppercase tracking-wider">Cloud Cover</span>
                  <span className="text-slate-200 font-semibold block">
                    {result.selected_scene.cloud_cover !== undefined ? `${result.selected_scene.cloud_cover.toFixed(1)}%` : "0.0%"}
                  </span>
                </div>
                <div className="bg-slate-900/90 border border-slate-800/80 p-2 rounded">
                  <span className="text-slate-500 block text-[9px] uppercase tracking-wider">Resolution (GSD)</span>
                  <span className="text-emerald-300 font-semibold block">
                    {result.selected_scene.spatial_resolution_m || 10}m Optical
                  </span>
                </div>
                <div className="bg-slate-900/90 border border-slate-800/80 p-2 rounded">
                  <span className="text-slate-500 block text-[9px] uppercase tracking-wider">Sensor</span>
                  <span className="text-slate-200 font-semibold block">
                    {result.selected_scene.platform || "Sentinel-2"} · MSI
                  </span>
                </div>
                <div className="bg-slate-900/90 border border-slate-800/80 p-2 rounded">
                  <span className="text-slate-500 block text-[9px] uppercase tracking-wider">CRS</span>
                  <span className="text-cyan-300 font-semibold block">
                    {result.google_maps_overlay?.crs || result.selected_scene.crs || "EPSG:4326"}
                  </span>
                </div>
              </div>

              {/* Distinguish Selected Scene from Available Scenes (Requirement 11) */}
              {result.scenes && result.scenes.length > 1 && (
                <div className="pt-1 border-t border-slate-800/60 text-[10px]">
                  <span className="text-slate-400 block mb-1">
                    Discovered Candidate Scenes ({result.scenes.length}):
                  </span>
                  <div className="flex gap-1.5 overflow-x-auto pb-1">
                    {result.scenes.map((s: DiscoveredSatelliteScene, sIdx: number) => {
                      const isSelected = s.scene_id === result.selected_scene?.scene_id;
                      return (
                        <div
                          key={sIdx}
                          className={`px-2 py-1 rounded border shrink-0 text-[10px] ${isSelected
                            ? "bg-cyan-500/25 border-cyan-400 text-cyan-200 font-bold shadow-sm"
                            : "bg-slate-900 border-slate-800 text-slate-400"
                            }`}
                        >
                          <div className="truncate max-w-[120px]">{s.scene_id.split("_")[0]} ({s.cloud_cover?.toFixed(0)}% cloud)</div>
                          <div className="text-[9px] text-slate-500">{isSelected ? "★ Primary Selected" : "Candidate"}</div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* 2. Counting Clickable Pills Banner (When user requested a count) */}
          {intel.isCountingQuery && enrichedDetections.length > 0 && (
            <div className="p-3 rounded-xl bg-slate-950/80 border border-cyan-500/25 space-y-1.5">
              <div className="flex items-center justify-between text-[11px] font-mono">
                <span className="text-cyan-300 font-bold uppercase tracking-wider text-[10px]">
                  {intel.targetNoun.toUpperCase()} VERIFIED INSTANCES
                </span>
                <span className="text-emerald-400 font-bold">
                  Count: {enrichedDetections.length}
                </span>
              </div>
              <div className="flex flex-wrap gap-1.5 pt-1">
                {enrichedDetections.map((d) => (
                  <button
                    key={d.index}
                    onClick={() => handleSelectDetection(d.index)}
                    className={`px-2 py-0.5 rounded font-mono text-xs font-bold transition-all cursor-pointer ${selectedDetIndex === d.index
                      ? "bg-cyan-400 text-slate-950 ring-2 ring-cyan-300 shadow-[0_0_10px_rgba(6,182,212,0.6)]"
                      : "bg-slate-900 border border-slate-800 text-slate-300 hover:border-cyan-400 hover:text-cyan-200"
                      }`}
                  >
                    #{d.tag}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* 3. Multi-Target Categories (When multiple entity types requested) */}
          {detectedCategories.length > 1 && (
            <div className="flex items-center gap-1.5 overflow-x-auto pb-1 text-xs font-mono">
              <span className="text-[10px] text-slate-500 uppercase mr-1">Filter:</span>
              <button
                onClick={() => {
                  setSelectedCategory("all");
                  setCurrentPage(1);
                }}
                className={`px-2.5 py-1 rounded text-[11px] font-bold border transition-colors cursor-pointer shrink-0 ${selectedCategory === "all"
                  ? "bg-cyan-500/25 border-cyan-400 text-cyan-200"
                  : "bg-slate-950 border-slate-800 text-slate-400 hover:text-slate-200"
                  }`}
              >
                All ({allCount})
              </button>
              {detectedCategories.map((cat, cIdx) => (
                <button
                  key={cIdx}
                  onClick={() => {
                    setSelectedCategory(cat.label);
                    setCurrentPage(1);
                  }}
                  className={`px-2.5 py-1 rounded text-[11px] font-bold border transition-colors cursor-pointer shrink-0 ${selectedCategory.toLowerCase() === cat.label.toLowerCase()
                    ? "bg-cyan-500/25 border-cyan-400 text-cyan-200"
                    : "bg-slate-950 border-slate-800 text-slate-400 hover:text-slate-200"
                    }`}
                >
                  {cat.label} ({cat.count})
                </button>
              ))}
            </div>
          )}

          {/* 4. Scene Description View (Pure Captioning Intent - Zero Detection Counter) */}
          {isCaptionOnly ? (
            <div className="flex-1 flex flex-col space-y-3">
              <div className="p-4 rounded-xl bg-slate-950/70 border border-slate-800/90 shadow-inner space-y-2">
                <div className="text-[10px] font-mono font-bold text-cyan-400 uppercase tracking-wider flex items-center gap-1.5">
                  <FileText size={12} />
                  <span>Scene Intelligence Overview</span>
                </div>
                <blockquote className="text-sm text-slate-100 font-medium leading-relaxed italic border-l-2 border-cyan-500/60 pl-3 py-1">
                  "{result.optical_caption || result.sar_caption || result.response_text || report?.summary || "Scene description generated."}"
                </blockquote>
              </div>

              {/* Scene Observations */}
              {intel.observations.length > 0 && (
                <div className="bg-slate-950/40 border border-slate-800/60 rounded-xl p-3.5 space-y-2 text-xs">
                  <div className="text-[10px] font-mono font-bold text-cyan-400 uppercase tracking-wider">
                    Key Scene Observations
                  </div>
                  <ul className="space-y-1.5 text-slate-300 text-xs font-sans">
                    {intel.observations.map((obs, oIdx) => (
                      <li key={oIdx} className="flex items-start gap-2">
                        <span className="text-cyan-500 mt-0.5">•</span>
                        <span>{obs}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          ) : (
            <>
              {/* Filter & Sort Controls */}
              <div className="flex flex-col gap-2 pt-1 border-t border-slate-800/80">
                <div className="flex items-center gap-1.5 overflow-x-auto pb-1 text-xs font-mono">
                  <button
                    onClick={() => {
                      setConfidenceFilter("all");
                      setCurrentPage(1);
                    }}
                    className={`px-2.5 py-1 rounded text-[11px] font-medium border transition-colors cursor-pointer shrink-0 ${confidenceFilter === "all"
                      ? "bg-cyan-500/25 border-cyan-400 text-cyan-200"
                      : "bg-slate-950 border-slate-800 text-slate-400 hover:text-slate-200 hover:border-slate-700"
                      }`}
                  >
                    All ({allCount})
                  </button>
                  <button
                    onClick={() => {
                      setConfidenceFilter("high");
                      setCurrentPage(1);
                    }}
                    className={`px-2.5 py-1 rounded text-[11px] font-medium border transition-colors cursor-pointer shrink-0 ${confidenceFilter === "high"
                      ? "bg-emerald-500/25 border-emerald-400 text-emerald-200"
                      : "bg-slate-950 border-slate-800 text-slate-400 hover:text-emerald-300 hover:border-emerald-500/30"
                      }`}
                  >
                    High ({highCount})
                  </button>
                  <button
                    onClick={() => {
                      setConfidenceFilter("medium");
                      setCurrentPage(1);
                    }}
                    className={`px-2.5 py-1 rounded text-[11px] font-medium border transition-colors cursor-pointer shrink-0 ${confidenceFilter === "medium"
                      ? "bg-cyan-500/25 border-cyan-400 text-cyan-200"
                      : "bg-slate-950 border-slate-800 text-slate-400 hover:text-cyan-300 hover:border-cyan-500/30"
                      }`}
                  >
                    Moderate ({medCount})
                  </button>
                  <button
                    onClick={() => {
                      setConfidenceFilter("low");
                      setCurrentPage(1);
                    }}
                    className={`px-2.5 py-1 rounded text-[11px] font-medium border transition-colors cursor-pointer shrink-0 ${confidenceFilter === "low"
                      ? "bg-amber-500/25 border-amber-400 text-amber-200"
                      : "bg-slate-950 border-slate-800 text-slate-400 hover:text-amber-300 hover:border-amber-500/30"
                      }`}
                  >
                    Low ({lowCount})
                  </button>
                </div>

                <div className="flex items-center justify-between text-xs font-mono text-slate-400">
                  <span className="text-[11px] flex items-center gap-1">
                    <SlidersHorizontal size={11} className="text-cyan-400" />
                    <span>Sort:</span>
                  </span>
                  <select
                    value={sortBy}
                    onChange={(e) => {
                      setSortBy(e.target.value as any);
                      setCurrentPage(1);
                    }}
                    className="bg-slate-950 border border-slate-800 text-slate-200 text-xs rounded px-2 py-1 font-mono focus:outline-none focus:border-cyan-500 cursor-pointer"
                  >
                    <option value="confidence_desc">Confidence (High → Low)</option>
                    <option value="confidence_asc">Confidence (Low → High)</option>
                    <option value="area_desc">Area (Largest First)</option>
                    <option value="id_asc">ID (#01 → #{enrichedDetections.length})</option>
                  </select>
                </div>
              </div>

              {/* Target Cards List */}
              <div className="flex-1 min-h-[280px] max-h-[440px] overflow-y-auto space-y-2 pr-1 custom-scrollbar">
                {paginatedDetections.length === 0 ? (
                  enrichedDetections.length === 0 ? (
                    <div className="h-full flex flex-col items-center justify-center text-center p-6 text-slate-300 text-xs font-mono bg-slate-950/60 rounded-xl border border-slate-800/90 space-y-2.5">
                      {detectionStatus === "MODEL_UNAVAILABLE" ? (
                        <>
                          <AlertCircle size={24} className="text-amber-400 mb-1" />
                          <p className="font-bold text-slate-100 text-sm">Detection Temporarily Unavailable</p>
                          <p className="text-slate-400 max-w-sm font-sans">
                            Specialist detection is temporarily unavailable for {primaryTargetLabel.toLowerCase()}.
                          </p>
                        </>
                      ) : detectionStatus === "LOCALIZATION_UNAVAILABLE" ? (
                        <>
                          <HelpCircle size={24} className="text-cyan-400 mb-1" />
                          <p className="font-bold text-slate-100 text-sm">Landscape Feature Detected</p>
                          <p className="text-slate-400 max-w-sm font-sans">
                            NexSpace analyzed the imagery, but could not reliably localize discrete bounding boxes for {primaryTargetLabel.toLowerCase()}.
                          </p>
                        </>
                      ) : (
                        <>
                          <CheckCircle2 size={24} className="text-slate-400 mb-1" />
                          <p className="font-bold text-slate-100 text-sm">No Matching Targets Identified</p>
                          <p className="text-slate-400 max-w-sm font-sans">
                            No matching {primaryTargetLabel.toLowerCase()} were identified in this imagery above confidence threshold.
                          </p>
                        </>
                      )}
                      {result.optical_caption && (
                        <div className="mt-2 p-3 rounded-lg bg-slate-900 border border-slate-800 text-left text-xs italic text-slate-200">
                          <span className="text-[10px] text-cyan-400 font-mono block not-italic uppercase mb-1">Scene Context:</span>
                          "{result.optical_caption}"
                        </div>
                      )}
                    </div>
                  ) : (
                    <div className="h-full flex flex-col items-center justify-center text-center p-6 text-slate-400 text-xs font-mono bg-slate-950/50 rounded-xl border border-slate-800/80">
                      <Filter size={20} className="text-slate-500 mb-2" />
                      <p>No targets match the active filter criteria.</p>
                      <button
                        onClick={() => {
                          setConfidenceFilter("all");
                          setSelectedCategory("all");
                          setCurrentPage(1);
                        }}
                        className="mt-2 text-cyan-400 underline hover:text-cyan-300 cursor-pointer"
                      >
                        Reset Filters
                      </button>
                    </div>
                  )
                ) : (
                  paginatedDetections.map((det) => {
                    const isSelected = selectedDetIndex === det.index;
                    const confInfo = getConfidenceInfo(det.score);
                    const isPolygon = det.geometryType === "polygon";

                    return (
                      <div
                        key={det.index}
                        onClick={() => handleSelectDetection(det.index)}
                        className={`p-2.5 rounded-xl border text-xs flex gap-3 items-center transition-all cursor-pointer ${isSelected
                          ? "bg-cyan-950/50 border-cyan-400 shadow-[0_0_12px_rgba(6,182,212,0.25)] ring-1 ring-cyan-400/50 text-white"
                          : "bg-slate-950/70 border-slate-800/90 hover:border-cyan-500/40 hover:bg-slate-900/80 text-slate-300"
                          }`}
                      >
                        {/* Vector Crop Thumbnail */}
                        <div className="w-14 h-14 rounded-lg border border-cyan-500/30 overflow-hidden bg-slate-950 shrink-0 relative shadow-inner flex items-center justify-center">
                          <svg
                            viewBox={`${det.box[0]} ${det.box[1]} ${det.width} ${det.height}`}
                            preserveAspectRatio="xMidYMid slice"
                            className="w-full h-full"
                          >
                            <image
                              href={sourceImage.dataUrl}
                              x="0"
                              y="0"
                              width="1000"
                              height="1000"
                              preserveAspectRatio="none"
                            />
                          </svg>
                          <span className="absolute bottom-0.5 right-0.5 text-[8px] font-mono bg-black/85 text-cyan-300 px-1 py-0.2 rounded font-bold">
                            #{det.tag}
                          </span>
                        </div>

                        {/* Target Information */}
                        <div className="flex-1 min-w-0 space-y-1 font-mono">
                          <div className="flex items-center justify-between gap-1">
                            <span className="font-bold flex items-center gap-1.5 truncate">
                              <span className={isSelected ? "text-cyan-300 font-bold" : "text-cyan-400"}>
                                TARGET {det.tag}
                              </span>
                              <span className="text-slate-200">· {det.cleanLabel}</span>
                            </span>
                            <span
                              className={`text-[10px] font-bold px-1.5 py-0.5 rounded border ${confInfo.bg} ${confInfo.border} ${confInfo.color}`}
                              title={`Model Score: ${det.score.toFixed(3)}`}
                            >
                              {confInfo.label.replace(" (Model Calibration)", "")} ({det.scorePercent}%)
                            </span>
                          </div>

                          <div className="text-[10px] text-slate-400 flex flex-wrap items-center gap-x-2.5 gap-y-0.5">
                            <span>
                              Geometry: <strong className="text-cyan-300">{isPolygon ? "Region Polygon" : "Discrete Object"}</strong>
                            </span>
                            <span>
                              Area: <strong className="text-slate-300">{det.areaPx.toLocaleString()} px²</strong>
                            </span>
                          </div>

                          {/* Geographic Coordinates & Google Maps Link */}
                          {det.coordinates ? (
                            <div className="flex items-center justify-between text-[10px] text-slate-400 pt-0.5 border-t border-slate-800/60">
                              <span className="text-cyan-300 flex items-center gap-1 font-mono">
                                <MapPin size={10} className="text-cyan-400 shrink-0" />
                                <span>{det.coordinates.formatted}</span>
                              </span>
                              {det.gmapsUrl && (
                                <a
                                  href={det.gmapsUrl}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  onClick={(e) => e.stopPropagation()}
                                  className="inline-flex items-center gap-1 text-[10px] text-sky-400 hover:text-sky-300 hover:underline cursor-pointer"
                                >
                                  <span>Open in Maps</span>
                                  <ExternalLink size={9} />
                                </a>
                              )}
                            </div>
                          ) : (
                            <div className="flex items-center justify-between text-[10px] text-slate-500 pt-0.5">
                              <span>📍 Pixel Referenced</span>
                              <span className="text-slate-400">{isSelected ? "Focused ✓" : "Click to Focus"}</span>
                            </div>
                          )}
                        </div>
                      </div>
                    );
                  })
                )}
              </div>

              {/* Pagination */}
              <div className="pt-2 border-t border-slate-800/80 flex items-center justify-between text-xs font-mono">
                <span className="text-[10px] text-slate-400">
                  Showing {startIdx}–{endIdx} of {sortedDetections.length} targets
                </span>
                <div className="flex items-center gap-1">
                  <button
                    disabled={activePage <= 1}
                    onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
                    className="px-2 py-1 rounded bg-slate-950 border border-slate-800 text-slate-400 hover:text-cyan-300 disabled:opacity-40 disabled:hover:text-slate-400 cursor-pointer disabled:cursor-not-allowed"
                    aria-label="Previous Page"
                  >
                    <ChevronLeft size={12} />
                  </button>
                  {Array.from({ length: totalPages }, (_, i) => i + 1).map((pg) => {
                    if (
                      totalPages > 5 &&
                      Math.abs(pg - activePage) > 1 &&
                      pg !== 1 &&
                      pg !== totalPages
                    ) {
                      if (pg === 2 || pg === totalPages - 1) {
                        return <span key={pg} className="text-slate-600 text-[10px] px-0.5">..</span>;
                      }
                      return null;
                    }
                    return (
                      <button
                        key={pg}
                        onClick={() => setCurrentPage(pg)}
                        className={`w-6 h-6 rounded text-[11px] font-mono font-medium border flex items-center justify-center transition-colors cursor-pointer ${activePage === pg
                          ? "bg-cyan-500/20 border-cyan-400 text-cyan-300 font-bold"
                          : "bg-slate-950 border-slate-800 text-slate-400 hover:text-slate-200"
                          }`}
                      >
                        {pg}
                      </button>
                    );
                  })}
                  <button
                    disabled={activePage >= totalPages}
                    onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
                    className="px-2 py-1 rounded bg-slate-950 border border-slate-800 text-slate-400 hover:text-cyan-300 disabled:opacity-40 disabled:hover:text-slate-400 cursor-pointer disabled:cursor-not-allowed"
                    aria-label="Next Page"
                  >
                    <ChevronRight size={12} />
                  </button>
                </div>
              </div>
            </>
          )}
        </div>
      </div>

      {/* 3. Auxiliary Intelligence Findings (VQA, Bi-Temporal Changes, Multimodal Optical + SAR) */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-2 border-t border-slate-800/80">
        {/* Visual Q&A Findings */}
        {vqaResults.length > 0 && (
          <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-4 space-y-2">
            <div className="text-xs font-semibold text-white flex items-center gap-2">
              <Waves size={15} className="text-cyan-400" />
              <span>Visual Feature Verification</span>
            </div>
            {vqaResults.map((v, i) => {
              const conf = getConfidenceInfo(v.confidence);
              return (
                <div key={i} className="bg-slate-950/60 border border-slate-800/80 rounded-lg p-2.5 space-y-1 text-xs font-sans">
                  <div className="text-slate-400 text-[11px] font-mono">Inquiry: {v.question}</div>
                  <div className="flex items-center justify-between">
                    <span className="text-slate-100 font-medium">
                      Finding: <strong className="text-cyan-300 capitalize">{v.answer}</strong>
                    </span>
                    <span className={`text-[10px] font-mono px-1.5 py-0.5 rounded ${conf.bg} ${conf.color}`}>
                      {conf.label.replace(" (Model Calibration)", "")}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {/* Bi-Temporal Semantic Change Assessment */}
        {result.change_analysis && (
          <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-4 space-y-2 md:col-span-2">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-white flex items-center gap-2">
                <RotateCcw size={15} className="text-amber-400" />
                <span>Bi-Temporal Surface Changes</span>
              </span>
              {result.change_analysis.change_category && (
                <span className="text-[10px] font-mono px-2 py-0.5 rounded font-bold bg-cyan-500/20 text-cyan-300 border border-cyan-500/30">
                  {result.change_analysis.change_category}
                </span>
              )}
            </div>
            <p className="text-xs text-slate-200 font-sans leading-relaxed">
              {result.change_analysis.what_changed || result.change_analysis.summary}
            </p>
            {result.change_analysis.before_interpretation && result.change_analysis.after_interpretation && (
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 pt-1">
                <div className="bg-slate-950/70 border border-slate-800/80 rounded p-2 text-xs font-sans text-slate-300">
                  <span className="text-[10px] font-mono text-cyan-400 block mb-0.5">BASELINE OBSERVATION:</span>
                  {result.change_analysis.before_interpretation}
                </div>
                <div className="bg-slate-950/70 border border-slate-800/80 rounded p-2 text-xs font-sans text-slate-300">
                  <span className="text-[10px] font-mono text-amber-400 block mb-0.5">CURRENT OBSERVATION:</span>
                  {result.change_analysis.after_interpretation}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Multimodal Optical + SAR Reasoning */}
        {result.optical_sar_analysis && (
          <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-4 space-y-3 md:col-span-2">
            <div className="flex items-center justify-between">
              <div className="text-xs font-semibold text-white flex items-center gap-2">
                <Leaf size={15} className="text-purple-400" />
                <span>Multimodal Optical &amp; Radar Synthesis</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-[10px] font-mono text-cyan-400 bg-cyan-950/60 px-2 py-0.5 rounded border border-cyan-800/60">
                  {result.optical_sar_analysis.fusion_operation || "Feature Concatenation"} ({result.optical_sar_analysis.fused_feature_dim || 1536}-d)
                </span>
                <span className="text-[10px] font-mono text-slate-400 bg-slate-800/80 px-2 py-0.5 rounded border border-slate-700">
                  Multi-Sensor
                </span>
              </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 pt-1">
              <div className="bg-slate-950/70 border border-cyan-500/20 rounded p-2.5 text-xs font-sans space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-[10px] font-mono text-cyan-400 uppercase font-bold">OPTICAL SPECTRAL SIGNAL</span>
                  <span className="text-[9px] font-mono text-slate-400">Sentinel-2 L2A</span>
                </div>
                <p className="text-slate-300 leading-relaxed text-[11px]">
                  {result.optical_sar_analysis.optical_evidence || "Visible surface structures and spectral signatures analyzed."}
                </p>
                {result.optical_sar_analysis.optical_only_metrics && Object.keys(result.optical_sar_analysis.optical_only_metrics).length > 0 && (
                  <div className="grid grid-cols-3 gap-1 pt-1 border-t border-slate-800/60 font-mono text-[10px]">
                    {result.optical_sar_analysis.optical_only_metrics.vegetation_fraction !== undefined && (
                      <div className="bg-slate-900/80 p-1 rounded">
                        <span className="text-slate-500 block text-[9px]">Veg Fraction</span>
                        <span className="text-emerald-400 font-bold">{String(result.optical_sar_analysis.optical_only_metrics.vegetation_fraction)}</span>
                      </div>
                    )}
                    {result.optical_sar_analysis.optical_only_metrics.water_fraction !== undefined && (
                      <div className="bg-slate-900/80 p-1 rounded">
                        <span className="text-slate-500 block text-[9px]">Water Fraction</span>
                        <span className="text-blue-400 font-bold">{String(result.optical_sar_analysis.optical_only_metrics.water_fraction)}</span>
                      </div>
                    )}
                    {result.optical_sar_analysis.optical_only_metrics.edge_density !== undefined && (
                      <div className="bg-slate-900/80 p-1 rounded">
                        <span className="text-slate-500 block text-[9px]">Edge Density</span>
                        <span className="text-amber-400 font-bold">{String(result.optical_sar_analysis.optical_only_metrics.edge_density)}</span>
                      </div>
                    )}
                  </div>
                )}
              </div>

              <div className="bg-slate-950/70 border border-purple-500/20 rounded p-2.5 text-xs font-sans space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-[10px] font-mono text-purple-400 uppercase font-bold">SAR MICROWAVE REFLECTION</span>
                  <span className="text-[9px] font-mono text-slate-400">Sentinel-1 GRD</span>
                </div>
                <p className="text-slate-300 leading-relaxed text-[11px]">
                  {result.optical_sar_analysis.sar_evidence || "Microwave backscatter roughness and dielectric properties analyzed."}
                </p>
                {result.optical_sar_analysis.sar_only_metrics && Object.keys(result.optical_sar_analysis.sar_only_metrics).length > 0 && (
                  <div className="grid grid-cols-3 gap-1 pt-1 border-t border-slate-800/60 font-mono text-[10px]">
                    {result.optical_sar_analysis.sar_only_metrics.mean_backscatter !== undefined && (
                      <div className="bg-slate-900/80 p-1 rounded">
                        <span className="text-slate-500 block text-[9px]">Mean Backscatter</span>
                        <span className="text-purple-300 font-bold">{String(result.optical_sar_analysis.sar_only_metrics.mean_backscatter)}</span>
                      </div>
                    )}
                    {result.optical_sar_analysis.sar_only_metrics.dynamic_range_db !== undefined && (
                      <div className="bg-slate-900/80 p-1 rounded">
                        <span className="text-slate-500 block text-[9px]">Dyn Range (dB)</span>
                        <span className="text-purple-400 font-bold">{String(result.optical_sar_analysis.sar_only_metrics.dynamic_range_db)}</span>
                      </div>
                    )}
                    {result.optical_sar_analysis.sar_only_metrics.high_backscatter_ratio !== undefined && (
                      <div className="bg-slate-900/80 p-1 rounded">
                        <span className="text-slate-500 block text-[9px]">Strong Bounce</span>
                        <span className="text-amber-300 font-bold">{String(result.optical_sar_analysis.sar_only_metrics.high_backscatter_ratio)}</span>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>

            <div className="bg-cyan-950/30 border border-cyan-500/30 rounded p-2.5 text-xs font-sans space-y-1">
              <span className="text-[10px] font-mono text-emerald-400 uppercase font-bold block">SYNTHESIZED CONCLUSION</span>
              <p className="text-slate-200 font-medium leading-relaxed">
                {result.optical_sar_analysis.fused_conclusion || result.optical_sar_analysis.correlation_summary || "Multi-sensor cross-validation confirmed."}
              </p>
            </div>

            {result.optical_sar_analysis.limitations && result.optical_sar_analysis.limitations.length > 0 && (
              <div className="bg-amber-950/20 border border-amber-500/30 rounded p-2 text-[10px] font-mono text-amber-300/90 space-y-0.5">
                <span className="font-bold uppercase tracking-wider block text-[9px] text-amber-400">Model Transparency &amp; Baseline Limitations:</span>
                <ul className="list-disc list-inside space-y-0.5">
                  {result.optical_sar_analysis.limitations.map((lim, idx) => (
                    <li key={idx}>{lim}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}
      </div>

      {/* 4. Investigation Summary Card */}
      <div className="p-4 rounded-xl bg-slate-950/90 border border-cyan-500/25 shadow-lg flex flex-col md:flex-row items-start md:items-center justify-between gap-4 font-mono text-xs">
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <ShieldCheck size={14} className="text-cyan-400" />
            <span className="text-[11px] font-bold text-white uppercase tracking-wider">
              INVESTIGATION SUMMARY
            </span>
          </div>
          <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-slate-300 text-[11px]">
            <span>Query: <strong className="text-white font-sans">"{intel.query}"</strong></span>
            <span>•</span>
            <span>Result: <strong className="text-cyan-300">{intel.headline}</strong></span>
            <span>•</span>
            <span>Location: <strong className="text-slate-200">{intel.locationInfo.resolvedName}</strong></span>
            <span>•</span>
            <span>Evidence: <strong className="text-emerald-400">{enrichedDetections.length} targets</strong></span>
          </div>
        </div>

        <div className="flex items-center gap-2 shrink-0">
          <button
            onClick={() => setViewMode("evidence")}
            className={`px-3 py-1.5 rounded-lg border text-xs font-bold transition-all cursor-pointer ${viewMode === "evidence"
              ? "bg-cyan-500 text-slate-950 border-cyan-400 shadow-[0_0_10px_rgba(6,182,212,0.4)]"
              : "bg-slate-900 border-slate-800 text-slate-300 hover:border-cyan-500/50"
              }`}
          >
            View Evidence
          </button>
          <button
            onClick={() => setViewMode("map")}
            className={`px-3 py-1.5 rounded-lg border text-xs font-bold transition-all cursor-pointer ${viewMode === "map"
              ? "bg-cyan-500 text-slate-950 border-cyan-400 shadow-[0_0_10px_rgba(6,182,212,0.4)]"
              : "bg-slate-900 border-slate-800 text-slate-300 hover:border-cyan-500/50"
              }`}
          >
            View Map
          </button>
          <button
            onClick={() => setShowTechnical(!showTechnical)}
            className="px-3 py-1.5 rounded-lg border border-slate-800 bg-slate-900 text-slate-400 hover:text-cyan-300 text-xs font-bold transition-all cursor-pointer"
          >
            Technical Details {showTechnical ? "▲" : "▼"}
          </button>
        </div>
      </div>

      {/* 5. Collapsible Technical Details Drawer (Collapsed by default for transparency) */}
      {showTechnical && (
        <div className="space-y-3 bg-[#050b14] border border-slate-800/90 rounded-xl p-4 text-xs font-mono">
          <div className="flex items-center justify-between pb-2 border-b border-slate-800/80">
            <span className="text-cyan-400 font-bold uppercase tracking-wider text-[11px] flex items-center gap-1.5">
              <Terminal size={12} />
              <span>Technical Diagnostics &amp; Telemetry</span>
            </span>
            <span className="text-[10px] text-slate-500">Execution Provenance</span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-[11px]">
            <div>
              <span className="text-slate-500 uppercase text-[10px] block">Specialist Routing:</span>
              <span className="text-cyan-300 font-semibold">{intel.technicalDetails.routingTools.join(", ") || "Autonomous"}</span>
            </div>
            <div>
              <span className="text-slate-500 uppercase text-[10px] block">Confidence Source:</span>
              <span className="text-slate-300">{intel.technicalDetails.confidenceSource}</span>
            </div>
            <div>
              <span className="text-slate-500 uppercase text-[10px] block">Sensor / Resolution:</span>
              <span className="text-slate-300">{intel.technicalDetails.sensorType} · {intel.locationInfo.resolution || "Unavailable"}</span>
            </div>
          </div>

          {trace.length > 0 && (
            <div className="space-y-1 pt-2 border-t border-slate-800/60">
              <div className="text-[10px] text-slate-500 uppercase">12-Stage Lifecycle Trace:</div>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-1.5">
                {trace.map((st, i) => (
                  <div key={i} className="bg-slate-950 border border-slate-800/80 p-1.5 rounded text-[10px]">
                    <div className="text-slate-300 truncate font-semibold">{st.stage}</div>
                    <div className="flex justify-between text-slate-500">
                      <span className="text-emerald-400">{st.status}</span>
                      <span>{st.duration_ms.toFixed(1)} ms</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {result.spatial_summary && (
            <div className="pt-2 border-t border-slate-800/60 text-[10px]">
              <span className="text-slate-500 uppercase">Geospatial Projection: </span>
              <span className="text-slate-300">
                {result.spatial_summary.geospatial_available
                  ? `CRS: ${result.spatial_summary.crs || "EPSG:4326"} · World Bounds: Active`
                  : "Location information is estimated from scene metadata."}
              </span>
            </div>
          )}
        </div>
      )}

      {/* 6. Navigation Actions */}
      <div className="flex flex-wrap gap-3 pt-2 border-t border-slate-800">
        <Link
          href="/evidence"
          className="flex items-center gap-1.5 bg-cyan-500/20 hover:bg-cyan-500/30 border border-cyan-500/40 text-cyan-300 rounded-lg px-3.5 py-2 text-xs font-mono font-semibold transition-all cursor-pointer"
        >
          <Target size={13} />
          <span>Open in Evidence Viewer →</span>
        </Link>
        <Link
          href="/execution"
          className="flex items-center gap-1.5 bg-[#091522] hover:bg-slate-800 border border-slate-700 text-slate-300 hover:text-cyan-300 rounded-lg px-3.5 py-2 text-xs font-mono font-semibold transition-all cursor-pointer"
        >
          <Clock size={13} />
          <span>View Execution Log →</span>
        </Link>
      </div>
    </div>
  );
});

// ----------------------------------------------------------------
// Query Page Main Component
// ----------------------------------------------------------------
export default function QueryPage() {
  const [query, setQuery] = useState("Describe this image and locate the buildings");
  const [canonicalSource, setCanonicalSource] = useState<CanonicalSourceImage>(() => {
    return getActiveSourceImage() || DEFAULT_DEMO_SOURCE;
  });
  const [sarImage, setSarImage] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [loadingPhase, setLoadingPhase] = useState("ROUTING QUERY VIA AGENT CONTROLLER...");
  const [apiResult, setApiResult] = useState<NexSpaceQueryResponse | null>(() => {
    const inv = getCurrentInvestigation();
    return inv?.response || null;
  });
  const [capabilities, setCapabilities] = useState<CapabilitiesRecord | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [showDiscoveryModal, setShowDiscoveryModal] = useState<boolean>(false);
  const [selectedSatelliteScene, setSelectedSatelliteScene] = useState<DiscoveredSatelliteScene | null>(null);

  const abortControllerRef = useRef<AbortController | null>(null);

  // Sync canonical source to storage and purge stale investigation
  const updateSourceImage = useCallback((src: CanonicalSourceImage) => {
    setCanonicalSource(src);
    setApiResult(null);
    setErrorMsg(null);
    setActiveSourceImage(src);
    clearCurrentInvestigation(true);
  }, []);

  const handleOpenDiscovery = useCallback(() => {
    setShowDiscoveryModal(true);
  }, []);

  const handleSelectSarImage = useCallback((img: string | null) => {
    setSarImage(img);
  }, []);

  const handleSceneSelectedFromDiscovery = (
    scene: DiscoveredSatelliteScene,
    retrievedImageBase64: string,
    investigationResult?: NexSpaceQueryResponse | null
  ) => {
    const satSource: CanonicalSourceImage = {
      id: `src-sat-${scene.scene_id}`,
      filename: `${scene.scene_id}.tif`,
      mediaType: "image/tiff",
      dataUrl: retrievedImageBase64,
      source: "upload",
    };
    setSelectedSatelliteScene(scene);
    setCanonicalSource(satSource);
    setActiveSourceImage(satSource);

    if (investigationResult) {
      setApiResult(investigationResult);
      setErrorMsg(null);
      const invState: CanonicalInvestigationState = {
        investigation_id: `inv-${Date.now()}`,
        query: query || `Analyze ${scene.platform} scene`,
        inputMode: "SATELLITE_DISCOVERY",
        source_image: satSource,
        sar_image: null,
        change_image_b: null,
        timestamp: new Date().toISOString(),
        response: investigationResult,
        selectedSatelliteScene: scene,
      };
      setCurrentInvestigation(invState);
    } else {
      handleExecuteWith(query, satSource, null);
    }
  };

  const fetchHealth = () => {
    fetch("/api/health")
      .then((r) => r.json())
      .then((d) => {
        if (d && d.capabilities) {
          setCapabilities(d.capabilities);
        }
        setErrorMsg(null);
      })
      .catch(() => {
        setErrorMsg(null);
      });
  };

  useEffect(() => {
    fetchHealth();
    // Re-sync with storage if present without overriding existing image
    const active = getActiveSourceImage(false);
    if (active) {
      setCanonicalSource(active);
    } else {
      setActiveSourceImage(DEFAULT_DEMO_SOURCE);
    }

    const currentInv = getCurrentInvestigation();
    if (currentInv?.response) {
      setApiResult(currentInv.response);
      if (currentInv.query) setQuery(currentInv.query);
      if (currentInv.source_image) setCanonicalSource(currentInv.source_image);
      if (currentInv.sar_image) setSarImage(currentInv.sar_image);
    }

    // Check if query was passed via URL parameter (e.g. from NEXA Assistant)
    if (typeof window !== "undefined") {
      const urlParams = new URLSearchParams(window.location.search);
      const qParam = urlParams.get("q");
      if (qParam && qParam.trim()) {
        setQuery(qParam.trim());
      }
    }
  }, []);

  const phases = [
    "CLASSIFYING INTENT & ROUTING...",
    "EXTRACTING SPATIAL & TENSOR FEATURES...",
    "RUNNING SPECIALIST VISION-LANGUAGE MODELS...",
    "EXTRACTING & VALIDATING EVIDENCE NODES...",
    "SYNTHESIZING INVESTIGATION REPORT...",
  ];

  const handleExecuteWith = async (
    qText: string,
    srcImage: CanonicalSourceImage,
    sImg: string | null
  ) => {
    if (!qText.trim() || loading) return;

    // Abort previous in-flight query request if still active
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    const abortController = new AbortController();
    abortControllerRef.current = abortController;

    setLoading(true);
    setErrorMsg(null);
    setApiResult(null);

    let activeSource = srcImage;

    const lowerQ = qText.toLowerCase();
    const isCurrentCustom = srcImage.source === "upload" || srcImage.source === "satellite" || (srcImage.id && srcImage.id.startsWith("sat-"));
    const KNOWN_LOCS = ["lucknow", "gomti", "mumbai", "marine", "delhi", "connaught", "bengaluru", "bangalore", "taj mahal", "agra", "jaipur", "ayodhya", "kanpur", "noida", "dubai", "pune", "ahmedabad", "varanasi", "chennai", "kolkata", "wayanad"];
    const hasExplicitNewLoc = KNOWN_LOCS.some(loc => lowerQ.includes(loc)) || /(?:in|at|around|near|over)\s+[a-zA-Z]{3,}/i.test(qText);

    // Only acquire new satellite image if query explicitly mentions a new geographic location
    if (hasExplicitNewLoc && (!isCurrentCustom || !srcImage.filename.toLowerCase().includes(lowerQ.slice(0, 8)))) {
      try {
        setLoadingPhase("GEOCODING & ACQUIRING COPERNICUS SATELLITE AOI...");
        const satRes = await fetch("/api/satellite/acquire", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          signal: abortController.signal,
          body: JSON.stringify({ query: qText })
        });
        if (satRes.ok) {
          const satData = await satRes.json();
          if (satData.sourceImage) {
            activeSource = satData.sourceImage;
            setCanonicalSource(satData.sourceImage);
            setActiveSourceImage(satData.sourceImage);
          }
        }
      } catch {
        // Continue with current source if satellite endpoint fails
      }
    }

    let phaseIndex = 0;
    setLoadingPhase(phases[0]);
    const interval = setInterval(() => {
      phaseIndex += 1;
      if (phaseIndex < phases.length) {
        setLoadingPhase(phases[phaseIndex]);
      }
    }, 600);

    try {
      // Only send optical_image if it is a valid base64 data URI (never a relative /api/... or http URL)
      const isDataUri = typeof srcImage.dataUrl === "string" && srcImage.dataUrl.startsWith("data:");

      // Determine if query is searching for a location, coordinate, or satellite imagery
      const hasCoordinates = /(-?\d{1,2}\.\d+,\s*-?\d{1,3}\.\d+)/.test(qText);
      const hasSatelliteTerms = /\b(satellite|imagery|sentinel|earth|orbit|raster|stac)\b/i.test(qText);
      const hasDetectionIntent = /\b(detect|find|locate|identify|show|count|where|spot|pinpoint|segment|track|classify|ships?|boats?|vessels?|roads?|highways?|bridges?|aircraft|airplanes?|buildings?|vehicles?|anomal|spill)\b/i.test(qText);

      // If user has an active image loaded AND has detection intent (e.g. "find ships", "find roads"), ALWAYS send optical_image!
      const isNewLocationSearch = (hasCoordinates || hasSatelliteTerms) && !hasDetectionIntent;
      const shouldOmitOptical = isNewLocationSearch || !isDataUri;

      const payloadOpticalImage = (!shouldOmitOptical && isDataUri) ? srcImage.dataUrl : undefined;

      const res = await fetch("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: abortController.signal,
        body: JSON.stringify({
          query: qText,
          optical_image: payloadOpticalImage,
          sar_image: sImg || undefined,
        }),
      });

      if (!res.ok) {
        throw new Error(`Backend HTTP error ${res.status}`);
      }

      const data = await res.json();
      setApiResult(data);

      // If backend discovered/retrieved a satellite scene, update source image with real raster
      let activeImage = srcImage;
      if (data.selected_scene || data.satellite_image_url || data.google_maps_overlay?.image_url) {
        const satRasterUrl =
          data.satellite_image_url ||
          data.google_maps_overlay?.image_url ||
          data.satellite_image_base64 ||
          data.google_maps_overlay?.image_base64;

        if (satRasterUrl) {
          activeImage = {
            id: `src-sat-${data.selected_scene?.scene_id || Date.now()}`,
            filename: `${data.selected_scene?.scene_id || "sentinel2"}_rgb.jpg`,
            mediaType: "image/jpeg",
            dataUrl: satRasterUrl,
            source: "upload",
            caption: data.geographic_context?.display_name || data.selected_scene?.platform || "Sentinel-2 Satellite Image",
          };
          setCanonicalSource(activeImage);
          setActiveSourceImage(activeImage);
        }
      }

      // Persist canonical investigation state across pages
      const invState: CanonicalInvestigationState = {
        investigation_id: data.request_id || data.investigation_id || `INV-${Date.now()}`,
        query: qText,
        source_image: activeImage,
        sar_image: sImg,
        selectedTargetId: null,
        timestamp: new Date().toISOString(),
        response: data,
        selectedSatelliteScene: data.selected_scene || null,
      };
      setCurrentInvestigation(invState);
      setActiveSourceImage(activeImage);
    } catch (err: unknown) {
      if (err instanceof DOMException && err.name === "AbortError") {
        return; // Clean cancellation
      }
      console.error("[QueryPage] Error executing query:", err);
      const msg = err instanceof Error ? err.message : "Failed to execute live investigation request.";
      setErrorMsg(msg);
    } finally {
      clearInterval(interval);
      setLoading(false);
    }
  };

  const handleExecute = () => {
    handleExecuteWith(query, canonicalSource, sarImage);
  };

  const handleReset = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    setQuery("");
    setApiResult(null);
    clearCurrentInvestigation(false);
    setCanonicalSource(DEFAULT_DEMO_SOURCE);
    setActiveSourceImage(DEFAULT_DEMO_SOURCE);
    setSarImage(null);
    setErrorMsg(null);
  };

  return (
    <div className="flex flex-col lg:flex-row min-h-screen bg-[#0a1420] text-white font-sans overflow-x-hidden">
      <Sidebar />

      <div className="flex-1 flex flex-col min-w-0">
        <PageHeader capabilities={capabilities} />

        <main className="flex-1 p-4 sm:p-6 lg:p-8 overflow-y-auto space-y-6">
          {/* ASK NEXSPACE Section */}
          <div className="bg-[#08121e]/90 border border-slate-800/90 rounded-xl p-4 sm:p-5 space-y-3.5 shadow-md backdrop-blur-md">
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 pb-1 border-b border-slate-800/60">
              <div>
                <div className="flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
                  <h2 className="text-sm font-bold text-white font-mono tracking-wide uppercase">
                    Ask NexSpace
                  </h2>
                  <span className="text-[10px] text-cyan-400/90 font-mono px-2 py-0.5 rounded bg-cyan-500/10 border border-cyan-500/20">
                    Open-Vocabulary Geospatial AI
                  </span>
                </div>
                <p className="text-xs text-slate-400 font-sans mt-0.5">
                  Ask questions about your imagery in natural language. Describe, locate, compare, or analyze targets.
                </p>
              </div>

              <div className="flex items-center gap-2">
                <button
                  onClick={handleReset}
                  disabled={loading}
                  className="flex items-center gap-1.5 text-[11px] font-mono text-slate-400 hover:text-rose-300 bg-slate-900/80 hover:bg-rose-500/10 border border-slate-800 hover:border-rose-500/30 px-2.5 py-1 rounded-lg transition-all cursor-pointer shrink-0"
                >
                  <RotateCcw size={11} />
                  <span>Reset Query</span>
                </button>
              </div>
            </div>

            {/* Main Query Bar */}
            <QueryBar
              value={query}
              onChange={(v) => setQuery(v)}
              onExecute={handleExecute}
              loading={loading}
              loadingPhase={loadingPhase}
            />

            {/* Example Query Chips */}
            <ExampleQueryChips onSelect={(s) => setQuery(s)} />
          </div>

          {/* Backend Error Alert & Retry */}
          {errorMsg && (
            <div className="bg-rose-500/10 border border-rose-500/30 p-3.5 rounded-xl flex items-center justify-between gap-3 text-rose-300 text-xs font-mono">
              <div className="flex items-center gap-2">
                <AlertTriangle size={15} className="text-rose-400 shrink-0" />
                <span>{errorMsg}</span>
              </div>
              <button
                onClick={fetchHealth}
                className="bg-rose-500/20 hover:bg-rose-500/30 border border-rose-500/40 text-rose-200 px-3 py-1 rounded text-xs font-bold transition-all cursor-pointer shrink-0"
              >
                Retry Connection
              </button>
            </div>
          )}

          {/* Raster Image Selector */}
          <ImageSelector
            canonicalSource={canonicalSource}
            onSelectSource={updateSourceImage}
            sarImage={sarImage}
            onSelectSarImage={handleSelectSarImage}
            onOpenDiscovery={handleOpenDiscovery}
          />

          {/* Preview Image Viewport (Before analysis) */}
          {!apiResult && canonicalSource.dataUrl && (
            <div className="space-y-2">
              <div className="flex items-center justify-between text-xs font-mono text-slate-400">
                <span className="flex items-center gap-1.5">
                  <Target size={13} className="text-cyan-400" />
                  <span>Target Image Viewport ({canonicalSource.filename})</span>
                </span>
                <span>Ready for Analysis</span>
              </div>
              <GroundingVisualOverlay
                imageSrc={canonicalSource.dataUrl}
              />
            </div>
          )}

          {/* Scan Results Panel */}
          {apiResult && (
            <ScanResultsPanel
              result={apiResult}
              sourceImage={canonicalSource}
              queryText={apiResult.query || query}
            />
          )}
        </main>
      </div>

      {/* Satellite Imagery Discovery & Catalog Modal */}
      <SatelliteDiscoveryModal
        isOpen={showDiscoveryModal}
        onClose={() => setShowDiscoveryModal(false)}
        onSceneSelected={handleSceneSelectedFromDiscovery}
        initialQuery={query}
      />
    </div>
  );
}