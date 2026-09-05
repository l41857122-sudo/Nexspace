"use client";

import { useState, useEffect, useMemo } from "react";
import Link from "next/link";
import {
  Bell,
  HelpCircle,
  ChevronRight,
  Building2,
  Crosshair,
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Sparkles,
  ArrowRight,
  ShieldCheck,
  Search,
  Focus,
  Layers,
  FileText,
} from "lucide-react";
import Sidebar from "./Sidebar";
import type { CanonicalSourceImage, CanonicalInvestigationState } from "../types/nexspace";
import { SAMPLE_OPTICAL_PORT } from "../utils/sampleImages";
import { normalizeBox } from "./QueryPage";
import {
  getCurrentInvestigation,
  getActiveSourceImage,
  updateSelectedTarget,
  DEFAULT_DEMO_SOURCE,
} from "../utils/investigationStorage";

// ---------------------------------------------
// Top bar
// ---------------------------------------------
function TopBar({
  investigation,
  sourceImage,
}: {
  investigation: CanonicalInvestigationState | null;
  sourceImage: CanonicalSourceImage;
}) {
  const isUpload = sourceImage.source === "upload";

  return (
    <header className="flex flex-col sm:flex-row sm:items-center justify-between px-4 sm:px-6 lg:px-8 py-4 border-b border-slate-800/80 bg-[#09131f]/60 backdrop-blur-md gap-3">
      <div className="flex flex-wrap items-center gap-1.5 text-sm">
        <span className="text-white font-bold tracking-tight">NexSpace</span>
        <ChevronRight size={12} className="text-slate-600" />
        <span className="text-slate-400">Scan Results</span>
        <ChevronRight size={12} className="text-slate-600" />
        <span className="text-cyan-400 font-mono text-xs">
          {investigation?.investigation_id || "INV-ACTIVE"}
        </span>
        <span className={`ml-2 text-[10px] font-mono px-2 py-0.5 rounded-full border ${
          isUpload
            ? "text-emerald-300 bg-emerald-500/15 border-emerald-500/30"
            : "text-blue-300 bg-blue-500/15 border-blue-500/30"
        }`}>
          {sourceImage.filename} ({isUpload ? "Uploaded by user" : "Demo tile"})
        </span>
      </div>
      <div className="flex items-center gap-2">
        <button
          className="p-2 rounded-lg bg-slate-900/60 border border-slate-800 text-slate-400 hover:text-cyan-300 hover:border-cyan-500/30 transition-all duration-180 cursor-pointer shadow-sm"
          aria-label="Notifications"
        >
          <Bell size={14} />
        </button>
        <button
          className="p-2 rounded-lg bg-slate-900/60 border border-slate-800 text-slate-400 hover:text-cyan-300 hover:border-cyan-500/30 transition-all duration-180 cursor-pointer shadow-sm"
          aria-label="Help"
        >
          <HelpCircle size={14} />
        </button>
      </div>
    </header>
  );
}

// ---------------------------------------------
// Interactive Map & Viewport Area
// ---------------------------------------------
function MapArea({
  investigation,
  sourceImage,
  normalizedDetections,
  selectedTargetId,
  onSelectTarget,
  isFocused,
  onToggleFocus,
}: {
  investigation: CanonicalInvestigationState | null;
  sourceImage: CanonicalSourceImage;
  normalizedDetections: Array<{ id: string; evidenceId: string; label: string; score: number | null; box: [number, number, number, number] }>;
  selectedTargetId: string | null;
  onSelectTarget: (id: string | null) => void;
  isFocused: boolean;
  onToggleFocus: () => void;
}) {
  const [zoom, setZoom] = useState(1);

  const selectedTarget = useMemo(() => {
    return normalizedDetections.find((d) => d.id === selectedTargetId) || null;
  }, [normalizedDetections, selectedTargetId]);

  // Calculate transform-origin based on selected target box center
  const transformOrigin = useMemo(() => {
    if (!selectedTarget || !isFocused) return "center center";
    const [xmin, ymin, xmax, ymax] = selectedTarget.box;
    const cx = ((xmin + xmax) / 2 / 10).toFixed(1);
    const cy = ((ymin + ymax) / 2 / 10).toFixed(1);
    return `${cx}% ${cy}%`;
  }, [selectedTarget, isFocused]);

  const effectiveZoom = isFocused ? Math.max(zoom, 2.0) : zoom;

  const geoMeta = investigation?.response?.geospatial_metadata;
  const coordText = geoMeta?.geospatial_available && geoMeta.crs
    ? `CRS: ${geoMeta.crs} · RESOLUTION: ${geoMeta.resolution?.x || 10}m/px`
    : `SOURCE: ${sourceImage.filename} · SPATIAL FRAME: 512x512 PIXEL COORDINATES`;

  return (
    <div className="flex-1 relative bg-[#07111c] overflow-hidden min-h-[480px] lg:min-h-0 flex flex-col justify-between p-4 select-none">
      {/* Background HUD Grid */}
      <div
        className="absolute inset-0 opacity-15 pointer-events-none"
        style={{
          backgroundImage:
            "linear-gradient(rgba(56,189,248,0.2) 1px, transparent 1px), linear-gradient(90deg, rgba(56,189,248,0.2) 1px, transparent 1px)",
          backgroundSize: "32px 32px",
        }}
      />

      {/* Top HUD Telemetry Banner */}
      <div className="relative z-20 flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2 bg-[#091522]/90 backdrop-blur-md px-3 py-1.5 rounded-lg border border-slate-800 font-mono text-[10px] text-cyan-300 shadow-sm">
          <ShieldCheck size={13} className="text-cyan-400" />
          <span className="font-semibold">ACTIVE INVESTIGATION VIEWPORT</span>
          <span className="text-slate-600">|</span>
          <span className="text-slate-400">TARGETS: {normalizedDetections.length} DETECTED</span>
          {selectedTarget && (
            <>
              <span className="text-slate-600">|</span>
              <span className="text-emerald-300 font-bold">SELECTED: {selectedTarget.id}</span>
            </>
          )}
        </div>

        {/* Viewport Actions */}
        <div className="flex items-center gap-1.5 bg-[#091522]/90 backdrop-blur-md p-1 rounded-lg border border-slate-800 font-mono text-[10px] text-slate-300">
          {selectedTarget && (
            <button
              onClick={onToggleFocus}
              className={`flex items-center gap-1 px-2 py-1 rounded transition-all cursor-pointer font-bold ${
                isFocused
                  ? "bg-cyan-500 text-[#091522]"
                  : "bg-cyan-500/20 text-cyan-300 hover:bg-cyan-500/30"
              }`}
              title="Focus Target"
            >
              <Focus size={12} />
              <span>{isFocused ? "Unfocus" : "Focus Target"}</span>
            </button>
          )}

          <button
            onClick={() => setZoom((z) => Math.min(Number((z + 0.25).toFixed(2)), 3.0))}
            className="p-1 hover:bg-cyan-500/10 hover:text-cyan-300 rounded transition-colors cursor-pointer"
            title="Zoom In"
          >
            <ZoomIn size={13} />
          </button>
          <span className="text-cyan-400 px-1 font-semibold">{effectiveZoom.toFixed(2)}×</span>
          <button
            onClick={() => setZoom((z) => Math.max(Number((z - 0.25).toFixed(2)), 0.75))}
            className="p-1 hover:bg-cyan-500/10 hover:text-cyan-300 rounded transition-colors cursor-pointer"
            title="Zoom Out"
          >
            <ZoomOut size={13} />
          </button>
          <button
            onClick={() => {
              setZoom(1);
              if (isFocused) onToggleFocus();
              onSelectTarget(null);
            }}
            className="p-1 hover:bg-cyan-500/10 hover:text-cyan-300 rounded text-slate-500 hover:text-slate-300 border-l border-slate-800 ml-0.5 transition-colors cursor-pointer"
            title="Reset View"
          >
            <RotateCcw size={11} />
          </button>
        </div>
      </div>

      {/* Main Viewport Container */}
      <div className="absolute inset-0 flex items-center justify-center overflow-hidden p-6">
        <div
          className="relative w-full h-full max-w-full max-h-full flex items-center justify-center transition-all duration-300 ease-out"
          style={{
            transform: `scale(${effectiveZoom})`,
            transformOrigin: transformOrigin,
          }}
        >
          {/* Actual Source Image */}
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={sourceImage.dataUrl}
            alt={sourceImage.filename}
            className="w-full h-full object-contain select-none pointer-events-none"
          />

          {/* Dynamic Grounding Overlays */}
          {normalizedDetections.map((d) => {
            const [xmin, ymin, xmax, ymax] = d.box;
            const top = (ymin / 1000) * 100;
            const left = (xmin / 1000) * 100;
            const width = Math.max(2, ((xmax - xmin) / 1000) * 100);
            const height = Math.max(2, ((ymax - ymin) / 1000) * 100);
            const isSelected = selectedTargetId === d.id;

            return (
              <div
                key={d.id}
                onClick={(e) => {
                  e.stopPropagation();
                  onSelectTarget(isSelected ? null : d.id);
                }}
                className={`absolute border-2 rounded transition-all cursor-pointer flex flex-col justify-between p-1 ${
                  isSelected
                    ? "border-cyan-300 ring-4 ring-cyan-500/50 bg-cyan-500/30 z-30 shadow-[0_0_24px_rgba(6,182,212,0.9)] scale-[1.03]"
                    : selectedTargetId
                    ? "border-cyan-500/40 bg-cyan-500/10 opacity-50 hover:opacity-100 hover:border-cyan-400 z-10"
                    : "border-cyan-400/80 bg-cyan-500/15 hover:border-cyan-300 hover:bg-cyan-500/25 shadow-[0_0_12px_rgba(6,182,212,0.4)] z-15"
                }`}
                style={{
                  top: `${top}%`,
                  left: `${left}%`,
                  width: `${width}%`,
                  height: `${height}%`,
                }}
              >
                <div className="flex justify-between items-center gap-1">
                  <span className={`text-[8px] font-mono font-bold px-1 py-0.5 rounded border truncate ${
                    isSelected
                      ? "bg-cyan-500 text-slate-950 border-cyan-300"
                      : "bg-slate-900/90 text-cyan-300 border-cyan-500/30"
                  }`}>
                    {d.id}: {d.label}
                  </span>
                  {d.score !== null && (
                    <span className="text-[8px] font-mono text-emerald-300 bg-emerald-500/25 px-1 py-0.5 rounded font-semibold whitespace-nowrap">
                      {Math.round(d.score * 100)}%
                    </span>
                  )}
                </div>
                <div className={`self-center ${isSelected ? "opacity-100 animate-pulse" : "opacity-50"}`}>
                  <Crosshair size={isSelected ? 16 : 13} className={isSelected ? "text-cyan-300" : "text-cyan-400"} />
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Empty State Banner if zero detections */}
      {normalizedDetections.length === 0 && (
        <div className="relative z-20 self-center bg-slate-900/80 border border-slate-800 rounded-lg px-3 py-1.5 text-xs text-slate-400 font-mono">
          No spatial bounding proposals for current query
        </div>
      )}

      {/* Bottom Coordinates HUD */}
      <div className="relative z-20 flex items-center justify-between text-[10px] font-mono text-slate-400 bg-[#091522]/90 backdrop-blur-md px-3 py-1.5 rounded-lg border border-slate-800 self-start shadow-sm">
        <span>{coordText}</span>
      </div>
    </div>
  );
}

// ---------------------------------------------
// Right Sidebar: Detected Targets & Inspector
// ---------------------------------------------
function TargetsSidebar({
  investigation,
  sourceImage,
  normalizedDetections,
  selectedTargetId,
  onSelectTarget,
  isFocused,
  onToggleFocus,
}: {
  investigation: CanonicalInvestigationState | null;
  sourceImage: CanonicalSourceImage;
  normalizedDetections: Array<{ id: string; evidenceId: string; label: string; score: number | null; box: [number, number, number, number] }>;
  selectedTargetId: string | null;
  onSelectTarget: (id: string | null) => void;
  isFocused: boolean;
  onToggleFocus: () => void;
}) {
  const selectedTarget = useMemo(() => {
    return normalizedDetections.find((d) => d.id === selectedTargetId) || null;
  }, [normalizedDetections, selectedTargetId]);

  const caption = typeof investigation?.response?.optical_caption === "string"
    ? investigation.response.optical_caption
    : (investigation?.response?.optical_caption as unknown as { caption?: string })?.caption || null;

  const primaryAnswer = investigation?.response?.response_text ||
    (investigation?.response?.investigation_report?.observations && investigation.response.investigation_report.observations[0]) ||
    caption ||
    "Scan complete.";

  return (
    <div className="w-full lg:w-96 border-t lg:border-t-0 lg:border-l border-slate-800/80 bg-[#0c1624]/60 backdrop-blur-md p-4 sm:p-5 flex flex-col justify-between overflow-y-auto space-y-4">
      <div className="space-y-4">
        {/* User Query Banner */}
        <div className="bg-slate-900/90 border border-cyan-500/30 rounded-xl p-3 space-y-1.5 shadow-sm">
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-mono text-cyan-400 font-bold uppercase tracking-wider flex items-center gap-1.5">
              <Search size={12} />
              <span>User Query</span>
            </span>
            <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-cyan-500/10 text-cyan-300 border border-cyan-500/20">
              {investigation?.investigation_id || "ACTIVE"}
            </span>
          </div>
          <p className="text-xs font-semibold text-white font-mono leading-snug">
            &ldquo;{investigation?.query || "No query recorded"}&rdquo;
          </p>
        </div>

        {/* Primary Answer Box */}
        <div className="bg-cyan-950/40 border border-cyan-500/40 rounded-xl p-3 space-y-1 shadow-sm">
          <span className="text-[10px] font-mono text-cyan-300 uppercase tracking-wider font-bold">
            Primary Findings / Response
          </span>
          <p className="text-xs text-slate-100 leading-relaxed font-sans font-medium">
            {primaryAnswer}
          </p>
        </div>

        {/* Header & Target Count */}
        <div className="flex items-center justify-between pb-2 border-b border-slate-800/80">
          <div>
            <h2 className="text-sm font-semibold text-white flex items-center gap-1.5 tracking-tight">
              <Layers size={14} className="text-cyan-400" />
              <span>Target Entities ({normalizedDetections.length})</span>
            </h2>
            <p className="text-[10px] text-slate-400 font-mono mt-0.5 truncate max-w-[200px]">
              Source: <strong className="text-cyan-300">{sourceImage.filename}</strong>
            </p>
          </div>
          <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-cyan-500/10 text-cyan-300 border border-cyan-500/20 font-bold">
            {normalizedDetections.length} Flagged
          </span>
        </div>

        {/* Selected Target Inspector Card */}
        {selectedTarget ? (
          <div className="bg-cyan-950/30 border border-cyan-500/40 rounded-xl p-3.5 space-y-2.5 shadow-[0_0_15px_rgba(6,182,212,0.15)]">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-cyan-300 flex items-center gap-1.5">
                <Crosshair size={14} className="text-cyan-400 animate-pulse" />
                <span>Selected: {selectedTarget.id}</span>
              </span>
              <span className="text-[10px] font-mono text-emerald-300 bg-emerald-500/20 px-2 py-0.5 rounded border border-emerald-500/30 font-bold">
                {selectedTarget.score !== null ? `${Math.round(selectedTarget.score * 100)}% Confidence` : "Candidate"}
              </span>
            </div>

            <div className="text-xs text-slate-200">
              <strong>Entity Type:</strong> <span className="text-white capitalize">{selectedTarget.label}</span>
            </div>

            <div className="text-[11px] font-mono text-slate-400 bg-slate-950/70 p-2 rounded border border-slate-800/80 space-y-0.5">
              <div>Box Coordinates: [{selectedTarget.box.join(", ")}]</div>
              <div>Linked Evidence Node: <strong className="text-cyan-300">{selectedTarget.evidenceId}</strong></div>
            </div>

            <div className="flex gap-2 pt-1">
              <button
                onClick={onToggleFocus}
                className={`flex-1 flex items-center justify-center gap-1 text-xs font-mono py-1.5 rounded transition-all cursor-pointer ${
                  isFocused
                    ? "bg-cyan-500 text-slate-950 font-bold"
                    : "bg-slate-900 hover:bg-slate-800 border border-slate-800 text-cyan-300"
                }`}
              >
                <Focus size={12} />
                <span>{isFocused ? "Reset Zoom" : "Focus on Map"}</span>
              </button>
              <button
                onClick={() => onSelectTarget(null)}
                className="px-2.5 py-1.5 rounded text-xs font-mono bg-slate-900/60 hover:bg-slate-800 border border-slate-800 text-slate-400 hover:text-slate-200 cursor-pointer"
              >
                Deselect
              </button>
            </div>
          </div>
        ) : null}

        {/* List of Selectable Target Cards */}
        <div className="space-y-2 max-h-72 overflow-y-auto pr-1">
          {normalizedDetections.map((d) => {
            const isSelected = selectedTargetId === d.id;
            const [xmin, ymin, xmax, ymax] = d.box;
            return (
              <div
                key={d.id}
                onClick={() => onSelectTarget(isSelected ? null : d.id)}
                className={`p-3 rounded-xl border transition-all cursor-pointer text-xs font-mono ${
                  isSelected
                    ? "bg-cyan-500/20 border-cyan-400 shadow-[0_0_12px_rgba(6,182,212,0.3)] ring-1 ring-cyan-500/50"
                    : "bg-slate-900/50 border-slate-800/80 hover:border-cyan-500/40 hover:bg-slate-900/80"
                }`}
              >
                <div className="flex items-center justify-between mb-1">
                  <span className="font-semibold text-white flex items-center gap-1.5">
                    <Building2 size={13} className={isSelected ? "text-cyan-300" : "text-cyan-400"} />
                    <span>{d.id}: {d.label}</span>
                  </span>
                  {d.score !== null && (
                    <span className="text-[10px] text-emerald-300 bg-emerald-500/20 px-1.5 py-0.5 rounded font-bold">
                      {Math.round(d.score * 100)}%
                    </span>
                  )}
                </div>
                <div className="text-[11px] text-slate-400 flex items-center justify-between">
                  <span>[{xmin}, {ymin}, {xmax}, {ymax}]</span>
                  <span className="text-cyan-400 text-[10px]">Inspect →</span>
                </div>
              </div>
            );
          })}

          {normalizedDetections.length === 0 && (
            <div className="p-4 rounded-xl border border-slate-800/80 bg-slate-900/30 text-center text-xs text-slate-400 font-mono space-y-1">
              <p>No target entities flagged.</p>
              <p className="text-[10px] text-slate-500">Spatial bounding proposals are query-specific.</p>
            </div>
          )}
        </div>
      </div>

      {/* Navigation Quick Links */}
      <div className="pt-3 border-t border-slate-800/80 flex flex-col gap-2">
        <Link
          href="/evidence"
          className="flex items-center justify-between p-2.5 rounded-lg bg-slate-900/70 hover:bg-cyan-500/10 border border-slate-800 hover:border-cyan-500/40 text-xs font-mono text-cyan-300 transition-all cursor-pointer"
        >
          <span className="flex items-center gap-1.5">
            <Layers size={13} />
            <span>Open in Evidence Viewer</span>
          </span>
          <ArrowRight size={13} />
        </Link>
        <Link
          href="/reports"
          className="flex items-center justify-between p-2.5 rounded-lg bg-slate-900/70 hover:bg-cyan-500/10 border border-slate-800 hover:border-cyan-500/40 text-xs font-mono text-cyan-300 transition-all cursor-pointer"
        >
          <span className="flex items-center gap-1.5">
            <FileText size={13} />
            <span>Generate Intelligence Report</span>
          </span>
          <ArrowRight size={13} />
        </Link>
        <Link
          href="/query"
          className="text-center py-2 rounded-lg bg-slate-900/40 hover:bg-slate-800/80 text-[11px] font-mono text-slate-400 hover:text-white border border-slate-800/60 transition-all"
        >
          <span>New Query / Upload</span>
        </Link>
      </div>
    </div>
  );
}

// ---------------------------------------------
// Scan Results Page Main Component
// ---------------------------------------------
export default function ScanResultsPage() {
  const [investigationState, setInvestigationState] = useState<CanonicalInvestigationState | null>(() => {
    return getCurrentInvestigation();
  });

  useEffect(() => {
    setInvestigationState(getCurrentInvestigation());

    const syncHandler = () => {
      setInvestigationState(getCurrentInvestigation());
    };

    window.addEventListener("nexspace-investigation-changed", syncHandler);
    window.addEventListener("nexspace-source-changed", syncHandler);
    window.addEventListener("storage", syncHandler);

    return () => {
      window.removeEventListener("nexspace-investigation-changed", syncHandler);
      window.removeEventListener("nexspace-source-changed", syncHandler);
      window.removeEventListener("storage", syncHandler);
    };
  }, []);

  const [selectedTargetId, setSelectedTargetId] = useState<string | null>(
    () => investigationState?.selectedTargetId || null
  );
  const [isFocused, setIsFocused] = useState(false);

  // Synchronized normalized detections extraction across all response channels
  const normalizedDetections = useMemo(() => {
    if (!investigationState?.response) return [];

    const list: Array<{ id: string; evidenceId: string; label: string; score: number | null; box: [number, number, number, number] }> = [];
    const seen = new Set<string>();

    // 1. Check grounding.detections
    const rawGrounding = investigationState.response.grounding?.detections || [];
    rawGrounding.forEach((det) => {
      const d = det as unknown as Record<string, unknown>;
      const nBox = normalizeBox(d);
      if (nBox) {
        const key = `${nBox.join(",")}-${d.label || ""}`;
        if (!seen.has(key)) {
          seen.add(key);
          list.push({
            id: `TARGET-${String(list.length + 1).padStart(2, "0")}`,
            evidenceId: `EVD-${String(list.length + 1).padStart(3, "0")}`,
            label: (d.label as string) || `Target #${list.length + 1}`,
            score: typeof d.score === "number" ? d.score : null,
            box: nBox,
          });
        }
      }
    });

    // 2. Check evidence bounding boxes
    const rawEvidence = investigationState.response.evidence || [];
    rawEvidence.forEach((ev) => {
      const e = ev as unknown as Record<string, unknown>;
      const p = (e.payload as Record<string, unknown>) || e;
      if (e.type === "bounding_box" || p.box || p.box_2d || p.bbox_normalized || p.bbox_pixel) {
        const nBox = normalizeBox(p);
        if (nBox) {
          const key = `${nBox.join(",")}-${p.label || e.label || ""}`;
          if (!seen.has(key)) {
            seen.add(key);
            list.push({
              id: `TARGET-${String(list.length + 1).padStart(2, "0")}`,
              evidenceId: `EVD-${String(list.length + 1).padStart(3, "0")}`,
              label: (p.label as string) || (e.label as string) || `Target #${list.length + 1}`,
              score: typeof p.score === "number" ? p.score : (typeof e.confidence === "number" ? e.confidence : null),
              box: nBox,
            });
          }
        }
      }
    });

    return list;
  }, [investigationState]);

  // Sync target selection back to canonical investigation in storage
  const handleSelectTarget = (id: string | null) => {
    setSelectedTargetId(id);
    if (!id) setIsFocused(false);
    updateSelectedTarget(id);
    const updated = getCurrentInvestigation();
    if (updated) setInvestigationState(updated);
  };

  const handleToggleFocus = () => {
    setIsFocused((f) => !f);
  };

  const sourceImage = useMemo<CanonicalSourceImage>(() => {
    if (investigationState?.source_image?.dataUrl) {
      return investigationState.source_image;
    }
    const active = getActiveSourceImage(false);
    if (active) return active;
    return DEFAULT_DEMO_SOURCE;
  }, [investigationState]);

  return (
    <div className="flex flex-col lg:flex-row min-h-screen bg-[#0a1420] text-white font-sans overflow-x-hidden">
      <Sidebar />

      <div className="flex-1 flex flex-col min-w-0">
        <TopBar investigation={investigationState} sourceImage={sourceImage} />

        <main className="flex-1 flex flex-col lg:flex-row overflow-hidden">
          <MapArea
            investigation={investigationState}
            sourceImage={sourceImage}
            normalizedDetections={normalizedDetections}
            selectedTargetId={selectedTargetId}
            onSelectTarget={handleSelectTarget}
            isFocused={isFocused}
            onToggleFocus={handleToggleFocus}
          />
          <TargetsSidebar
            investigation={investigationState}
            sourceImage={sourceImage}
            normalizedDetections={normalizedDetections}
            selectedTargetId={selectedTargetId}
            onSelectTarget={handleSelectTarget}
            isFocused={isFocused}
            onToggleFocus={handleToggleFocus}
          />
        </main>
      </div>
    </div>
  );
}