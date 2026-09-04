"use client";

import { useState, useMemo } from "react";
import Link from "next/link";
import {
  Download,
  Layers,
  ZoomIn,
  ZoomOut,
  CheckCircle2,
  Flag,
  XCircle,
  Crosshair,
  Info,
  ShieldCheck,
  Building2,
  FileText,
  Sparkles,
  ChevronDown,
  ChevronUp,
  HelpCircle,
  Focus,
  RotateCcw,
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
    <header className="flex flex-col gap-3 px-4 sm:px-6 lg:px-8 py-4 border-b border-slate-800/80 bg-[#0d1826]/40 backdrop-blur-md">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div className="flex flex-wrap items-center gap-2 sm:gap-3">
          <div className="flex items-center gap-2">
            <ShieldCheck className="text-cyan-400" size={16} />
            <h1 className="text-sm font-semibold text-white tracking-tight">Evidence Viewer</h1>
          </div>
          <span className="text-[10px] font-mono text-cyan-400 bg-cyan-500/10 px-2 py-0.5 rounded-full border border-cyan-500/20">
            Investigation ID: {investigation?.investigation_id || "LIVE-INV-ACTIVE"}
          </span>
          <span className={`text-[10px] font-mono px-2 py-0.5 rounded-full border ${
            isUpload
              ? "text-emerald-300 bg-emerald-500/15 border-emerald-500/30"
              : "text-blue-300 bg-blue-500/15 border-blue-500/30"
          }`}>
            Source: <strong>{sourceImage.filename}</strong> ({isUpload ? "Uploaded by user" : "Verified demo tile"})
          </span>
        </div>
        <div className="flex items-center gap-2 self-start sm:self-auto">
          <button
            onClick={() => window.print()}
            className="flex items-center gap-1.5 bg-slate-900/60 hover:bg-slate-800/80 border border-slate-800 hover:border-cyan-500/30 transition-all text-xs text-slate-300 hover:text-cyan-300 rounded-lg px-3 py-1.5 font-mono cursor-pointer"
          >
            <Download size={12} />
            <span>Print / Export PDF</span>
          </button>
          <Link
            href="/results"
            className="flex items-center gap-1.5 bg-cyan-500/20 hover:bg-cyan-500/30 border border-cyan-500/40 text-xs text-cyan-300 rounded-lg px-3 py-1.5 font-mono"
          >
            <span>Scan Results</span>
          </Link>
        </div>
      </div>

      <div className="flex items-center gap-2 bg-cyan-500/10 border border-cyan-500/25 rounded-lg px-3 py-2 text-[11px] text-cyan-300 font-mono">
        <Info size={14} className="text-cyan-400 shrink-0" />
        <span>
          <strong>Investigation Evidence Verification:</strong> Visual inspection of the exact source image and neural object localizations. Click any bounding box or target card to inspect its evidence node.
        </span>
      </div>
    </header>
  );
}

// ---------------------------------------------
// Image viewer with interactive target focus & zoom
// ---------------------------------------------
function EvidenceImage({
  investigation,
  sourceImage,
  selectedTargetId,
  onSelectTarget,
  isFocused,
  onToggleFocus,
}: {
  investigation: CanonicalInvestigationState | null;
  sourceImage: CanonicalSourceImage;
  selectedTargetId: string | null;
  onSelectTarget: (id: string | null) => void;
  isFocused: boolean;
  onToggleFocus: () => void;
}) {
  const [zoom, setZoom] = useState(1);

  const normalizedDetections = useMemo(() => {
    const raw = investigation?.response?.grounding?.detections || [];
    return raw
      .map((det, idx) => {
        const d = det as unknown as Record<string, unknown>;
        const nBox = normalizeBox(d);
        if (!nBox) return null;
        return {
          id: `TARGET-${String(idx + 1).padStart(2, "0")}`,
          evidenceId: `EVD-${String(idx + 1).padStart(3, "0")}`,
          label: (d.label as string) || `Target #${idx + 1}`,
          score: typeof d.score === "number" ? d.score : null,
          box: nBox,
        };
      })
      .filter((d): d is NonNullable<typeof d> => d !== null);
  }, [investigation]);

  const selectedTarget = useMemo(() => {
    return normalizedDetections.find((d) => d.id === selectedTargetId) || null;
  }, [normalizedDetections, selectedTargetId]);

  const transformOrigin = useMemo(() => {
    if (!selectedTarget || !isFocused) return "center center";
    const [xmin, ymin, xmax, ymax] = selectedTarget.box;
    const cx = ((xmin + xmax) / 2 / 10).toFixed(1);
    const cy = ((ymin + ymax) / 2 / 10).toFixed(1);
    return `${cx}% ${cy}%`;
  }, [selectedTarget, isFocused]);

  const effectiveZoom = isFocused ? Math.max(zoom, 2.0) : zoom;

  const geoMeta = investigation?.response?.geospatial_metadata;
  const coordString = geoMeta?.geospatial_available && geoMeta.crs
    ? `CRS: ${geoMeta.crs} · RESOLUTION: ${geoMeta.resolution?.x || 10}m/px`
    : `SOURCE: ${sourceImage.filename} · SPATIAL FRAME: 512x512 PIXEL COORDINATES`;

  return (
    <div className="w-full flex-1 flex flex-col bg-[#09121d] relative overflow-hidden rounded-xl border border-slate-800/90 min-h-[480px] shadow-[0_4px_24px_rgba(0,0,0,0.3)]">
      {/* HUD Header Bar */}
      <div className="flex items-center justify-between px-3 sm:px-4 py-2 bg-[#09111c] border-b border-slate-800/80 text-[11px] font-mono text-slate-400 z-20">
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
          <span className="text-slate-200 font-semibold uppercase tracking-wider text-[10px]">
            ACTIVE SOURCE IMAGE &amp; SPATIAL TARGETS ({normalizedDetections.length})
          </span>
          {selectedTarget && (
            <span className="text-emerald-300 font-bold ml-2">
              [FOCUS: {selectedTarget.id} ({selectedTarget.evidenceId})]
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          <span className="bg-slate-900/80 border border-slate-800 rounded-md px-2 py-1 text-[10px] text-cyan-300 font-mono flex items-center gap-1.5">
            <Layers size={11} />
            <span>{sourceImage.filename}</span>
          </span>
        </div>
      </div>

      {/* Main Satellite Viewport with Real User Image */}
      <div className="flex-1 relative bg-[#09121d] overflow-hidden p-4 flex flex-col justify-between">
        {/* Actual Investigation Source Image Container */}
        <div className="absolute inset-0 flex items-center justify-center overflow-hidden">
          <div
            className="relative w-full h-full max-w-full max-h-full flex items-center justify-center transition-all duration-300 ease-out"
            style={{
              transform: `scale(${effectiveZoom})`,
              transformOrigin: transformOrigin,
            }}
          >
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={sourceImage.dataUrl}
              alt={sourceImage.filename}
              className="w-full h-full object-contain select-none pointer-events-none"
            />

            {/* Dynamic Grounding Overlays with Target Selection */}
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
                  className={`absolute border-2 rounded flex flex-col justify-between p-1.5 transition-all cursor-pointer ${
                    isSelected
                      ? "border-cyan-300 ring-4 ring-cyan-500/50 bg-cyan-500/30 z-30 shadow-[0_0_24px_rgba(6,182,212,0.9)] scale-[1.03]"
                      : selectedTargetId
                      ? "border-cyan-500/40 bg-cyan-500/10 opacity-50 hover:opacity-100 hover:border-cyan-400 z-10"
                      : "border-cyan-400 bg-cyan-500/15 hover:border-cyan-300 hover:bg-cyan-500/25 shadow-[0_0_15px_rgba(6,182,212,0.5)] z-15"
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
                      <span className="text-[8px] font-mono text-emerald-300 bg-emerald-500/20 px-1 py-0.5 rounded font-semibold whitespace-nowrap">
                        {Math.round(d.score * 100)}%
                      </span>
                    )}
                  </div>
                  <div className={`self-center ${isSelected ? "opacity-100 animate-pulse" : "opacity-50"}`}>
                    <Crosshair size={isSelected ? 18 : 14} className={isSelected ? "text-cyan-300" : "text-cyan-400"} />
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Empty state notice if zero detections */}
        {normalizedDetections.length === 0 && (
          <div className="relative z-20 self-center bg-slate-900/80 border border-slate-800 rounded-lg px-3 py-1.5 text-xs text-slate-400 font-mono">
            No localized spatial targets for current query
          </div>
        )}

        {/* Interactive Zoom Controls */}
        <div className="absolute bottom-3 right-3 flex items-center gap-1 bg-slate-900/90 p-1 rounded-lg border border-slate-800 z-20 font-mono text-[10px]">
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
              <span>{isFocused ? "Unfocus" : "Focus"}</span>
            </button>
          )}

          <button
            onClick={() => setZoom((z) => Math.min(Number((z + 0.25).toFixed(2)), 3.0))}
            className="p-1.5 rounded hover:bg-cyan-500/10 text-slate-300 hover:text-cyan-300 transition-colors cursor-pointer"
            title="Zoom In"
          >
            <ZoomIn size={14} />
          </button>
          <span className="text-cyan-400 font-semibold px-1">{effectiveZoom.toFixed(2)}×</span>
          <button
            onClick={() => setZoom((z) => Math.max(Number((z - 0.25).toFixed(2)), 0.8))}
            className="p-1.5 rounded hover:bg-cyan-500/10 text-slate-300 hover:text-cyan-300 transition-colors cursor-pointer"
            title="Zoom Out"
          >
            <ZoomOut size={14} />
          </button>
          <button
            onClick={() => {
              setZoom(1);
              if (isFocused) onToggleFocus();
              onSelectTarget(null);
            }}
            className="p-1.5 rounded hover:bg-cyan-500/10 text-slate-400 hover:text-cyan-300 transition-colors cursor-pointer"
            title="Reset View"
          >
            <RotateCcw size={12} />
          </button>
        </div>

        {/* Coordinates Banner */}
        <div className="relative z-20 text-[10px] font-mono text-slate-300 bg-slate-900/90 px-3 py-1 rounded-md border border-slate-800 self-start shadow">
          {coordString}
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------
// Right panel: Findings & Target Evidence Verification
// ---------------------------------------------
function EvidencePanel({
  investigation,
  sourceImage,
  selectedTargetId,
  onSelectTarget,
  isFocused,
  onToggleFocus,
}: {
  investigation: CanonicalInvestigationState | null;
  sourceImage: CanonicalSourceImage;
  selectedTargetId: string | null;
  onSelectTarget: (id: string | null) => void;
  isFocused: boolean;
  onToggleFocus: () => void;
}) {
  const [status, setStatus] = useState<string>("CONFIRMED");
  const [showAdvanced, setShowAdvanced] = useState(false);

  const normalizedDetections = useMemo(() => {
    const raw = investigation?.response?.grounding?.detections || [];
    return raw
      .map((det, idx) => {
        const d = det as unknown as Record<string, unknown>;
        const nBox = normalizeBox(d);
        if (!nBox) return null;
        return {
          id: `TARGET-${String(idx + 1).padStart(2, "0")}`,
          evidenceId: `EVD-${String(idx + 1).padStart(3, "0")}`,
          label: (d.label as string) || `Target #${idx + 1}`,
          score: typeof d.score === "number" ? d.score : null,
          box: nBox,
        };
      })
      .filter((d): d is NonNullable<typeof d> => d !== null);
  }, [investigation]);

  const selectedTarget = useMemo(() => {
    return normalizedDetections.find((d) => d.id === selectedTargetId) || null;
  }, [normalizedDetections, selectedTargetId]);

  const caption = typeof investigation?.response?.optical_caption === "string"
    ? investigation.response.optical_caption
    : (investigation?.response?.optical_caption as unknown as { caption?: string })?.caption || "Scene analysis completed.";
  const vqa = investigation?.response?.vqa_results || [];

  return (
    <div className="w-full border border-slate-800/90 bg-[#0c1624]/60 backdrop-blur-md rounded-xl p-4 sm:p-5 flex flex-col justify-between shadow-[0_4px_24px_rgba(0,0,0,0.3)] min-h-[480px] space-y-4">
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <p className="text-sm font-semibold text-white tracking-tight flex items-center gap-1.5">
            <Sparkles size={14} className="text-cyan-400" />
            <span>What We Found</span>
          </p>
          <span className="text-[10px] font-mono px-2.5 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-bold">
            {status}
          </span>
        </div>

        {/* Selected Target Specific Evidence Inspector */}
        {selectedTarget ? (
          <div className="bg-cyan-950/30 border border-cyan-500/40 rounded-xl p-3.5 space-y-2.5 shadow-[0_0_15px_rgba(6,182,212,0.15)]">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-cyan-300 flex items-center gap-1.5">
                <Crosshair size={14} className="text-cyan-400 animate-pulse" />
                <span>Selected Target: {selectedTarget.id}</span>
              </span>
              <span className="text-[10px] font-mono text-emerald-300 bg-emerald-500/20 px-2 py-0.5 rounded border border-emerald-500/30 font-bold">
                {selectedTarget.score !== null ? `${Math.round(selectedTarget.score * 100)}% Confidence` : "Candidate"}
              </span>
            </div>

            <div className="text-xs text-slate-200">
              <strong>Structure Identification:</strong> <span className="text-white capitalize">{selectedTarget.label}</span>
            </div>

            <div className="text-[11px] font-mono text-slate-400 bg-slate-950/70 p-2 rounded border border-slate-800/80 space-y-0.5">
              <div>Evidence Node: <strong className="text-cyan-300">{selectedTarget.evidenceId}</strong></div>
              <div>Bounding Box (0-1000): [{selectedTarget.box.join(", ")}]</div>
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
        ) : (
          /* General findings overview */
          <div className="bg-slate-900/60 border border-slate-800/80 rounded-xl p-3.5 space-y-2.5 text-xs">
            <div className="flex items-start gap-2 text-slate-200">
              <Building2 size={15} className="text-cyan-400 shrink-0 mt-0.5" />
              <div>
                <strong>Target Structures:</strong>{" "}
                {normalizedDetections.length > 0
                  ? `${normalizedDetections.length} candidate structure(s) localized in imagery`
                  : "No target structures detected"}
              </div>
            </div>

            <div className="flex items-start gap-2 text-slate-200">
              <FileText size={15} className="text-cyan-400 shrink-0 mt-0.5" />
              <div>
                <strong>Image Summary:</strong> {caption}
              </div>
            </div>

            {vqa.length > 0 && (
              <div className="flex items-start gap-2 text-slate-200">
                <HelpCircle size={15} className="text-cyan-400 shrink-0 mt-0.5" />
                <div>
                  <strong>Visual Q&amp;A:</strong> {vqa.map(v => `${v.question} → ${v.answer}`).join("; ")}
                </div>
              </div>
            )}
          </div>
        )}

        {/* List of Selectable Targets */}
        {normalizedDetections.length > 0 && (
          <div className="space-y-1.5">
            <div className="text-[10px] font-mono text-slate-400 uppercase tracking-wide">
              Selectable Evidence Nodes ({normalizedDetections.length}):
            </div>
            <div className="max-h-36 overflow-y-auto space-y-1 pr-1">
              {normalizedDetections.map((d) => {
                const isSelected = selectedTargetId === d.id;
                return (
                  <div
                    key={d.id}
                    onClick={() => onSelectTarget(isSelected ? null : d.id)}
                    className={`px-2.5 py-1.5 rounded-lg border text-[11px] font-mono cursor-pointer transition-all flex items-center justify-between ${
                      isSelected
                        ? "bg-cyan-500/25 border-cyan-400 text-cyan-200 ring-1 ring-cyan-500/50"
                        : "bg-slate-900/60 border-slate-800 text-slate-300 hover:border-cyan-500/30"
                    }`}
                  >
                    <span>{d.id}: {d.label}</span>
                    <span className="text-[10px] text-emerald-400">{d.evidenceId}</span>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Advanced details accordion */}
        <div className="border-t border-slate-800/60 pt-2">
          <button
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="w-full flex items-center justify-between text-xs font-mono text-slate-400 hover:text-cyan-300 transition-colors py-1 cursor-pointer"
          >
            <span>Advanced Execution Details</span>
            {showAdvanced ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
          </button>

          {showAdvanced && (
            <div className="mt-2 space-y-2 bg-slate-950/80 border border-slate-800/80 rounded-lg p-3 text-[11px] font-mono text-slate-300">
              <div>Query: {investigation?.query || "N/A"}</div>
              <div>Investigation ID: {investigation?.investigation_id || "N/A"}</div>
              <div>Source ID: {sourceImage.id}</div>
              <div>SHA-256 Hash: {sourceImage.sha256 ? `${sourceImage.sha256.slice(0, 16)}...` : "Calculated"}</div>
              <div>Media Type: {sourceImage.mediaType}</div>
            </div>
          )}
        </div>
      </div>

      {/* Analyst Sign-off Buttons */}
      <div className="pt-2 border-t border-slate-800/60">
        <p className="text-[10px] font-semibold text-slate-400 mb-2 uppercase tracking-widest font-mono">
          Analyst Verification Action
        </p>

        <button
          onClick={() => setStatus("CONFIRMED")}
          className="w-full flex items-center justify-center gap-2 bg-emerald-500 hover:bg-emerald-400 active:scale-95 transition-all text-[#0a1420] text-xs font-bold rounded-lg py-2.5 mb-2 cursor-pointer shadow-[0_0_12px_rgba(52,211,153,0.3)]"
        >
          <CheckCircle2 size={14} />
          <span>Confirm Verification</span>
        </button>

        <div className="flex gap-2">
          <button
            onClick={() => setStatus("FLAGGED")}
            className="flex-1 flex items-center justify-center gap-1.5 bg-slate-900/60 hover:bg-slate-800/80 border border-slate-800 hover:border-amber-500/40 transition-all text-xs text-slate-300 hover:text-amber-300 rounded-lg py-2 cursor-pointer font-mono"
          >
            <Flag size={12} className="text-amber-400" />
            <span>Flag</span>
          </button>
          <button
            onClick={() => setStatus("REJECTED")}
            className="flex-1 flex items-center justify-center gap-1.5 bg-slate-900/60 hover:bg-slate-800/80 border border-slate-800 hover:border-red-500/40 transition-all text-xs text-slate-300 hover:text-red-300 rounded-lg py-2 cursor-pointer font-mono"
          >
            <XCircle size={12} className="text-red-400" />
            <span>Reject</span>
          </button>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------
// Evidence Viewer Page Main Component
// ---------------------------------------------
export default function EvidenceViewerPage() {
  const [investigationState, setInvestigationState] = useState<CanonicalInvestigationState | null>(() => {
    return getCurrentInvestigation();
  });

  const [selectedTargetId, setSelectedTargetId] = useState<string | null>(
    () => investigationState?.selectedTargetId || null
  );
  const [isFocused, setIsFocused] = useState(false);

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

        <main className="flex-1 p-4 sm:p-6 lg:p-8 grid grid-cols-1 lg:grid-cols-3 gap-5 items-start overflow-y-auto">
          <div className="lg:col-span-2">
            <EvidenceImage
              investigation={investigationState}
              sourceImage={sourceImage}
              selectedTargetId={selectedTargetId}
              onSelectTarget={handleSelectTarget}
              isFocused={isFocused}
              onToggleFocus={handleToggleFocus}
            />
          </div>
          <div className="lg:col-span-1">
            <EvidencePanel
              investigation={investigationState}
              sourceImage={sourceImage}
              selectedTargetId={selectedTargetId}
              onSelectTarget={handleSelectTarget}
              isFocused={isFocused}
              onToggleFocus={handleToggleFocus}
            />
          </div>
        </main>
      </div>
    </div>
  );
}