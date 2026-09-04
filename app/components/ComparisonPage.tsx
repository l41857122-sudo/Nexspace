"use client";

import { useState, useEffect, useRef } from "react";
import {
  Scan,
  Activity,
  ZoomIn,
  ZoomOut,
  ArrowLeftRight,
  Upload,
  RotateCcw,
  Sparkles,
  Crosshair,
  Loader2,
} from "lucide-react";
import Sidebar from "./Sidebar";
import type { NexSpaceChangeAnalysisResponse, AnomalyRegion } from "../types/nexspace";
import { SAMPLE_CHANGE_A, SAMPLE_CHANGE_B } from "../utils/sampleImages";
import { validateAndProcessImageFile, ACCEPT_FILE_ATTR } from "../utils/imageValidation";

// ---------------------------------------------
// Top bar
// ---------------------------------------------
function TopBar({
  deltaPct,
  backendOnline,
}: {
  deltaPct?: number;
  backendOnline: boolean | null;
}) {
  return (
    <header className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 px-4 sm:px-6 lg:px-8 py-4 border-b border-slate-800/80 bg-[#0d1826]/40 backdrop-blur-md">
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-2">
          <div className="p-1.5 rounded-lg bg-cyan-500/10 border border-cyan-500/20 text-cyan-400">
            <Scan size={15} />
          </div>
          <div>
            <h1 className="text-sm font-semibold text-white tracking-tight">
              Bi-Temporal Change Analysis Terminal
            </h1>
            <p className="text-[10px] text-slate-400 font-mono">
              Co-registered differential pixel analysis · NexSpace ML Engine
            </p>
          </div>
        </div>
        {deltaPct !== undefined && (
          <span className="text-[10px] font-mono text-cyan-400 bg-cyan-500/10 px-2 py-0.5 rounded-full border border-cyan-500/20 hidden md:inline-block">
            DELTA Δ {(deltaPct * 100).toFixed(1)}%
          </span>
        )}
      </div>

      <div className="flex items-center gap-2 self-end sm:self-auto">
        <div className={`flex items-center gap-1.5 text-[11px] font-mono px-2.5 py-1 rounded-lg border ${
          backendOnline === false
            ? "text-rose-400 bg-rose-500/10 border-rose-500/20"
            : backendOnline === true
            ? "text-emerald-400 bg-emerald-500/10 border-emerald-500/20"
            : "text-slate-400 bg-slate-900 border-slate-800"
        }`}>
          <span className={`w-1.5 h-1.5 rounded-full ${
            backendOnline === false ? "bg-rose-500" : backendOnline === true ? "bg-emerald-400 animate-pulse" : "bg-amber-400"
          }`} />
          <span>Status: {backendOnline === false ? "AI Backend Offline" : backendOnline === true ? "Live ML Pipeline" : "Checking..."}</span>
        </div>
      </div>
    </header>
  );
}

// ---------------------------------------------
// Image Viewer with Split / Overlay Slider
// ---------------------------------------------
function ImageViewer({
  opacity,
  deltaHeatmap,
  heatmapImage,
  activeAnomaly,
  hoveredAnomaly,
  setActiveAnomaly,
  zoom,
  setZoom,
  imageA,
  imageB,
  labelA,
  labelB,
  anomalies,
}: {
  opacity: number;
  deltaHeatmap: boolean;
  heatmapImage?: string | null;
  activeAnomaly: string | null;
  hoveredAnomaly: string | null;
  setActiveAnomaly: (v: string | null) => void;
  zoom: number;
  setZoom: (z: number | ((prev: number) => number)) => void;
  imageA: string;
  imageB: string;
  labelA: string;
  labelB: string;
  anomalies: AnomalyRegion[];
}) {
  return (
    <div className="w-full flex flex-col bg-[#0b1624] relative overflow-hidden rounded-xl border border-slate-800/90 shadow-[0_4px_30px_rgba(0,0,0,0.5)] min-h-[480px]">
      {/* HUD Header Bar */}
      <div className="flex items-center justify-between px-3 sm:px-4 py-2 bg-[#09111c] border-b border-slate-800/80 text-[11px] font-mono text-slate-400 z-20">
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
          <span className="text-slate-200 font-semibold uppercase tracking-wider text-[10px]">
            BI-TEMPORAL SYNCHRONIZED COMPARISON: {labelA} ↔ {labelB}
          </span>
        </div>

        {/* Zoom Controls */}
        <div className="flex items-center gap-1 bg-slate-900/80 border border-slate-800 rounded-lg p-0.5 shadow-inner">
          <button
            onClick={() => setZoom((z) => Math.min(Number((z + 0.25).toFixed(2)), 2.5))}
            aria-label="Zoom in"
            className="p-1 hover:bg-cyan-500/10 hover:text-cyan-300 rounded text-slate-400 transition-colors cursor-pointer"
            title="Zoom In"
          >
            <ZoomIn size={13} />
          </button>
          <span className="text-[10px] font-mono text-cyan-400 px-1.5 font-semibold min-w-[36px] text-center">
            {zoom.toFixed(2)}×
          </span>
          <button
            onClick={() => setZoom((z) => Math.max(Number((z - 0.25).toFixed(2)), 0.75))}
            aria-label="Zoom out"
            className="p-1 hover:bg-cyan-500/10 hover:text-cyan-300 rounded text-slate-400 transition-colors cursor-pointer"
            title="Zoom Out"
          >
            <ZoomOut size={13} />
          </button>
          <button
            onClick={() => setZoom(1)}
            aria-label="Reset zoom"
            className="p-1 hover:bg-cyan-500/10 hover:text-cyan-300 rounded text-slate-500 hover:text-slate-300 border-l border-slate-800 ml-0.5 cursor-pointer"
            title="Reset Zoom"
          >
            <RotateCcw size={11} />
          </button>
        </div>
      </div>

      {/* Viewport Canvas */}
      <div className="flex-1 relative bg-[#07111c] overflow-hidden flex items-center justify-center p-4">
        <div
          className="relative w-full h-full max-w-full max-h-full flex items-center justify-center transition-transform duration-200 ease-out"
          style={{ transform: `scale(${zoom})` }}
        >
          {/* Baseline Image A (Base) */}
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={imageA}
            alt="Baseline Image A"
            className="w-full h-full object-contain select-none pointer-events-none"
          />

          {/* Post-Event Image B (Blended Layer) */}
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={imageB}
            alt="Post-Event Image B"
            style={{ opacity: opacity / 100 }}
            className="absolute inset-0 w-full h-full object-contain select-none pointer-events-none transition-opacity duration-150"
          />

          {/* Differential Heatmap Layer */}
          {deltaHeatmap && heatmapImage && (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={heatmapImage}
              alt="Differential Heatmap"
              style={{ opacity: Math.min(1, (opacity / 100) * 0.9) }}
              className="absolute inset-0 w-full h-full object-contain select-none pointer-events-none mix-blend-screen"
            />
          )}

          {/* Dynamic Anomaly Bounding Overlays */}
          {anomalies.map((anom) => {
            const rawBox = anom.bbox_normalized || anom.bbox_pixel || [0, 0, 100, 100];
            const [xmin, ymin, xmax, ymax] = rawBox;
            const top = (ymin / 1000) * 100;
            const left = (xmin / 1000) * 100;
            const width = Math.max(2, ((xmax - xmin) / 1000) * 100);
            const height = Math.max(2, ((ymax - ymin) / 1000) * 100);
            const isSelected = activeAnomaly === anom.id;
            const isHovered = hoveredAnomaly === anom.id;

            return (
              <div
                key={anom.id}
                onClick={(e) => {
                  e.stopPropagation();
                  setActiveAnomaly(isSelected ? null : anom.id);
                }}
                className={`absolute border-2 rounded transition-all cursor-pointer flex flex-col justify-between p-1 z-20 ${
                  isSelected || isHovered
                    ? "border-amber-300 ring-4 ring-amber-500/50 bg-amber-500/30 scale-[1.03] shadow-[0_0_20px_rgba(245,158,11,0.8)]"
                    : "border-amber-400/80 bg-amber-500/15 hover:border-amber-300 hover:bg-amber-500/25 shadow-[0_0_10px_rgba(245,158,11,0.4)]"
                }`}
                style={{
                  top: `${top}%`,
                  left: `${left}%`,
                  width: `${width}%`,
                  height: `${height}%`,
                }}
              >
                <div className="flex justify-between items-center gap-1">
                  <span className="text-[8px] font-mono font-bold bg-slate-950/90 text-amber-300 border border-amber-500/30 px-1 py-0.5 rounded truncate">
                    {anom.label}
                  </span>
                  <span className="text-[8px] font-mono text-amber-200 bg-amber-950/80 px-1 py-0.5 rounded font-bold">
                    {(anom.severity || "medium").toUpperCase()}
                  </span>
                </div>
                <div className="self-center opacity-70">
                  <Crosshair size={13} className="text-amber-400" />
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Footer Badges */}
      <div className="px-4 py-2 bg-[#09111c] border-t border-slate-800/80 flex items-center justify-between text-[10px] font-mono text-slate-400">
        <div className="flex items-center gap-3">
          <span>Baseline A: <strong className="text-slate-200">{labelA}</strong></span>
          <span className="text-slate-600">|</span>
          <span>Post-Event B: <strong className="text-cyan-300">{labelB}</strong></span>
        </div>
        <div>Anomalies Detected: <strong className="text-amber-300">{anomalies.length}</strong></div>
      </div>
    </div>
  );
}

// ---------------------------------------------
// Right Control & Findings Panel
// ---------------------------------------------
function AnalysisPanel({
  opacity,
  setOpacity,
  deltaHeatmap,
  setDeltaHeatmap,
  activeAnomaly,
  hoveredAnomaly,
  setActiveAnomaly,
  setHoveredAnomaly,
  data,
  loading,
  onRunAnalysis,
  onSwap,
  onReset,
  onUploadA,
  onUploadB,
}: {
  opacity: number;
  setOpacity: (v: number) => void;
  deltaHeatmap: boolean;
  setDeltaHeatmap: (v: boolean) => void;
  activeAnomaly: string | null;
  hoveredAnomaly: string | null;
  setActiveAnomaly: (v: string | null) => void;
  setHoveredAnomaly: (v: string | null) => void;
  data: NexSpaceChangeAnalysisResponse | null;
  loading: boolean;
  onRunAnalysis: () => void;
  onSwap: () => void;
  onReset: () => void;
  onUploadA: (e: React.ChangeEvent<HTMLInputElement>) => void;
  onUploadB: (e: React.ChangeEvent<HTMLInputElement>) => void;
}) {
  const fileInputARef = useRef<HTMLInputElement>(null);
  const fileInputBRef = useRef<HTMLInputElement>(null);
  const anomalies = data?.anomalies || [];
  const deltaPct = data?.changed_fraction ? (data.changed_fraction * 100).toFixed(1) : "0.0";

  return (
    <div className="w-full border border-slate-800/90 bg-[#0c1624]/60 backdrop-blur-md rounded-xl p-5 shadow-[0_4px_24px_rgba(0,0,0,0.3)] space-y-4">
      <div className="flex items-center justify-between border-b border-slate-800 pb-3">
        <div>
          <h3 className="text-sm font-semibold text-white tracking-tight flex items-center gap-1.5">
            <Sparkles size={14} className="text-cyan-400" />
            <span>Differential Assessment</span>
          </h3>
          <p className="text-[11px] text-slate-400 font-mono mt-0.5">
            {anomalies.length} Anomaly Clusters Detected
          </p>
        </div>
        <button
          onClick={onRunAnalysis}
          disabled={loading}
          className="flex items-center gap-1.5 bg-cyan-500/20 hover:bg-cyan-500/30 text-cyan-300 border border-cyan-500/40 px-3 py-1.5 rounded-lg text-xs font-mono font-semibold transition-all cursor-pointer"
        >
          {loading ? <Loader2 size={12} className="animate-spin" /> : <Activity size={12} />}
          <span>Re-Analyze</span>
        </button>
      </div>

      {/* Image Pairing Controls */}
      <div className="space-y-2 bg-slate-900/60 p-3 rounded-lg border border-slate-800">
        <div className="text-[10px] font-mono text-slate-400 uppercase tracking-wider flex items-center justify-between">
          <span>Source Pairing Controls</span>
          <div className="flex items-center gap-3">
            <button
              onClick={onSwap}
              className="flex items-center gap-1 text-[10px] text-cyan-400 hover:text-cyan-300 font-mono cursor-pointer"
            >
              <ArrowLeftRight size={11} />
              <span>Swap A ↔ B</span>
            </button>
            <button
              onClick={onReset}
              className="flex items-center gap-1 text-[10px] text-slate-400 hover:text-slate-200 font-mono cursor-pointer"
            >
              <RotateCcw size={10} />
              <span>Reset</span>
            </button>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-2 pt-1">
          <input ref={fileInputARef} type="file" accept={ACCEPT_FILE_ATTR} onChange={onUploadA} className="hidden" />
          <button
            onClick={() => fileInputARef.current?.click()}
            className="flex items-center justify-center gap-1 text-[11px] font-mono py-1.5 px-2 bg-slate-950 border border-slate-800 hover:border-cyan-500/40 rounded text-slate-300 cursor-pointer"
          >
            <Upload size={11} className="text-cyan-400" />
            <span>Upload Baseline A</span>
          </button>
          <input ref={fileInputBRef} type="file" accept={ACCEPT_FILE_ATTR} onChange={onUploadB} className="hidden" />
          <button
            onClick={() => fileInputBRef.current?.click()}
            className="flex items-center justify-center gap-1 text-[11px] font-mono py-1.5 px-2 bg-slate-950 border border-slate-800 hover:border-cyan-500/40 rounded text-slate-300 cursor-pointer"
          >
            <Upload size={11} className="text-cyan-400" />
            <span>Upload Post-Event B</span>
          </button>
        </div>
      </div>

      {/* Summary Box */}
      <div className="bg-slate-900/80 border border-slate-800 p-3 rounded-lg text-xs text-slate-300 font-sans space-y-1">
        <div className="text-[10px] font-mono text-cyan-400 uppercase">Analysis Summary</div>
        <p className="leading-relaxed">
          {data?.summary || "Analyzing pixel intensity delta across paired satellite scenes..."}
        </p>
      </div>

      {/* Key Metric Cards */}
      <div className="grid grid-cols-2 gap-2 font-mono">
        <div className="bg-slate-900/70 border border-slate-800 p-2.5 rounded-lg">
          <span className="text-[10px] text-slate-500 block uppercase">Changed Pixel Fraction</span>
          <span className="text-base font-bold text-cyan-300">{deltaPct}%</span>
        </div>
        <div className="bg-slate-900/70 border border-slate-800 p-2.5 rounded-lg">
          <span className="text-[10px] text-slate-500 block uppercase">Mean Intensity Delta</span>
          <span className="text-base font-bold text-slate-200">{data?.mean_intensity_delta?.toFixed(1) || "0.0"}</span>
        </div>
      </div>

      {/* Anomaly List */}
      <div className="space-y-2">
        <div className="text-[10px] font-mono text-slate-400 uppercase tracking-wider">
          Detected Anomaly Clusters ({anomalies.length})
        </div>
        <div className="space-y-2 max-h-[180px] overflow-y-auto pr-1">
          {anomalies.map((anom) => (
            <div
              key={anom.id}
              onMouseEnter={() => setHoveredAnomaly(anom.id)}
              onMouseLeave={() => setHoveredAnomaly(null)}
              onClick={() => setActiveAnomaly(activeAnomaly === anom.id ? null : anom.id)}
              className={`p-2.5 rounded-lg border transition-all cursor-pointer font-mono text-xs ${
                activeAnomaly === anom.id || hoveredAnomaly === anom.id
                  ? "bg-amber-500/15 border-amber-400/60 shadow-sm ring-1 ring-amber-500/30"
                  : "bg-slate-900/60 border-slate-800/80 hover:border-slate-700"
              }`}
            >
              <div className="flex items-center justify-between">
                <span className="font-semibold text-slate-200">{anom.label}</span>
                <span className="text-[10px] bg-amber-500/20 text-amber-300 px-1.5 py-0.5 rounded border border-amber-500/30">
                  {anom.severity || "medium"} severity
                </span>
              </div>
              <div className="text-[10px] text-slate-400 mt-1">
                {anom.ground_area ? `Approximate area: ${anom.ground_area.toFixed(1)} ${anom.area_unit || "m²"}` : "Physical area unavailable"}
              </div>
            </div>
          ))}
          {anomalies.length === 0 && (
            <div className="text-xs text-slate-500 font-mono italic p-2 bg-slate-950/40 rounded border border-slate-800/50">
              No significant anomaly clusters exceeding threshold.
            </div>
          )}
        </div>
      </div>

      {/* Visual Heatmap Toggle & Opacity Slider */}
      <div className="space-y-3 pt-2 border-t border-slate-800">
        <div className="flex items-center justify-between">
          <span className="text-xs text-slate-300 font-medium">Differential Heatmap</span>
          <button
            onClick={() => setDeltaHeatmap(!deltaHeatmap)}
            className={`text-[10px] font-mono px-2.5 py-1 rounded border transition-all cursor-pointer ${
              deltaHeatmap
                ? "bg-cyan-500/20 text-cyan-300 border-cyan-500/40 font-bold"
                : "bg-slate-900 text-slate-500 border-slate-800"
            }`}
          >
            {deltaHeatmap ? "VISIBLE" : "HIDDEN"}
          </button>
        </div>

        {deltaHeatmap && (
          <div>
            <div className="flex items-center justify-between text-[10px] font-mono text-slate-400 mb-1">
              <span>Heatmap Blend Opacity</span>
              <span className="text-cyan-400">{opacity}%</span>
            </div>
            <input
              type="range"
              min={0}
              max={100}
              value={opacity}
              onChange={(e) => setOpacity(Number(e.target.value))}
              className="w-full accent-cyan-400 cursor-pointer h-1.5 bg-slate-800 rounded-lg"
            />
          </div>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------
// Comparison Page Main Component
// ---------------------------------------------
export default function ComparisonPage() {
  const [opacity, setOpacity] = useState(75);
  const [deltaHeatmap, setDeltaHeatmap] = useState(true);
  const [activeAnomaly, setActiveAnomaly] = useState<string | null>(null);
  const [hoveredAnomaly, setHoveredAnomaly] = useState<string | null>(null);
  const [zoom, setZoom] = useState(1);
  const [data, setData] = useState<NexSpaceChangeAnalysisResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);

  const [imageA, setImageA] = useState<string>(SAMPLE_CHANGE_A);
  const [labelA, setLabelA] = useState<string>("bitemporal_a.png (Time A)");
  const [imageB, setImageB] = useState<string>(SAMPLE_CHANGE_B);
  const [labelB, setLabelB] = useState<string>("bitemporal_b.png (Time B)");

  const fetchChangeAnalysis = async (imgA = imageA, imgB = imageB) => {
    setLoading(true);
    try {
      const res = await fetch("/api/change-analysis", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image_a: imgA,
          image_b: imgB,
          change_threshold: 0.15,
        }),
      });
      if (res.ok) {
        setBackendOnline(true);
        const result = await res.json();
        setData(result);
      } else {
        setBackendOnline(false);
      }
    } catch (err) {
      console.error("[ComparisonPage] Error fetching change analysis:", err);
      setBackendOnline(false);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    let active = true;
    const loadInitial = async () => {
      setLoading(true);
      try {
        const res = await fetch("/api/change-analysis", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            image_a: SAMPLE_CHANGE_A,
            image_b: SAMPLE_CHANGE_B,
            change_threshold: 0.15,
          }),
        });
        if (res.ok && active) {
          setBackendOnline(true);
          const result = await res.json();
          setData(result);
        } else if (active) {
          setBackendOnline(false);
        }
      } catch (err) {
        console.error("[ComparisonPage] Error fetching change analysis:", err);
        if (active) setBackendOnline(false);
      } finally {
        if (active) setLoading(false);
      }
    };
    loadInitial();
    return () => {
      active = false;
    };
  }, []);

  const handleSwap = () => {
    const nextA = imageB;
    const nextLabelA = labelB;
    const nextB = imageA;
    const nextLabelB = labelA;
    setImageA(nextA);
    setLabelA(nextLabelA);
    setImageB(nextB);
    setLabelB(nextLabelB);
    fetchChangeAnalysis(nextA, nextB);
  };

  const handleUploadA = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const input = e.target;
    const file = input.files?.[0];
    if (!file) return;
    const result = await validateAndProcessImageFile(file);
    if (result.valid && result.source) {
      setImageA(result.source.dataUrl);
      setLabelA(result.source.filename);
      fetchChangeAnalysis(result.source.dataUrl, imageB);
    }
    input.value = "";
  };

  const handleUploadB = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const input = e.target;
    const file = input.files?.[0];
    if (!file) return;
    const result = await validateAndProcessImageFile(file);
    if (result.valid && result.source) {
      setImageB(result.source.dataUrl);
      setLabelB(result.source.filename);
      fetchChangeAnalysis(imageA, result.source.dataUrl);
    }
    input.value = "";
  };

  const handleReset = () => {
    setImageA(SAMPLE_CHANGE_A);
    setLabelA("bitemporal_a.png (Time A)");
    setImageB(SAMPLE_CHANGE_B);
    setLabelB("bitemporal_b.png (Time B)");
    fetchChangeAnalysis(SAMPLE_CHANGE_A, SAMPLE_CHANGE_B);
  };

  return (
    <div className="flex flex-col lg:flex-row min-h-screen bg-[#0a1420] text-white font-sans overflow-x-hidden">
      <Sidebar />

      <div className="flex-1 flex flex-col min-w-0">
        <TopBar
          deltaPct={data?.changed_fraction}
          backendOnline={backendOnline}
        />

        <main className="flex-1 p-4 sm:p-6 lg:p-8 grid grid-cols-1 lg:grid-cols-3 gap-5 items-start overflow-y-auto">
          <div className="lg:col-span-2">
            <ImageViewer
              opacity={opacity}
              deltaHeatmap={deltaHeatmap}
              heatmapImage={data?.heatmap_image || data?.overlay_image}
              activeAnomaly={activeAnomaly}
              hoveredAnomaly={hoveredAnomaly}
              setActiveAnomaly={setActiveAnomaly}
              zoom={zoom}
              setZoom={setZoom}
              imageA={imageA}
              imageB={imageB}
              labelA={labelA}
              labelB={labelB}
              anomalies={data?.anomalies || []}
            />
          </div>
          <div className="lg:col-span-1">
            <AnalysisPanel
              opacity={opacity}
              setOpacity={setOpacity}
              deltaHeatmap={deltaHeatmap}
              setDeltaHeatmap={setDeltaHeatmap}
              activeAnomaly={activeAnomaly}
              hoveredAnomaly={hoveredAnomaly}
              setActiveAnomaly={setActiveAnomaly}
              setHoveredAnomaly={setHoveredAnomaly}
              data={data}
              loading={loading}
              onRunAnalysis={() => fetchChangeAnalysis()}
              onSwap={handleSwap}
              onReset={handleReset}
              onUploadA={handleUploadA}
              onUploadB={handleUploadB}
            />
          </div>
        </main>
      </div>
    </div>
  );
}