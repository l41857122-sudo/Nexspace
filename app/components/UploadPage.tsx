"use client";

import { useState, useRef } from "react";
import {
  FileUp,
  CheckCircle2,
  Loader2,
  AlertTriangle,
  UploadCloud,
  FileCheck,
  Radio,
  Layers,
  Database,
  ArrowRight,
  FolderOpen,
  Scan,
} from "lucide-react";
import Sidebar from "./Sidebar";

export default function UploadPage() {
  const [fileName, setFileName] = useState("S2A_MSIL2A_20240315T105341.zip");
  const [fileSize, setFileSize] = useState("4.8 GB");
  const [progress, setProgress] = useState(44);
  const [uploadStatus, setUploadStatus] = useState<"idle" | "uploading" | "completed">("uploading");
  const [statusMessage, setStatusMessage] = useState("Validating projection (EPSG:32651)...");
  const [datasetId, setDatasetId] = useState("LB_08_2024_03_RegionAlpha");
  const [isDragging, setIsDragging] = useState(false);
  const [bands, setBands] = useState([
    { label: "RGB (B4,B3,B2)", checked: true },
    { label: "NIR (B8)", checked: true },
    { label: "SWIR (B11,B12)", checked: false },
  ]);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const startUploadSimulation = async (name: string, size: string) => {
    setFileName(name);
    setFileSize(size);
    setDatasetId(name.replace(/\.[^/.]+$/, ""));
    setProgress(10);
    setUploadStatus("uploading");
    setStatusMessage("Reading geospatial raster header...");

    try {
      const res = await fetch("/api/scenes/upload", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ fileName: name, fileSize: size })
      });
      const job = await res.json();
      const jobId = job.id;

      const interval = setInterval(async () => {
        try {
          const statusRes = await fetch(`/api/scenes/${jobId}/status`);
          const statusData = await statusRes.json();
          setProgress(statusData.progress);
          setStatusMessage(statusData.statusMessage);
          if (statusData.status === "completed" || statusData.progress >= 100) {
            setUploadStatus("completed");
            clearInterval(interval);
          }
        } catch (e) {
          clearInterval(interval);
        }
      }, 500);
    } catch (err) {
      console.error(err);
    }
  };

  const handleFileSelect = (files: FileList | null) => {
    if (files && files.length > 0) {
      const file = files[0];
      const sizeStr = (file.size / (1024 * 1024 * 1024)).toFixed(1) + " GB";
      startUploadSimulation(file.name, sizeStr);
    }
  };

  const toggleBand = (index: number) => {
    setBands((prev) =>
      prev.map((b, i) => (i === index ? { ...b, checked: !b.checked } : b))
    );
  };

  return (
    <div className="flex flex-col lg:flex-row min-h-screen bg-[#0a1420] text-white font-sans overflow-x-hidden">
      <Sidebar />

      <div className="flex-1 flex flex-col min-w-0">
        {/* Header */}
        <header className="px-4 sm:px-6 lg:px-8 py-4 border-b border-slate-800/80 bg-[#09131f]/60 backdrop-blur-md flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div>
            <div className="flex items-center gap-2">
              <div className="p-1.5 rounded-lg bg-cyan-500/10 border border-cyan-500/20 text-cyan-400">
                <Database size={15} />
              </div>
              <h1 className="text-sm font-semibold text-white tracking-tight">
                Ingest Orbital Data
              </h1>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
                Raster Pipeline
              </span>
            </div>
            <p className="text-[11px] text-slate-400 font-mono mt-0.5">
              Automated metadata extraction &amp; Cloud-Optimized GeoTIFF generation
            </p>
          </div>

          <div className="flex items-center gap-1.5 text-[11px] font-mono text-emerald-400 bg-emerald-500/10 border border-emerald-500/20 px-2.5 py-1 rounded-lg self-end sm:self-auto">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
            <span>Ingest Node: Ready</span>
          </div>
        </header>

        <main className="flex-1 p-4 sm:p-6 lg:p-8 grid grid-cols-1 lg:grid-cols-3 gap-5 items-start overflow-y-auto">
          {/* Main Column */}
          <div className="lg:col-span-2 space-y-5">
            {/* Drag & Drop Zone */}
            <div
              onDragOver={(e) => {
                e.preventDefault();
                setIsDragging(true);
              }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={(e) => {
                e.preventDefault();
                setIsDragging(false);
                handleFileSelect(e.dataTransfer.files);
              }}
              onClick={() => fileInputRef.current?.click()}
              className={`relative border-2 border-dashed rounded-xl flex flex-col items-center justify-center py-10 px-4 text-center cursor-pointer transition-all duration-200 group overflow-hidden ${
                isDragging
                  ? "border-cyan-400 bg-cyan-500/15 shadow-[0_0_30px_rgba(6,182,212,0.3)] scale-[1.005]"
                  : "border-cyan-500/30 hover:border-cyan-400/80 bg-[#0c1624]/60 hover:bg-[#0c1624]/90 backdrop-blur-md shadow-[0_4px_24px_rgba(0,0,0,0.3)] hover:shadow-[0_0_20px_rgba(6,182,212,0.15)]"
              }`}
            >
              {/* Subtle Scanning line during drag or processing */}
              {(isDragging || uploadStatus === "uploading") && (
                <div className="absolute inset-0 overflow-hidden pointer-events-none">
                  <div
                    className="w-full h-[2px] bg-gradient-to-r from-transparent via-cyan-400/60 to-transparent animate-scan"
                    style={{ animationDuration: "3.5s" }}
                  />
                </div>
              )}

              <input
                type="file"
                ref={fileInputRef}
                onChange={(e) => handleFileSelect(e.target.files)}
                className="hidden"
                accept=".tif,.tiff,.jp2,.nc,.zip"
              />
              <div className="p-3.5 rounded-full bg-cyan-500/10 border border-cyan-500/25 text-cyan-400 mb-3 group-hover:scale-110 group-hover:border-cyan-400/60 group-hover:bg-cyan-500/20 transition-all duration-200 shadow-[0_0_15px_rgba(6,182,212,0.25)]">
                <UploadCloud size={28} />
              </div>
              <p className="text-sm font-semibold text-white tracking-tight group-hover:text-cyan-300 transition-colors">
                Drag &amp; Drop Imagery or Browse Local Files
              </p>
              <p className="text-[11px] text-slate-400 mt-1 font-mono">
                Supported formats: GeoTIFF, JP2, NetCDF, Zip (Max 5GB per archive)
              </p>
              <button
                type="button"
                className="mt-4 inline-flex items-center gap-1.5 bg-gradient-to-r from-cyan-500 to-sky-400 group-hover:from-cyan-400 group-hover:to-sky-300 text-[#071320] text-xs font-bold rounded-lg px-4 py-2 pointer-events-none transition-all duration-200 shadow-[0_0_12px_rgba(6,182,212,0.35)] group-hover:shadow-[0_0_18px_rgba(6,182,212,0.55)] group-hover:scale-[1.02]"
              >
                <FolderOpen size={13} />
                <span>Browse Local Files</span>
              </button>
            </div>

            {/* Dataset Configuration */}
            <div className="w-full border border-slate-800/90 bg-[#0c1624]/60 backdrop-blur-md rounded-xl p-4 sm:p-5 shadow-[0_4px_24px_rgba(0,0,0,0.3)]">
              <p className="text-sm font-semibold text-white tracking-tight mb-4 flex items-center gap-2">
                <Layers size={14} className="text-cyan-400" />
                <span>Dataset Configuration</span>
              </p>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div>
                  <label className="text-[10px] text-slate-400 uppercase tracking-widest font-mono">
                    Dataset Identifier
                  </label>
                  <input
                    type="text"
                    value={datasetId}
                    onChange={(e) => setDatasetId(e.target.value)}
                    placeholder="e.g. LB_08_2024_03_RegionAlpha"
                    className="mt-1.5 w-full bg-slate-900/80 border border-slate-800 rounded-lg px-3 py-2 text-xs text-slate-200 font-mono placeholder:text-slate-600 focus:outline-none focus:border-cyan-500/50 focus:shadow-[0_0_0_2px_rgba(6,182,212,0.1)] transition-all duration-180"
                  />
                </div>
                <div>
                  <label className="text-[10px] text-slate-400 uppercase tracking-widest font-mono">
                    Mission Source
                  </label>
                  <div className="mt-1.5 bg-slate-900/80 border border-slate-800 rounded-lg px-3 py-2 text-xs text-slate-300 font-mono flex items-center justify-between">
                    <span>Sentinel-2 MSI (Auto-detected)</span>
                    <FileCheck size={14} className="text-emerald-400" />
                  </div>
                </div>
              </div>

              <div className="mt-4 pt-3 border-t border-slate-800/60">
                <label className="text-[10px] text-slate-400 uppercase tracking-widest font-mono">
                  Spectral Bands (Selected)
                </label>
                <div className="flex flex-wrap items-center gap-4 mt-2">
                  {bands.map((b, idx) => (
                    <label
                      key={b.label}
                      className="group flex items-center gap-2 text-xs text-slate-300 hover:text-cyan-300 cursor-pointer select-none transition-colors duration-150"
                    >
                      <input
                        type="checkbox"
                        checked={b.checked}
                        onChange={() => toggleBand(idx)}
                        className="accent-cyan-500 rounded cursor-pointer transition-transform duration-150 group-hover:scale-110"
                      />
                      <span>{b.label}</span>
                    </label>
                  ))}
                </div>
              </div>
            </div>

            {/* Active upload progress */}
            <div className="w-full border border-slate-800/90 bg-[#0c1624]/60 backdrop-blur-md rounded-xl p-4 sm:p-5 shadow-[0_4px_24px_rgba(0,0,0,0.3)]">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-1 mb-2">
                <p className="text-xs text-slate-200 font-mono font-medium truncate max-w-[280px]">
                  {fileName}
                </p>
                <span className="text-[11px] text-slate-400 font-mono">
                  {((parseFloat(fileSize) * progress) / 100).toFixed(1)} GB / {fileSize}
                </span>
              </div>

              <div className="flex items-center gap-2 mb-2">
                {uploadStatus === "completed" ? (
                  <CheckCircle2 size={13} className="text-emerald-400" />
                ) : (
                  <Loader2 size={12} className="text-cyan-400 animate-spin" />
                )}
                <span className={`text-[11px] font-mono ${uploadStatus === "completed" ? "text-emerald-300 font-semibold" : "text-cyan-300"}`}>
                  {uploadStatus === "completed" ? "Ingestion Complete" : `Processing · ${progress}%`}
                </span>
              </div>

              <div className="h-1.5 rounded-full bg-slate-800 overflow-hidden mb-3">
                <div
                  className={`h-full rounded-full transition-all duration-300 ease-out ${
                    uploadStatus === "completed" ? "bg-emerald-400 shadow-[0_0_10px_rgba(52,211,153,0.8)]" : "bg-gradient-to-r from-cyan-500 to-sky-400 shadow-[0_0_10px_rgba(6,182,212,0.8)]"
                  }`}
                  style={{ width: `${progress}%` }}
                />
              </div>

              <div className="flex flex-wrap items-center gap-2 text-[11px] text-slate-400 font-mono">
                <span className="text-cyan-400">&gt;</span>
                <span>{statusMessage}</span>
              </div>
            </div>
          </div>

          {/* Side Column: Validation & Recents */}
          <div className="lg:col-span-1 space-y-5">
            {/* Validation Checklist */}
            <div className="w-full border border-slate-800/90 bg-[#0c1624]/60 backdrop-blur-md rounded-xl p-4 sm:p-5 shadow-[0_4px_24px_rgba(0,0,0,0.3)]">
              <p className="text-sm font-semibold text-white tracking-tight mb-3 flex items-center gap-2">
                <CheckCircle2 size={14} className="text-cyan-400" />
                <span>Validation Checklist</span>
              </p>

              <div className="space-y-1">
                <div className="flex items-start gap-2.5 py-2.5 border-b border-slate-800/60 transition-colors duration-150 hover:bg-cyan-500/[0.02] px-1 rounded-lg">
                  <CheckCircle2 size={14} className="text-emerald-400 mt-0.5 shrink-0" />
                  <div>
                    <p className="text-xs text-slate-200 font-medium">Coordinate Reference System</p>
                    <p className="text-[10px] text-slate-400 uppercase tracking-wide mt-0.5 font-mono">
                      Detected · Valid UTM (EPSG:32651)
                    </p>
                  </div>
                </div>

                <div className="flex items-start gap-2.5 py-2.5 border-b border-slate-800/60 transition-colors duration-150 hover:bg-cyan-500/[0.02] px-1 rounded-lg">
                  {progress >= 60 ? (
                    <CheckCircle2 size={14} className="text-emerald-400 mt-0.5 shrink-0" />
                  ) : (
                    <Loader2 size={14} className="text-cyan-400 animate-spin mt-0.5 shrink-0" />
                  )}
                  <div>
                    <p className="text-xs text-slate-200 font-medium">Spatial Resolution</p>
                    <p className="text-[10px] text-slate-400 uppercase tracking-wide mt-0.5 font-mono">
                      {progress >= 60 ? "Extracted · 10m GSD" : "Analyzing · Extraction"}
                    </p>
                  </div>
                </div>

                <div className="flex items-start gap-2.5 py-2.5 transition-colors duration-150 hover:bg-cyan-500/[0.02] px-1 rounded-lg">
                  {progress === 100 ? (
                    <CheckCircle2 size={14} className="text-emerald-400 mt-0.5 shrink-0" />
                  ) : (
                    <Loader2 size={14} className="text-slate-500 animate-spin mt-0.5 shrink-0" />
                  )}
                  <div>
                    <p className="text-xs text-slate-200 font-medium">Cloud Cover Assessment</p>
                    <p className="text-[10px] text-slate-400 uppercase tracking-wide mt-0.5 font-mono">
                      {progress === 100 ? "Verified · 4.2% Cloud" : "Awaiting QA · Band Analysis"}
                    </p>
                  </div>
                </div>
              </div>
            </div>

            {/* Recent Ingests */}
            <div className="w-full border border-slate-800/90 bg-[#0c1624]/60 backdrop-blur-md rounded-xl p-4 sm:p-5 shadow-[0_4px_24px_rgba(0,0,0,0.3)]">
              <p className="text-sm font-semibold text-white tracking-tight mb-3">Recent Ingests</p>

              <div className="space-y-1">
                <div className="flex items-start gap-2.5 py-2.5 border-b border-slate-800/60 transition-all duration-150 hover:bg-cyan-500/[0.03] px-1.5 rounded-lg group cursor-default">
                  <CheckCircle2 size={14} className="text-emerald-400 mt-0.5 shrink-0" />
                  <div className="flex-1 min-w-0">
                    <p className="text-xs text-slate-200 font-mono truncate group-hover:text-cyan-300 transition-colors">
                      LC08_L1TP_195023_20231...
                    </p>
                    <p className="text-[10px] text-slate-400 mt-0.5 font-mono">
                      12 min ago · Landsat 8
                    </p>
                  </div>
                </div>

                <div className="flex items-start gap-2.5 py-2.5 border-b border-slate-800/60 transition-all duration-150 hover:bg-red-500/[0.03] px-1.5 rounded-lg group cursor-default">
                  <AlertTriangle size={14} className="text-red-400 mt-0.5 shrink-0" />
                  <div className="flex-1 min-w-0">
                    <p className="text-xs text-slate-200 font-mono truncate group-hover:text-red-300 transition-colors">
                      PlanetScope_Analytics_2023...
                    </p>
                    <p className="text-[10px] text-red-400 mt-0.5 font-mono">
                      Error: Corrupt metadata
                    </p>
                  </div>
                </div>
              </div>

              <button
                onClick={() => startUploadSimulation("PlanetScope_2024_08_ReScan.tif", "1.6 GB")}
                className="mt-3.5 w-full flex items-center justify-center gap-1.5 text-xs text-cyan-400 hover:text-cyan-300 font-mono py-2 rounded-lg bg-slate-900/60 hover:bg-slate-900/90 border border-slate-800 hover:border-cyan-500/30 transition-all duration-180 hover:scale-[1.02] active:scale-[0.98] cursor-pointer"
              >
                <span>+ Quick Ingest Sample</span>
              </button>
            </div>
          </div>
        </main>

        <footer className="px-4 sm:px-6 pb-4 mt-auto">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 px-1 pt-2 text-[10px] font-mono text-slate-400 border-t border-slate-800/80">
            <div className="flex items-center gap-1.5 text-emerald-400">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
              <span>Status: AI Ingest Pipeline Active</span>
            </div>
            <div>LAT: 34.0522° N · LON: −118.2437° W · ELEV: 120M</div>
          </div>
        </footer>
      </div>
    </div>
  );
}