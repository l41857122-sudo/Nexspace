"use client";

import { useState, useEffect } from "react";
import {
  Satellite,
  LayoutGrid,
  Search,
  ArrowRight,
  RefreshCw,
  Sparkles,
  Bell,
  UserCircle,
  Clock,
  Radio,
  Cpu,
  HardDrive,
  Activity,
  Filter,
} from "lucide-react";
import Sidebar from "./Sidebar";

// ---------------------------------------------
// Top bar
// ---------------------------------------------
function TopBar() {
  const [searchValue, setSearchValue] = useState("");

  return (
    <header className="flex flex-col md:flex-row md:items-center justify-between gap-4 px-4 sm:px-6 lg:px-8 py-4 border-b border-slate-800/80 bg-[#09131f]/60 backdrop-blur-md">
      <div>
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse shadow-[0_0_8px_rgba(6,182,212,0.8)]" />
          <p className="text-[10px] tracking-widest text-slate-400 uppercase font-mono">
            Workspace: NexSpace · Global Ops
          </p>
        </div>
        <h1 className="text-base font-semibold text-white mt-0.5 tracking-tight flex items-center gap-2">
          <span>Project Aegis</span>
          <span className="text-[10px] font-mono font-normal text-cyan-400 bg-cyan-500/10 border border-cyan-500/20 px-2 py-0.5 rounded-full">
            Active Monitoring
          </span>
        </h1>
      </div>

      <div className="flex flex-wrap items-center gap-2.5 sm:gap-3">
        {/* Search Bar */}
        <div className="flex items-center gap-2 bg-[#09121d] border border-slate-800 focus-within:border-cyan-500/40 focus-within:shadow-[0_0_12px_rgba(6,182,212,0.15)] rounded-lg px-3 py-2 text-xs text-slate-400 w-full sm:w-64 lg:w-72 transition-all duration-180">
          <Search size={14} className="shrink-0 text-slate-400" />
          <input
            type="text"
            value={searchValue}
            onChange={(e) => setSearchValue(e.target.value)}
            placeholder="Search coordinates, datasets, missions..."
            className="bg-transparent text-slate-200 placeholder:text-slate-500 focus:outline-none w-full text-xs"
          />
        </div>

        {/* New Analysis Button */}
        <button className="inline-flex items-center gap-1.5 bg-gradient-to-r from-cyan-500 to-sky-400 hover:from-cyan-400 hover:to-sky-300 text-[#071320] text-xs font-bold rounded-lg px-3.5 py-2 transition-all duration-180 shadow-[0_0_12px_rgba(6,182,212,0.3)] hover:shadow-[0_0_18px_rgba(6,182,212,0.5)] hover:scale-[1.02] active:scale-[0.98] cursor-pointer">
          <Sparkles size={13} className="text-[#071320]" />
          <span>+ New Analysis</span>
        </button>

        {/* Actions Group */}
        <div className="flex items-center gap-2 border-l border-slate-800 pl-2.5 ml-auto sm:ml-0">
          <button
            className="p-2 rounded-lg bg-slate-900/60 border border-slate-800 text-slate-400 hover:text-cyan-300 hover:border-cyan-500/30 transition-all duration-180 active:scale-95 cursor-pointer"
            aria-label="Refresh telemetry data"
            title="Refresh telemetry data"
          >
            <RefreshCw size={14} />
          </button>
          <button
            className="relative p-2 rounded-lg bg-slate-900/60 border border-slate-800 text-slate-400 hover:text-cyan-300 hover:border-cyan-500/30 transition-all duration-180 active:scale-95 cursor-pointer"
            aria-label="Notifications"
            title="Notifications"
          >
            <Bell size={14} />
            <span className="absolute top-1.5 right-1.5 w-1.5 h-1.5 bg-cyan-400 rounded-full" />
          </button>
          <button className="p-1 rounded-lg hover:bg-white/5 transition-colors cursor-pointer" aria-label="User profile">
            <UserCircle size={22} className="text-slate-400 hover:text-slate-200" />
          </button>
        </div>
      </div>
    </header>
  );
}

// ---------------------------------------------
// Integrated KPI Telemetry Strip
// ---------------------------------------------
function KpiTelemetryStrip() {
  return (
    <div className="w-full border border-slate-800/90 bg-[#0c1624]/70 backdrop-blur-md rounded-xl grid grid-cols-2 lg:grid-cols-4 divide-y sm:divide-y-0 sm:divide-x divide-slate-800/80 mb-6 shadow-[0_4px_24px_rgba(0,0,0,0.3)]">
      {/* 1. Active Missions */}
      <div className="p-4 sm:p-5 flex flex-col justify-between group hover:bg-cyan-500/[0.02] transition-colors">
        <div className="flex items-center justify-between">
          <span className="text-[10px] text-slate-400 group-hover:text-slate-300 uppercase tracking-widest font-mono">
            Active Missions
          </span>
          <div className="p-1.5 rounded-md bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
            <LayoutGrid size={14} />
          </div>
        </div>
        <div className="mt-2.5">
          <p className="text-2xl sm:text-3xl font-bold text-white tracking-tight">
            12
          </p>
          <p className="text-[10px] font-mono text-emerald-400 flex items-center gap-1 mt-1">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
            All telemetry nominal
          </p>
        </div>
      </div>

      {/* 2. Total Area Scanned */}
      <div className="p-4 sm:p-5 flex flex-col justify-between group hover:bg-cyan-500/[0.02] transition-colors">
        <div className="flex items-center justify-between">
          <span className="text-[10px] text-slate-400 group-hover:text-slate-300 uppercase tracking-widest font-mono">
            Total Area Scanned
          </span>
          <div className="p-1.5 rounded-md bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
            <Satellite size={14} />
          </div>
        </div>
        <div className="mt-2.5">
          <p className="text-2xl sm:text-3xl font-bold text-white tracking-tight">
            48.5M <span className="text-xs font-mono font-normal text-slate-400">km²</span>
          </p>
          <p className="text-[10px] font-mono text-cyan-400 mt-1">
            +3.2M km² this cycle
          </p>
        </div>
      </div>

      {/* 3. Compute Utilization */}
      <div className="p-4 sm:p-5 flex flex-col justify-between group hover:bg-cyan-500/[0.02] transition-colors">
        <div className="flex items-center justify-between">
          <span className="text-[10px] text-slate-400 group-hover:text-slate-300 uppercase tracking-widest font-mono">
            Compute Utilization
          </span>
          <div className="p-1.5 rounded-md bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
            <RefreshCw size={14} />
          </div>
        </div>
        <div className="mt-2.5">
          <div className="flex items-baseline justify-between">
            <p className="text-2xl sm:text-3xl font-bold text-white tracking-tight">
              87%
            </p>
            <span className="text-[10px] font-mono text-cyan-400">High Load</span>
          </div>
          <div className="h-1 w-full bg-slate-800 rounded-full overflow-hidden mt-2">
            <div
              className="h-full bg-gradient-to-r from-cyan-500 to-emerald-400 rounded-full transition-all duration-300"
              style={{ width: "87%" }}
            />
          </div>
        </div>
      </div>

      {/* 4. Success Rate */}
      <div className="p-4 sm:p-5 flex flex-col justify-between group hover:bg-cyan-500/[0.02] transition-colors">
        <div className="flex items-center justify-between">
          <span className="text-[10px] text-slate-400 group-hover:text-slate-300 uppercase tracking-widest font-mono">
            Success Rate
          </span>
          <div className="p-1.5 rounded-md bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
            <ArrowRight size={14} />
          </div>
        </div>
        <div className="mt-2.5">
          <p className="text-2xl sm:text-3xl font-bold text-emerald-400 tracking-tight">
            99.2%
          </p>
          <p className="text-[10px] font-mono text-slate-400 mt-1">
            0.08% false positive RMS
          </p>
        </div>
      </div>
    </div>
  );
}

function StatusRow({
  label,
  value,
  icon: Icon,
}: {
  label: string;
  value: number;
  icon?: React.ElementType;
}) {
  const color =
    value > 80 ? "bg-emerald-400" : value > 50 ? "bg-amber-400" : "bg-red-400";
  const badgeColor =
    value > 80
      ? "text-emerald-400 bg-emerald-500/10 border-emerald-500/20"
      : value > 50
      ? "text-amber-400 bg-amber-500/10 border-amber-500/20"
      : "text-red-400 bg-red-500/10 border-red-500/20";

  return (
    <div className="flex flex-col gap-1.5 py-2.5 border-b border-slate-800/60 last:border-b-0">
      <div className="flex items-center justify-between text-xs">
        <div className="flex items-center gap-2 text-slate-300">
          <span className={`w-1.5 h-1.5 rounded-full ${color} animate-pulse`} />
          {Icon && <Icon size={13} className="text-slate-400" />}
          <span className="font-medium">{label}</span>
        </div>
        <span
          className={`text-[10px] font-mono font-semibold px-2 py-0.5 rounded border ${badgeColor}`}
        >
          {value}%
        </span>
      </div>

      <div className="h-1 w-full bg-slate-800/80 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-300 opacity-90 ${
            value > 80
              ? "bg-emerald-400"
              : value > 50
              ? "bg-amber-400"
              : "bg-red-400"
          }`}
          style={{ width: `${value}%` }}
        />
      </div>
    </div>
  );
}

function SensorRow({ name, meta }: { name: string; meta: string }) {
  return (
    <div className="flex items-center justify-between py-2 border-b border-slate-800/60 last:border-b-0 text-xs group hover:bg-cyan-500/[0.02] px-2 rounded-lg transition-colors duration-150">
      <div>
        <p className="text-slate-200 font-medium group-hover:text-cyan-300 transition-colors">
          {name}
        </p>
        <p className="text-slate-400 text-[11px] font-mono mt-0.5">{meta}</p>
      </div>
      <div className="p-1 rounded bg-slate-800/80 text-slate-400 group-hover:text-cyan-400 transition-colors">
        <Radio size={12} className="animate-pulse" />
      </div>
    </div>
  );
}

type AnalysisStatus = "Processing" | "Completed" | "Awaiting QA" | "Error";

const statusStyles: Record<AnalysisStatus, string> = {
  Processing: "bg-cyan-500/15 text-cyan-300 border-cyan-500/30",
  Completed: "bg-emerald-500/15 text-emerald-300 border-emerald-500/30",
  "Awaiting QA": "bg-amber-500/15 text-amber-300 border-amber-500/30",
  Error: "bg-red-500/15 text-red-300 border-red-500/30",
};

import type { CanonicalInvestigationState } from "../types/nexspace";
import { getCurrentInvestigation } from "../utils/investigationStorage";

interface AnalysisItem {
  name: string;
  type: string;
  status: AnalysisStatus;
  detail: string;
  metadata: string;
}

function RecentAnalyses() {
  const [analyses, setAnalyses] = useState<AnalysisItem[]>([]);
  const [liveInvestigation, setLiveInvestigation] = useState<CanonicalInvestigationState | null>(null);

  useEffect(() => {
    const inv = getCurrentInvestigation();
    if (inv) setLiveInvestigation(inv);

    fetch("/api/analyses")
      .then((res) => res.json())
      .then((data) => {
        if (Array.isArray(data)) setAnalyses(data);
      })
      .catch((err) => console.error(err));
  }, []);

  return (
    <div className="w-full border border-slate-800/90 bg-[#0c1624]/60 backdrop-blur-md rounded-xl p-4 sm:p-5 flex flex-col justify-between shadow-[0_4px_24px_rgba(0,0,0,0.3)]">
      <div className="flex items-center justify-between mb-4">
        <div>
          <div className="flex items-center gap-2">
            <h3 className="text-sm font-semibold text-white tracking-tight">
              Recent Analyses
            </h3>
            <span className="text-[9px] font-mono px-2 py-0.5 rounded-full bg-slate-800 text-slate-300 border border-slate-700">
              {liveInvestigation ? "LIVE SESSION + CATALOG" : "CATALOG DEMO"}
            </span>
          </div>
          <p className="text-[11px] text-slate-400 font-mono mt-0.5">
            Active &amp; completed orbital extraction jobs
          </p>
        </div>
        <button className="flex items-center gap-1.5 text-xs text-slate-400 hover:text-white bg-slate-900/80 border border-slate-800 px-2.5 py-1.5 rounded-lg transition-colors cursor-pointer">
          <Filter size={12} className="text-cyan-400" />
          <span>Filter</span>
        </button>
      </div>

      <div className="overflow-x-auto w-full">
        <table className="w-full text-xs min-w-[550px]">
          <thead>
            <tr className="text-slate-400 text-left uppercase tracking-wider font-mono text-[10px] border-b border-slate-800 pb-2">
              <th className="font-normal pb-2.5 pl-2">Mission / Name</th>
              <th className="font-normal pb-2.5">Type</th>
              <th className="font-normal pb-2.5">Status</th>
              <th className="font-normal pb-2.5">Metadata</th>
              <th className="font-normal pb-2.5 pr-2 text-right">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/60">
            {liveInvestigation && (
              <tr className="bg-cyan-500/10 hover:bg-cyan-500/15 transition-colors duration-150 group">
                <td className="py-3 pl-2 pr-3 text-cyan-200 font-semibold max-w-[240px] truncate">
                  [LIVE] {liveInvestigation.query}
                </td>
                <td className="py-3 pr-3 text-cyan-300 font-mono text-[11px]">
                  Agent Grounding &amp; VQA
                </td>
                <td className="py-3 pr-3">
                  <span className="inline-flex items-center gap-1 rounded-md px-2.5 py-0.5 text-[10px] font-mono border font-semibold bg-emerald-500/20 text-emerald-300 border-emerald-500/40">
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                    Live Session
                  </span>
                </td>
                <td className="py-3 pr-3 text-slate-300 font-mono text-[11px]">
                  {liveInvestigation.source_image?.filename || "Active Raster"} | {liveInvestigation.response?.grounding?.detections?.length || 0} Detections
                </td>
                <td className="py-3 pr-2 text-right">
                  <a href="/results" className="p-1 rounded hover:bg-cyan-500/20 text-cyan-400 inline-block">
                    <ArrowRight size={13} />
                  </a>
                </td>
              </tr>
            )}
            {analyses.map((row) => (
              <tr
                key={row.name}
                className="hover:bg-cyan-500/[0.04] transition-colors duration-150 group"
              >
                <td className="py-3 pl-2 pr-3 text-slate-200 font-medium max-w-[240px] truncate group-hover:text-cyan-300 transition-colors">
                  {row.name}
                </td>
                <td className="py-3 pr-3 text-slate-400 font-mono text-[11px]">
                  {row.type}
                </td>
                <td className="py-3 pr-3">
                  <span
                    className={`inline-flex items-center gap-1 rounded-md px-2.5 py-0.5 text-[10px] font-mono border font-semibold ${statusStyles[row.status] || statusStyles.Processing}`}
                  >
                    {row.status}
                    {row.detail && row.status === "Processing"
                      ? ` (${row.detail})`
                      : ""}
                  </span>
                  {row.status === "Error" && (
                    <p className="text-[10px] text-red-400 font-mono mt-0.5">
                      {row.detail}
                    </p>
                  )}
                </td>
                <td className="py-3 pr-3 text-slate-400 font-mono text-[11px]">
                  {row.metadata}
                </td>
                <td className="py-3 pr-2 text-right">
                  <button className="p-1 rounded hover:bg-cyan-500/20 text-slate-400 hover:text-cyan-300 transition-colors cursor-pointer" aria-label="Inspect row">
                    <ArrowRight size={13} />
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

interface QueryItem {
  text: string;
  time: string | null;
  status: string;
}

function RecentQueries() {
  const [queriesList, setQueriesList] = useState<QueryItem[]>([]);
  const [liveInvestigation, setLiveInvestigation] = useState<CanonicalInvestigationState | null>(null);

  useEffect(() => {
    const inv = getCurrentInvestigation();
    if (inv) setLiveInvestigation(inv);

    fetch("/api/queries/recent")
      .then((res) => res.json())
      .then((data) => {
        if (Array.isArray(data)) setQueriesList(data);
      })
      .catch((err) => console.error(err));
  }, []);

  return (
    <div className="w-full border border-slate-800/90 bg-[#0c1624]/60 backdrop-blur-md rounded-xl p-4 sm:p-5 flex flex-col justify-between shadow-[0_4px_24px_rgba(0,0,0,0.3)]">
      <div className="mb-4">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold text-white tracking-tight">
            Recent NLP Queries
          </h3>
          <span className="text-[9px] font-mono text-slate-400">
            {liveInvestigation ? "1 Active Session" : "Standby"}
          </span>
        </div>
        <p className="text-[11px] text-slate-400 font-mono mt-0.5">
          Natural language spatial searches
        </p>
      </div>

      <div className="space-y-2.5">
        {liveInvestigation && (
          <a
            href="/results"
            className="block p-3 rounded-lg bg-cyan-950/40 border border-cyan-500/40 hover:border-cyan-400 transition-all duration-180 group"
          >
            <div className="flex items-center justify-between text-[10px] font-mono text-emerald-400 mb-1">
              <span className="flex items-center gap-1 font-semibold">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                ACTIVE LIVE INVESTIGATION
              </span>
              <span className="text-cyan-300">{liveInvestigation.investigation_id}</span>
            </div>
            <p className="text-xs text-white font-medium italic leading-relaxed">
              "{liveInvestigation.query}"
            </p>
            <div className="flex items-center justify-between text-[10px] font-mono text-slate-300 mt-2 pt-1.5 border-t border-cyan-500/20">
              <span className="truncate max-w-[180px] text-slate-400">
                Raster: {liveInvestigation.source_image?.filename}
              </span>
              <span className="text-cyan-300 group-hover:translate-x-0.5 transition-transform flex items-center gap-1 font-semibold">
                <span>View Results</span>
                <ArrowRight size={10} />
              </span>
            </div>
          </a>
        )}

        {queriesList.map((q, i) => (
          <div
            key={i}
            className="p-3 rounded-lg bg-slate-900/50 border border-slate-800/80 hover:border-cyan-500/30 hover:bg-cyan-500/[0.03] transition-all duration-180 group cursor-pointer"
          >
            <p className="text-xs text-slate-300 italic leading-relaxed group-hover:text-slate-100 transition-colors">
              "{q.text}"
            </p>
            <div className="flex items-center justify-between text-[10px] font-mono text-slate-400 mt-2.5 pt-2 border-t border-slate-800/60">
              <span className="flex items-center gap-1">
                <Clock size={11} className="text-cyan-400" />
                {q.time ? `Exec Time: ${q.time}` : "In Progress..."}
              </span>
              <span className="text-cyan-400 group-hover:translate-x-0.5 transition-transform flex items-center gap-1 font-semibold">
                <span>View</span>
                <ArrowRight size={10} />
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// -------------------------------------------------------------
// PRIMARY VISUAL FOCUS: Global Scan Coverage (Integrated HUD)
// -------------------------------------------------------------
function GlobeCoveragePanel() {
  return (
    <div className="w-full border border-cyan-500/25 bg-[#0c1624]/75 backdrop-blur-md rounded-xl p-4 sm:p-6 relative overflow-hidden min-h-[350px] flex flex-col justify-between shadow-[0_0_40px_rgba(6,182,212,0.08)]">
      {/* Background HUD Grid Accent */}
      <div
        className="absolute inset-0 opacity-15 pointer-events-none"
        style={{
          backgroundImage:
            "linear-gradient(rgba(56,189,248,0.1) 1px, transparent 1px), linear-gradient(90deg, rgba(56,189,248,0.1) 1px, transparent 1px)",
          backgroundSize: "32px 32px",
        }}
      />

      <div className="flex items-center justify-between mb-4 relative z-10">
        <div>
          <div className="flex items-center gap-2">
            <h2 className="text-base font-bold text-white tracking-tight">
              Global Scan Coverage
            </h2>
            <span className="text-[10px] font-mono text-slate-400">
              · Project Aegis
            </span>
          </div>
          <p className="text-[11px] text-slate-400 font-mono mt-0.5">
            Real-time orbital tracking &amp; sensor footprint
          </p>
        </div>

        <div className="flex items-center gap-2">
          <span className="text-[10px] font-mono font-semibold text-cyan-300 bg-cyan-500/15 px-2.5 py-1 rounded-md border border-cyan-500/30 flex items-center gap-1.5 shadow-[0_0_12px_rgba(6,182,212,0.3)]">
            <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse" />
            LIVE FEED
          </span>
        </div>
      </div>

      {/* Interactive Orbital Visualization Viewport */}
      <div className="relative flex-1 rounded-xl bg-[#08121e] border border-cyan-500/20 flex items-center justify-center overflow-hidden min-h-[250px] my-1 group">
        {/* Corner Reticles */}
        <div className="absolute top-2 left-2 w-3 h-3 border-t-2 border-l-2 border-cyan-500/40" />
        <div className="absolute top-2 right-2 w-3 h-3 border-t-2 border-r-2 border-cyan-500/40" />
        <div className="absolute bottom-2 left-2 w-3 h-3 border-b-2 border-l-2 border-cyan-500/40" />
        <div className="absolute bottom-2 right-2 w-3 h-3 border-b-2 border-r-2 border-cyan-500/40" />

        {/* Subtle grid readout overlay top left */}
        <div className="absolute top-3 left-4 text-[10px] font-mono text-cyan-400/70 uppercase tracking-widest flex items-center gap-1.5 z-20">
          <Activity size={12} className="text-cyan-400 animate-pulse" />
          <span>Orbital Vector Analysis</span>
        </div>

        {/* 3D Globe with radar sweep & tactical crosshair */}
        <div className="w-44 h-44 sm:w-56 sm:h-56 rounded-full bg-gradient-to-br from-slate-800 via-slate-900 to-[#0a1420] border border-cyan-500/30 relative shadow-[0_0_60px_rgba(6,182,212,0.25)] flex items-center justify-center overflow-hidden">
          {/* Outer rotating dashed telemetry ring */}
          <div
            className="absolute inset-1.5 rounded-full border border-cyan-400/30 border-dashed animate-spin"
            style={{ animationDuration: "35s" }}
          />

          {/* Inner counter-rotating ring */}
          <div
            className="absolute inset-5 rounded-full border border-sky-400/20 border-dashed animate-spin"
            style={{ animationDuration: "25s", animationDirection: "reverse" }}
          />

          {/* Radar Sweep Effect */}
          <div
            className="absolute inset-0 rounded-full bg-[conic-gradient(from_0deg,_rgba(6,182,212,0.35)_0deg,_transparent_65deg,_transparent_360deg)] animate-spin pointer-events-none"
            style={{ animationDuration: "8s", animationTimingFunction: "linear" }}
          />

          {/* Tactical Latitude/Longitude Radial Grid */}
          <div className="absolute inset-0 opacity-25 bg-[radial-gradient(circle_at_center,_#38bdf8_1px,_transparent_1px)] bg-[size:16px_16px]" />

          {/* Primary Target Hotspot */}
          <span className="absolute top-14 left-16 w-3.5 h-3.5 rounded-full bg-amber-400/30 animate-ping" />
          <span className="absolute top-14 left-16 w-2.5 h-2.5 rounded-full bg-amber-400 shadow-[0_0_15px_rgba(252,211,77,0.9)] border border-amber-200" />

          {/* Secondary Telemetry Data Point */}
          <span className="absolute bottom-16 right-16 w-2 h-2 rounded-full bg-cyan-400 animate-pulse shadow-[0_0_10px_rgba(6,182,212,0.8)]" />
        </div>

        {/* Bottom Telemetry Overlay */}
        <div className="absolute bottom-3 left-4 text-[10px] font-mono text-cyan-400/80 bg-slate-900/80 px-2 py-1 rounded border border-cyan-500/20 z-20">
          ALT: 500 km
        </div>
        <div className="absolute bottom-3 right-4 text-[10px] font-mono text-slate-300 bg-slate-900/80 px-2 py-1 rounded border border-slate-800 z-20">
          LAT 34.0522° N, LNG 118.2437° W
        </div>
      </div>
    </div>
  );
}

function SystemStatusPanel() {
  return (
    <div className="w-full border border-slate-800/90 bg-[#0c1624]/60 backdrop-blur-md rounded-xl p-4 sm:p-5 flex flex-col justify-between shadow-[0_4px_24px_rgba(0,0,0,0.3)]">
      <div>
        <div className="mb-3">
          <h3 className="text-sm font-semibold text-white tracking-tight">
            System Status
          </h3>
          <p className="text-[11px] text-slate-400 font-mono mt-0.5">
            Compute cluster &amp; storage array load
          </p>
        </div>

        <div className="space-y-1">
          <StatusRow label="Cluster Alpha (GPU)" value={92} icon={Cpu} />
          <StatusRow label="Node Beta (Ingest)" value={45} icon={Activity} />
          <StatusRow label="Storage Array" value={88} icon={HardDrive} />
        </div>
      </div>

      <div className="mt-5 pt-4 border-t border-slate-800/80">
        <div className="flex items-center justify-between mb-2">
          <h4 className="text-xs font-semibold text-slate-200">
            Available Sensors
          </h4>
          <span className="text-[10px] font-mono text-emerald-400 flex items-center gap-1">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
            3 Active
          </span>
        </div>

        <div className="space-y-1">
          <SensorRow
            name="Sentinel-2 Multispectral"
            meta="10m Res | GSD: 10m"
          />
          <SensorRow name="Landsat-8 OLI" meta="30m Res | Multispectral" />
          <SensorRow name="WorldView-3" meta="0.3m Res | High-Res" />
        </div>
      </div>
    </div>
  );
}

export default function DashboardPage() {
  return (
    <div className="flex flex-col lg:flex-row min-h-screen bg-[#0a1420] text-white font-sans overflow-x-hidden">
      <Sidebar />

      <div className="flex-1 flex flex-col min-w-0">
        <TopBar />

        <main className="flex-1 p-4 sm:p-6 lg:p-8 overflow-y-auto">
          {/* Integrated KPI Telemetry Strip */}
          <KpiTelemetryStrip />

          {/* Primary Visualization & System Status Row */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
            <div className="lg:col-span-2">
              <GlobeCoveragePanel />
            </div>
            <div className="lg:col-span-1">
              <SystemStatusPanel />
            </div>
          </div>

          {/* Recent Analyses & NLP Queries Row */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2">
              <RecentAnalyses />
            </div>
            <div className="lg:col-span-1">
              <RecentQueries />
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}