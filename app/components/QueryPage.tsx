"use client";

import { useState, useRef, useEffect } from "react";
import {
  Search,
  Play,
  Waves,
  Building2,
  Leaf,
  Crosshair,
  Clock,
  RotateCcw,
  Loader2,
  ChevronRight,
  ScanLine,
  Activity,
  Radio,
  Ship,
  Flame,
  TreePine,
  CalendarDays,
  Gauge,
  Cloud,
  CheckCircle2,
  Sparkles,
} from "lucide-react";
import Sidebar from "./Sidebar";

// ----------------------------------------------------------------
// Inner top bar
// ----------------------------------------------------------------
function PageHeader() {
  return (
    <header className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 px-4 sm:px-6 lg:px-8 py-4 border-b border-slate-800/80 bg-[#09131f]/60 backdrop-blur-md">
      <div>
        <div className="flex items-center gap-2">
          <Search size={15} className="text-cyan-400" />
          <h1 className="text-sm font-semibold text-white tracking-tight">
            NLP Query Terminal
          </h1>
          <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
            AI Active
          </span>
        </div>
        <p className="text-[11px] text-slate-400 font-mono mt-0.5">
          Natural language geospatial analysis · SatQuery AI Engine
        </p>
      </div>

      <div className="flex items-center gap-2 self-end sm:self-auto">
        <div className="flex items-center gap-1.5 text-[11px] font-mono text-emerald-400 bg-emerald-500/10 border border-emerald-500/20 px-2.5 py-1 rounded-lg">
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
          <span>Model: Spatial-GPT4v</span>
        </div>
        <div className="flex items-center gap-1.5 text-[11px] font-mono text-cyan-400 bg-cyan-500/10 border border-cyan-500/20 px-2.5 py-1 rounded-lg">
          <Radio size={11} className="animate-pulse" />
          <span>LAT −3.47 · LON −68.21</span>
        </div>
      </div>
    </header>
  );
}

// ----------------------------------------------------------------
// Suggestion chips
// ----------------------------------------------------------------
const suggestions = [
  { label: "Port activity changes", icon: Ship },
  { label: "Detect new infrastructure", icon: Building2 },
  { label: "Vegetation health (NDVI)", icon: Leaf },
  { label: "Active wildfires", icon: Flame },
  { label: "Coastal erosion", icon: Waves },
  { label: "Deforestation patterns", icon: TreePine },
];

function SuggestionChips({
  onSelect,
}: {
  onSelect: (label: string) => void;
}) {
  return (
    <div className="flex flex-wrap gap-2 mt-3">
      {suggestions.map(({ label, icon: Icon }) => (
        <button
          key={label}
          onClick={() => onSelect(label)}
          className="group flex items-center gap-1.5 bg-[#09131f]/70 hover:bg-cyan-500/15 border border-slate-800/80 hover:border-cyan-500/40 rounded-lg px-3 py-1.5 text-xs text-slate-400 hover:text-cyan-300 transition-all duration-180 hover:scale-[1.02] hover:-translate-y-0.5 active:scale-[0.98] cursor-pointer shadow-sm"
        >
          <Icon
            size={12}
            className="text-slate-500 group-hover:text-cyan-400 transition-colors duration-180"
          />
          <span>{label}</span>
        </button>
      ))}
    </div>
  );
}

// ----------------------------------------------------------------
// AI Query input bar
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
  const inputRef = useRef<HTMLInputElement>(null);
  const [focused, setFocused] = useState(false);

  const containerClass = loading
    ? "border-cyan-400/60 shadow-[0_0_0_2px_rgba(6,182,212,0.2),0_0_24px_rgba(6,182,212,0.15)] bg-[#0a1624]"
    : focused
    ? "border-cyan-500/50 shadow-[0_0_0_2px_rgba(6,182,212,0.12),0_0_16px_rgba(6,182,212,0.08)] bg-[#0a1624]"
    : "border-slate-800/90 shadow-sm hover:border-slate-700/80 bg-[#09131f]/90";

  return (
    <div className="relative">
      <div
        className={`flex items-center gap-3 border rounded-xl px-4 py-3 sm:py-3.5 transition-all duration-180 ease-out ${containerClass}`}
      >
        {/* Terminal AI icon */}
        <div className="shrink-0 flex items-center gap-1.5">
          <div
            className={`p-1.5 rounded-lg border transition-colors duration-180 ${
              focused || loading
                ? "bg-cyan-500/15 border-cyan-500/30 text-cyan-400"
                : "bg-slate-800/60 border-slate-700/60 text-slate-500"
            }`}
          >
            <Activity size={14} className={loading ? "animate-pulse" : ""} />
          </div>
        </div>

        <input
          ref={inputRef}
          type="text"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onFocus={() => setFocused(true)}
          onBlur={() => setFocused(false)}
          onKeyDown={(e) => e.key === "Enter" && !loading && onExecute()}
          placeholder="Ask a natural language geospatial query — e.g. Detect vessel wake patterns in the Suez Canal over 48h..."
          className="flex-1 bg-transparent text-sm text-slate-200 placeholder:text-slate-500 focus:outline-none min-w-0 font-sans"
        />

        <button
          onClick={onExecute}
          disabled={loading}
          className="shrink-0 flex items-center gap-2 bg-gradient-to-r from-cyan-500 to-sky-400 hover:from-cyan-400 hover:to-sky-300 disabled:opacity-60 disabled:cursor-not-allowed text-[#071320] text-xs font-bold rounded-lg px-4 py-2 transition-all duration-180 shadow-[0_0_12px_rgba(6,182,212,0.35)] hover:shadow-[0_0_18px_rgba(6,182,212,0.55)] hover:scale-[1.02] active:scale-[0.98] cursor-pointer"
        >
          {loading ? (
            <>
              <Loader2 size={13} className="animate-spin text-[#071320]" />
              <span>Analyzing</span>
            </>
          ) : (
            <>
              <Play size={12} />
              <span>Execute</span>
            </>
          )}
        </button>
      </div>

      {/* Progressive loading state banner */}
      {loading && (
        <div className="mt-2 flex items-center justify-between text-[11px] font-mono text-cyan-300 bg-cyan-500/10 border border-cyan-500/25 px-3 py-1.5 rounded-lg shadow-sm">
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-cyan-400 animate-ping" />
            <span>{loadingPhase}</span>
          </div>
          <span className="text-[10px] text-cyan-400/80">LAT −3.46 · LON −68.21</span>
        </div>
      )}
    </div>
  );
}

// ----------------------------------------------------------------
// Advanced Filters
// ----------------------------------------------------------------
function AdvancedFilters() {
  const [cloudCover, setCloudCover] = useState(20);
  const [selectedSensor, setSelectedSensor] = useState("SAR");

  return (
    <div className="w-full border border-slate-800/90 bg-[#0c1624]/60 backdrop-blur-md rounded-xl p-4 sm:p-5 shadow-[0_4px_24px_rgba(0,0,0,0.3)]">
      <div className="mb-4">
        <h3 className="text-sm font-semibold text-white tracking-tight">
          Advanced Filters
        </h3>
        <p className="text-[11px] text-slate-400 font-mono mt-0.5">
          Constrain spatial-temporal search parameters
        </p>
      </div>

      <div className="space-y-4">
        {/* Date Range */}
        <div>
          <label className="flex items-center gap-1.5 text-[10px] text-slate-400 uppercase tracking-widest font-mono mb-1.5">
            <CalendarDays size={11} className="text-cyan-400" />
            Date Range
          </label>
          <select className="w-full bg-slate-900/80 border border-slate-800 hover:border-slate-700 focus:border-cyan-500/50 focus:shadow-[0_0_0_2px_rgba(6,182,212,0.1)] rounded-lg px-3 py-2 text-xs text-slate-300 transition-all duration-180 cursor-pointer focus:outline-none appearance-none">
            <option value="48h">Last 48 Hours</option>
            <option value="7d">Last 7 Days</option>
            <option value="30d">Last 30 Days</option>
            <option value="90d">Last 90 Days</option>
            <option value="custom">Custom Range</option>
          </select>
        </div>

        {/* Spatial Resolution */}
        <div>
          <label className="flex items-center gap-1.5 text-[10px] text-slate-400 uppercase tracking-widest font-mono mb-1.5">
            <Gauge size={11} className="text-cyan-400" />
            Spatial Resolution
          </label>
          <select className="w-full bg-slate-900/80 border border-slate-800 hover:border-slate-700 focus:border-cyan-500/50 focus:shadow-[0_0_0_2px_rgba(6,182,212,0.1)] rounded-lg px-3 py-2 text-xs text-slate-300 transition-all duration-180 cursor-pointer focus:outline-none appearance-none">
            <option value="high">High (&lt; 1m GSD)</option>
            <option value="medium">Medium (1–10m GSD)</option>
            <option value="low">Low (10–30m GSD)</option>
          </select>
        </div>

        {/* Cloud Cover Slider */}
        <div>
          <div className="flex items-center justify-between mb-1.5">
            <label className="flex items-center gap-1.5 text-[10px] text-slate-400 uppercase tracking-widest font-mono">
              <Cloud size={11} className="text-cyan-400" />
              Cloud Cover
            </label>
            <span className="text-[11px] font-mono font-semibold text-cyan-400">
              &lt;{cloudCover}%
            </span>
          </div>
          <div className="relative">
            <div className="h-1 w-full bg-slate-800 rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-cyan-500 to-sky-400 rounded-full transition-all duration-180"
                style={{ width: `${cloudCover}%` }}
              />
            </div>
            <input
              type="range"
              min={0}
              max={100}
              value={cloudCover}
              onChange={(e) => setCloudCover(Number(e.target.value))}
              className="absolute inset-0 w-full opacity-0 cursor-pointer h-4 -top-1.5"
            />
          </div>
        </div>

        {/* Sensor Source */}
        <div>
          <label className="flex items-center gap-1.5 text-[10px] text-slate-400 uppercase tracking-widest font-mono mb-1.5">
            <ScanLine size={11} className="text-cyan-400" />
            Sensor Modality
          </label>
          <div className="flex flex-wrap gap-1.5">
            {["SAR", "Optical", "Multispectral", "Thermal"].map((s) => (
              <button
                key={s}
                onClick={() => setSelectedSensor(s)}
                className={`text-[10px] font-mono px-2.5 py-1 rounded-md border transition-all duration-180 cursor-pointer ${
                  selectedSensor === s
                    ? "bg-cyan-500/20 text-cyan-300 border-cyan-500/40 font-semibold shadow-[0_0_10px_rgba(6,182,212,0.2)]"
                    : "bg-slate-900/80 border-slate-800 text-slate-400 hover:border-cyan-500/30 hover:text-cyan-300"
                }`}
              >
                {s}
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

// ----------------------------------------------------------------
// Live Preview (Mission Control Viewport)
// ----------------------------------------------------------------
function LivePreview() {
  return (
    <div className="w-full border border-cyan-500/25 bg-[#0c1624]/75 backdrop-blur-md rounded-xl p-4 sm:p-5 relative overflow-hidden min-h-[340px] flex flex-col shadow-[0_0_30px_rgba(6,182,212,0.06)]">
      {/* Subtle HUD grid background */}
      <div
        className="absolute inset-0 opacity-[0.08] pointer-events-none"
        style={{
          backgroundImage:
            "linear-gradient(rgba(56,189,248,0.3) 1px, transparent 1px), linear-gradient(90deg, rgba(56,189,248,0.3) 1px, transparent 1px)",
          backgroundSize: "28px 28px",
        }}
      />

      <div className="flex items-center justify-between mb-4 relative z-10">
        <div>
          <h3 className="text-sm font-semibold text-white tracking-tight">
            Live Preview
          </h3>
          <p className="text-[11px] text-slate-400 font-mono mt-0.5">
            Orbital sensor footprint · Real-time acquisition
          </p>
        </div>
        <span className="flex items-center gap-1.5 text-[10px] font-mono font-semibold text-cyan-300 bg-cyan-500/15 px-2.5 py-1 rounded-md border border-cyan-500/30">
          <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse" />
          Target: Sector 9B
        </span>
      </div>

      {/* Viewport */}
      <div className="relative flex-1 rounded-xl bg-[#08121e] border border-cyan-500/15 flex items-center justify-center overflow-hidden min-h-[220px]">
        {/* Corner labels */}
        <div className="absolute top-3 left-3 text-[10px] font-mono text-cyan-400/70 z-10">
          BBOX: [−62.5, −4.2, −58.1, −1.0]
        </div>
        <div className="absolute top-3 right-3 text-[10px] font-mono text-slate-400 z-10">
          ZOOM: 16.4× · GSD: 0.5M
        </div>

        {/* Scan-line sweep */}
        <div className="absolute inset-0 overflow-hidden pointer-events-none z-10">
          <div
            className="w-full h-[2px] bg-gradient-to-r from-transparent via-cyan-400/60 to-transparent animate-scan"
            style={{ animationDuration: "4s" }}
          />
        </div>

        {/* Selection bounding box */}
        <div className="absolute inset-8 border border-cyan-500/20 rounded-md pointer-events-none z-10">
          {[
            "top-0 left-0 border-t-2 border-l-2",
            "top-0 right-0 border-t-2 border-r-2",
            "bottom-0 left-0 border-b-2 border-l-2",
            "bottom-0 right-0 border-b-2 border-r-2",
          ].map((cls, i) => (
            <div
              key={i}
              className={`absolute w-4 h-4 border-cyan-400/70 ${cls}`}
            />
          ))}
        </div>

        {/* Central target */}
        <div className="relative flex items-center justify-center z-20">
          <span
            className="absolute w-20 h-20 rounded-full border border-cyan-400/15 animate-ping"
            style={{ animationDuration: "3s" }}
          />
          <span
            className="absolute w-12 h-12 rounded-full border border-cyan-400/25 animate-ping"
            style={{ animationDuration: "2s", animationDelay: "0.5s" }}
          />

          {/* Crosshair */}
          <div className="relative w-10 h-10 flex items-center justify-center">
            <Crosshair size={36} className="text-cyan-400/80" />
            <span className="absolute w-2 h-2 rounded-full bg-amber-400 shadow-[0_0_10px_rgba(252,211,77,0.9)]" />
          </div>
        </div>

        {/* Bottom telemetry */}
        <div className="absolute bottom-3 left-3 text-[10px] font-mono text-slate-300 bg-slate-900/80 px-2 py-0.5 rounded border border-slate-800 z-10">
          −1.9° S, −58.1° E
        </div>
        <div className="absolute bottom-3 right-3 text-[10px] font-mono text-cyan-400/90 bg-slate-900/80 px-2 py-0.5 rounded border border-cyan-500/20 z-10">
          ALT: 450 km · PASS 892
        </div>
      </div>
    </div>
  );
}

// ----------------------------------------------------------------
// Query History
// ----------------------------------------------------------------
const history = [
  {
    text: "Show me all active wildfires in the Amazon over the last 48 hours",
    time: "2 mins ago",
    icon: Flame,
    status: "Completed",
  },
  {
    text: "Detect new infrastructure in Eastern Europe",
    time: "1 hr ago",
    icon: Building2,
    status: "Completed",
  },
  {
    text: "Calculate NDVI for California Central Valley",
    time: "Yesterday",
    icon: Leaf,
    status: "Completed",
  },
];

function QueryHistory({
  onRunAgain,
}: {
  onRunAgain: (text: string) => void;
}) {
  return (
    <div className="w-full border border-slate-800/90 bg-[#0c1624]/60 backdrop-blur-md rounded-xl p-4 sm:p-5 shadow-[0_4px_24px_rgba(0,0,0,0.3)]">
      <div className="mb-4">
        <h3 className="text-sm font-semibold text-white tracking-tight">
          Query History
        </h3>
        <p className="text-[11px] text-slate-400 font-mono mt-0.5">
          {history.length} recent queries
        </p>
      </div>

      <div className="space-y-2">
        {history.map((q, i) => (
          <div
            key={i}
            className="group p-3 rounded-lg bg-slate-900/50 border border-slate-800/80 hover:border-cyan-500/30 hover:bg-cyan-500/[0.03] transition-all duration-180 cursor-default"
          >
            <div className="flex items-start gap-2.5">
              <div className="mt-0.5 p-1.5 rounded-md bg-slate-800/80 border border-slate-700/60 shrink-0">
                <q.icon size={12} className="text-cyan-400" />
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-xs text-slate-300 leading-relaxed group-hover:text-slate-100 transition-colors line-clamp-2">
                  {q.text}
                </p>
                <div className="flex items-center justify-between mt-2">
                  <div className="flex items-center gap-1.5 text-[10px] font-mono text-slate-400">
                    <Clock size={10} />
                    <span>{q.time}</span>
                  </div>
                  <button
                    onClick={() => onRunAgain(q.text)}
                    className="flex items-center gap-1 text-[10px] font-mono text-slate-400 hover:text-cyan-400 transition-colors opacity-0 group-hover:opacity-100 cursor-pointer"
                  >
                    <RotateCcw size={10} />
                    <span>Run again</span>
                  </button>
                </div>
              </div>
              <ChevronRight
                size={14}
                className="text-slate-600 group-hover:text-cyan-500 shrink-0 mt-0.5 transition-colors"
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ----------------------------------------------------------------
// Status bar
// ----------------------------------------------------------------
function StatusBar({ loading }: { loading: boolean }) {
  return (
    <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pt-3 mt-3 border-t border-slate-800/80 text-[10px] font-mono text-slate-400">
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-1.5 text-emerald-400">
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
          <span>AI Engine: Active</span>
        </div>
        {loading && (
          <div className="flex items-center gap-1.5 text-cyan-400">
            <Loader2 size={11} className="animate-spin" />
            <span>Processing query pipeline...</span>
          </div>
        )}
      </div>
      <div className="flex items-center gap-3 text-slate-400">
        <span>LAT: −3.4653</span>
        <span className="text-slate-700">·</span>
        <span>LON: −68.2144</span>
        <span className="text-slate-700">·</span>
        <span>ALT: 450 KM</span>
      </div>
    </div>
  );
}

// ----------------------------------------------------------------
// Query Page
// ----------------------------------------------------------------
export default function QueryPage() {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [loadingPhase, setLoadingPhase] = useState("INITIALIZING ORBITAL LINK...");

  const phases = [
    "INITIALIZING ORBITAL LINK...",
    "ACQUIRING SENSOR DATA...",
    "PROCESSING GEO-REFERENCE...",
    "RUNNING NEURAL ANALYSIS...",
    "GENERATING INTELLIGENCE...",
  ];

  const handleExecute = () => {
    if (!query.trim() || loading) return;
    setLoading(true);

    let phaseIndex = 0;
    setLoadingPhase(phases[0]);
    const interval = setInterval(() => {
      phaseIndex += 1;
      if (phaseIndex < phases.length) {
        setLoadingPhase(phases[phaseIndex]);
      } else {
        clearInterval(interval);
      }
    }, 550);

    setTimeout(() => {
      clearInterval(interval);
      setLoading(false);
    }, 2800);
  };

  const handleSuggestion = (label: string) => {
    setQuery(label);
  };

  const handleRunAgain = (text: string) => {
    setQuery(text);
    setTimeout(() => {
      handleExecute();
    }, 50);
  };

  return (
    <div className="flex flex-col lg:flex-row min-h-screen bg-[#0a1420] text-white font-sans overflow-x-hidden">
      <Sidebar />

      <div className="flex-1 flex flex-col min-w-0">
        <PageHeader />

        <main className="flex-1 p-4 sm:p-6 lg:p-8 overflow-y-auto">
          {/* Query input + chips */}
          <div className="mb-6">
            <QueryBar
              value={query}
              onChange={setQuery}
              onExecute={handleExecute}
              loading={loading}
              loadingPhase={loadingPhase}
            />
            <SuggestionChips onSelect={handleSuggestion} />
          </div>

          {/* Main panels grid */}
          <div className="grid grid-cols-1 md:grid-cols-3 xl:grid-cols-4 gap-5">
            {/* Filters — col 1 */}
            <div className="md:col-span-1 xl:col-span-1 order-2 md:order-1">
              <AdvancedFilters />
            </div>

            {/* Live Preview — col 2-3 (primary focus) */}
            <div className="md:col-span-2 xl:col-span-2 order-1 md:order-2">
              <LivePreview />
            </div>

            {/* Query History — col 4 */}
            <div className="md:col-span-3 xl:col-span-1 order-3">
              <QueryHistory onRunAgain={handleRunAgain} />
            </div>
          </div>

          {/* Status bar */}
          <StatusBar loading={loading} />
        </main>
      </div>
    </div>
  );
}