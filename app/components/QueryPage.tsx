"use client";

import { useState, useRef } from "react";
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
  Cpu,
  AlertTriangle,
  FileCode,
  CheckCircle2
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
            AI Controller Active
          </span>
        </div>
        <p className="text-[11px] text-slate-400 font-mono mt-0.5">
          Natural language geospatial analysis · SatQuery AI Controller
        </p>
      </div>

      <div className="flex items-center gap-2 self-end sm:self-auto">
        <div className="flex items-center gap-1.5 text-[11px] font-mono text-emerald-400 bg-emerald-500/10 border border-emerald-500/20 px-2.5 py-1 rounded-lg">
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
          <span>Model: PaliGemma-3B + BLIP</span>
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
  { label: "Is there a river or water body present?", icon: Waves },
  { label: "How many residential buildings are in this area?", icon: Building2 },
  { label: "Describe the land-cover and major objects visible", icon: Ship },
  { label: "What changed between these two dates?", icon: Flame },
  { label: "Are there industrial structures and roads present?", icon: TreePine },
  { label: "Describe this scene using both Optical and SAR sensors", icon: Leaf },
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
          placeholder="Ask a natural language geospatial query — e.g. How many residential buildings are in this area?..."
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
              <span>Routing...</span>
            </>
          ) : (
            <>
              <Play size={12} />
              <span>Execute</span>
            </>
          )}
        </button>
      </div>

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
          </select>
        </div>

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

      <div className="relative flex-1 rounded-xl bg-[#08121e] border border-cyan-500/15 flex items-center justify-center overflow-hidden min-h-[220px]">
        <div className="absolute top-3 left-3 text-[10px] font-mono text-cyan-400/70 z-10">
          BBOX: [−62.5, −4.2, −58.1, −1.0]
        </div>
        <div className="absolute top-3 right-3 text-[10px] font-mono text-slate-400 z-10">
          ZOOM: 16.4× · GSD: 0.5M
        </div>

        <div className="absolute inset-0 overflow-hidden pointer-events-none z-10">
          <div
            className="w-full h-[2px] bg-gradient-to-r from-transparent via-cyan-400/60 to-transparent animate-scan"
            style={{ animationDuration: "4s" }}
          />
        </div>

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

        <div className="relative flex items-center justify-center z-20">
          <span
            className="absolute w-20 h-20 rounded-full border border-cyan-400/15 animate-ping"
            style={{ animationDuration: "3s" }}
          />
          <span
            className="absolute w-12 h-12 rounded-full border border-cyan-400/25 animate-ping"
            style={{ animationDuration: "2s", animationDelay: "0.5s" }}
          />

          <div className="relative w-10 h-10 flex items-center justify-center">
            <Crosshair size={36} className="text-cyan-400/80" />
            <span className="absolute w-2 h-2 rounded-full bg-amber-400 shadow-[0_0_10px_rgba(252,211,77,0.9)]" />
          </div>
        </div>

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
// ML Controller Output Panel
// ----------------------------------------------------------------
function ControllerResultPanel({ result }: { result: any }) {
  if (!result) return null;

  const decision = result.routing_decision || {};
  const vqaResults = result.vqa_results || [];
  const requiresWarning = decision.requires_count_warning;

  return (
    <div className="w-full border border-cyan-500/30 bg-[#091524] rounded-xl p-5 shadow-[0_0_24px_rgba(6,182,212,0.12)] space-y-4">
      <div className="flex items-center justify-between border-b border-slate-800 pb-3">
        <div className="flex items-center gap-2">
          <Cpu className="text-cyan-400" size={18} />
          <h3 className="text-sm font-semibold text-white font-mono">
            Agentic Controller Execution Output
          </h3>
        </div>
        <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
          Status: 200 OK
        </span>
      </div>

      {/* Target Tools Badges */}
      <div>
        <div className="text-[10px] font-mono uppercase tracking-wider text-slate-400 mb-1.5 flex items-center gap-1.5">
          <FileCode size={12} className="text-cyan-400" />
          <span>Target Tools Selected</span>
        </div>
        <div className="flex flex-wrap gap-2">
          {decision.target_tools?.map((tool: string) => (
            <span
              key={tool}
              className="text-xs font-mono font-semibold px-2.5 py-1 rounded bg-cyan-500/20 text-cyan-300 border border-cyan-500/40"
            >
              {tool}
            </span>
          ))}
        </div>
      </div>

      {/* Execution Reasoning */}
      <div className="bg-slate-900/90 border border-slate-800 rounded-lg p-3">
        <div className="text-[10px] font-mono text-cyan-400 uppercase tracking-wider mb-1">
          Execution Reasoning
        </div>
        <p className="text-xs text-slate-300 leading-relaxed font-sans">
          {decision.execution_reasoning}
        </p>
      </div>

      {/* Counting Warning Banner */}
      {requiresWarning && (
        <div className="flex items-start gap-2.5 bg-amber-500/15 border border-amber-500/30 p-3 rounded-lg text-amber-300 text-xs font-mono">
          <AlertTriangle size={16} className="shrink-0 mt-0.5 text-amber-400" />
          <div>
            <div className="font-bold">Low Confidence Counting Warning</div>
            <div>Exact numeric counts are derived with low model confidence (~0.25-0.40). Treat this count as an estimate.</div>
          </div>
        </div>
      )}

      {/* Restructured Queries / VQA findings */}
      {vqaResults.length > 0 && (
        <div className="space-y-2">
          <div className="text-[10px] font-mono text-cyan-400 uppercase tracking-wider">
            Structured RSVQA Sub-Queries & Findings
          </div>
          <div className="space-y-1.5">
            {vqaResults.map((r: any, idx: number) => (
              <div key={idx} className="flex items-center justify-between bg-slate-900/60 border border-slate-800/80 px-3 py-2 rounded text-xs">
                <span className="font-mono text-slate-300">{r.question}</span>
                <div className="flex items-center gap-2">
                  <span className="font-bold text-cyan-300">{r.answer}</span>
                  <span className={`text-[10px] font-mono px-1.5 py-0.5 rounded ${r.low_confidence ? "bg-amber-500/20 text-amber-300 border border-amber-500/30" : "bg-emerald-500/20 text-emerald-300"}`}>
                    conf: {r.confidence}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Synthesized Response */}
      <div className="bg-cyan-950/30 border border-cyan-500/20 p-3.5 rounded-lg space-y-1">
        <div className="text-[10px] font-mono text-cyan-300 uppercase tracking-wider font-semibold">
          Synthesized User Response
        </div>
        <div className="text-xs text-slate-200 whitespace-pre-wrap leading-relaxed">
          {result.response_text}
        </div>
      </div>
    </div>
  );
}

// ----------------------------------------------------------------
// Query Page Main Component
// ----------------------------------------------------------------
export default function QueryPage() {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [loadingPhase, setLoadingPhase] = useState("ROUTING QUERY VIA AGENT...");
  const [apiResult, setApiResult] = useState<any>(null);

  const phases = [
    "CLASSIFYING INTENT...",
    "RESTRUCTURING QUERY FOR RSVQA...",
    "SELECTING TARGET TOOLS...",
    "EXECUTING VQA & CAPTIONING PIPELINE...",
    "SYNTHESIZING AGENT RESPONSE...",
  ];

  const handleExecute = async () => {
    if (!query.trim() || loading) return;
    setLoading(true);
    setApiResult(null);

    let phaseIndex = 0;
    setLoadingPhase(phases[0]);
    const interval = setInterval(() => {
      phaseIndex += 1;
      if (phaseIndex < phases.length) {
        setLoadingPhase(phases[phaseIndex]);
      }
    }, 400);

    try {
      const res = await fetch("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: query,
          optical_image: "dummy_data",
        })
      });
      const data = await res.json();
      setApiResult(data);
    } catch (err) {
      console.error(err);
    } finally {
      clearInterval(interval);
      setLoading(false);
    }
  };

  const handleSuggestion = (label: string) => {
    setQuery(label);
  };

  return (
    <div className="flex flex-col lg:flex-row min-h-screen bg-[#0a1420] text-white font-sans overflow-x-hidden">
      <Sidebar />

      <div className="flex-1 flex flex-col min-w-0">
        <PageHeader />

        <main className="flex-1 p-4 sm:p-6 lg:p-8 overflow-y-auto space-y-6">
          {/* Query input + chips */}
          <div>
            <QueryBar
              value={query}
              onChange={setQuery}
              onExecute={handleExecute}
              loading={loading}
              loadingPhase={loadingPhase}
            />
            <SuggestionChips onSelect={handleSuggestion} />
          </div>

          {/* Controller execution output panel */}
          {apiResult && <ControllerResultPanel result={apiResult} />}

          {/* Main panels grid */}
          <div className="grid grid-cols-1 md:grid-cols-3 xl:grid-cols-4 gap-5">
            <div className="md:col-span-1 xl:col-span-1">
              <AdvancedFilters />
            </div>

            <div className="md:col-span-2 xl:col-span-3">
              <LivePreview />
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}