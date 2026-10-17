"use client";

import { useState, useEffect, useMemo } from "react";
import Link from "next/link";
import {
  CheckCircle2,
  CircleDot,
  Boxes,
  Terminal,
  Search,
  Sparkles,
  Layers,
  Cpu,
  AlertTriangle,
  XCircle,
  HelpCircle,
  SkipForward,
  Clock,
  ShieldAlert,
  Activity,
  Filter,
} from "lucide-react";

import Sidebar from "./Sidebar";
import type {
  CanonicalSourceImage,
  CanonicalInvestigationState,
  ToolExecutionRecord,
  StructuredLogEvent,
  TraceStage,
} from "../types/nexspace";
import {
  getCurrentInvestigation,
  getActiveSourceImage,
  DEFAULT_DEMO_SOURCE,
} from "../utils/investigationStorage";

// ---------------------------------------------
// Page header
// ---------------------------------------------
function Header({
  investigation,
  sourceImage,
}: {
  investigation: CanonicalInvestigationState | null;
  sourceImage: CanonicalSourceImage;
}) {
  const isUpload = sourceImage.source === "upload";
  const outcome = investigation?.response?.structured_trace?.overall_outcome || (investigation ? "SUCCEEDED" : "STANDBY");
  const isFailed = outcome === "FAILED" || investigation?.response?.status === "failed";
  const isPartial = outcome === "PARTIAL_SUCCESS" || outcome === "DEGRADED";

  return (
    <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 px-4 sm:px-6 lg:px-8 py-4 border-b border-slate-800/80 bg-[#0d1826]/40 backdrop-blur-md">
      <div>
        <div className="flex items-center gap-1.5 text-[10px] font-medium mb-1 font-mono">
          <span
            className={`w-1.5 h-1.5 rounded-full ${
              isFailed
                ? "bg-rose-400"
                : isPartial
                ? "bg-amber-400"
                : investigation
                ? "bg-emerald-400 animate-pulse"
                : "bg-slate-500"
            }`}
          />
          <span
            className={
              isFailed
                ? "text-rose-400"
                : isPartial
                ? "text-amber-400"
                : investigation
                ? "text-emerald-400"
                : "text-slate-400"
            }
          >
            {investigation
              ? `TRACE: ${outcome.toUpperCase()}`
              : "STANDBY / AWAITING QUERY"}
          </span>
          {investigation?.response?.structured_trace?.total_duration_ms ? (
            <span className="text-cyan-300 ml-2">
              ({investigation.response.structured_trace.total_duration_ms.toFixed(0)} ms total)
            </span>
          ) : null}
        </div>
        <p className="text-base font-semibold text-white tracking-tight">
          {investigation ? `Query: "${investigation.query}"` : "Live Agentic Execution Trace"}
        </p>
        <p className="text-[11px] text-slate-400 mt-0.5 font-mono">
          Source: <span className="text-cyan-300">{sourceImage.filename}</span> ({isUpload ? "User Upload" : "Catalog Demo"}) · ID:{" "}
          <span className="text-slate-300">{investigation?.investigation_id || "INV-STANDBY"}</span>
        </p>
      </div>
      <div className="flex items-center gap-2">
        <Link
          href="/query"
          className="text-xs text-cyan-300 border border-cyan-400/30 bg-cyan-500/10 hover:bg-cyan-500/20 transition-all rounded-lg px-3 py-1.5 font-mono active:scale-95"
        >
          New Query
        </Link>
        <Link
          href="/execution-log"
          className="text-xs text-slate-300 border border-slate-700 bg-slate-800/60 hover:bg-slate-700/60 transition-all rounded-lg px-3 py-1.5 font-mono active:scale-95"
        >
          Structured Logs
        </Link>
      </div>
    </div>
  );
}

// ---------------------------------------------
// Pipeline topology
// ---------------------------------------------
type StageState = "succeeded" | "failed" | "skipped" | "unavailable" | "running" | "pending";

function Stage({
  label,
  state,
  sublabel,
  duration,
}: {
  label: string;
  state: StageState;
  sublabel?: string;
  duration?: string;
}) {
  let ring = "bg-slate-900/80 text-slate-500 border border-slate-800";
  let icon = <CircleDot size={15} />;

  if (state === "succeeded") {
    ring = "bg-emerald-500 text-[#0a1420] shadow-[0_0_12px_rgba(16,185,129,0.5)]";
    icon = <CheckCircle2 size={16} />;
  } else if (state === "running") {
    ring = "bg-cyan-500 text-[#0a1420] ring-4 ring-cyan-500/25 shadow-[0_0_20px_rgba(6,182,212,0.6)] animate-pulse";
    icon = <Cpu size={15} />;
  } else if (state === "failed") {
    ring = "bg-rose-500 text-white shadow-[0_0_12px_rgba(244,63,94,0.5)]";
    icon = <XCircle size={16} />;
  } else if (state === "unavailable") {
    ring = "bg-amber-500 text-[#0a1420] shadow-[0_0_10px_rgba(245,158,11,0.5)]";
    icon = <AlertTriangle size={15} />;
  } else if (state === "skipped") {
    ring = "bg-slate-800 text-slate-400 border border-slate-700";
    icon = <SkipForward size={14} />;
  }

  return (
    <div className="flex flex-col items-center gap-2 min-w-[110px] flex-1">
      <div
        className={`w-9 h-9 rounded-full flex items-center justify-center ${ring} shrink-0 transition-all duration-300`}
      >
        {icon}
      </div>
      <div className="text-center">
        <p
          className={`text-[11px] whitespace-nowrap font-medium ${
            state === "pending"
              ? "text-slate-500 font-mono"
              : state === "running"
              ? "text-cyan-300 font-semibold"
              : state === "failed"
              ? "text-rose-300 font-semibold"
              : state === "unavailable"
              ? "text-amber-300"
              : "text-slate-200"
          }`}
        >
          {label}
        </p>
        {duration && (
          <p className="text-[9px] text-cyan-400 font-mono">{duration}</p>
        )}
        {sublabel && (
          <p className="text-[10px] text-slate-400 font-mono truncate max-w-[120px]">{sublabel}</p>
        )}
      </div>
    </div>
  );
}

function PipelineTopology({ investigation }: { investigation: CanonicalInvestigationState | null }) {
  const isComplete = Boolean(investigation);
  const structuredStages = investigation?.response?.structured_trace?.stages;
  const legacyTrace = investigation?.response?.execution_trace;

  const stagesList = useMemo(() => {
    if (Array.isArray(structuredStages) && structuredStages.length > 0) {
      return structuredStages.map((st: TraceStage, idx: number) => {
        const rawStatus = (st.status || "pending").toLowerCase();
        let state: StageState = "pending";
        if (rawStatus === "succeeded" || rawStatus === "completed" || rawStatus === "done") {
          state = "succeeded";
        } else if (rawStatus === "failed" || rawStatus === "error") {
          state = "failed";
        } else if (rawStatus === "unavailable" || rawStatus === "partial_success") {
          state = "unavailable";
        } else if (rawStatus === "skipped") {
          state = "skipped";
        } else if (rawStatus === "running" || rawStatus === "started") {
          state = "running";
        }

        const durationStr = typeof st.duration_ms === "number" && st.duration_ms > 0
          ? `${st.duration_ms.toFixed(1)}ms`
          : undefined;

        const name = st.stage_name || (st as any).stage || `Stage ${idx + 1}`;
        const detailMsg = typeof (st as any).details === "string"
          ? (st as any).details
          : (st.metadata?.reason as string) || (st.error ? String(st.error) : undefined);

        return {
          label: name,
          state,
          duration: durationStr,
          sublabel: detailMsg,
        };
      });
    }

    if (Array.isArray(legacyTrace) && legacyTrace.length > 0) {
      return legacyTrace.map((st: any, idx: number) => {
        const rawStatus = (st.status || "SUCCESS").toLowerCase();
        const state: StageState = rawStatus === "success" || rawStatus === "completed"
          ? "succeeded"
          : rawStatus === "error" || rawStatus === "failed"
          ? "failed"
          : "pending";
        const durationStr = typeof st.duration_ms === "number" ? `${st.duration_ms.toFixed(1)}ms` : undefined;
        return {
          label: st.stage || `Stage ${idx + 1}`,
          state,
          duration: durationStr,
          sublabel: st.details || undefined,
        };
      });
    }

    // Default canonical baseline stages when standby
    return [
      { label: "1. Ingestion", state: isComplete ? "succeeded" : "pending", duration: isComplete ? "OK" : undefined },
      { label: "2. Tool Routing", state: isComplete ? "succeeded" : "pending", duration: isComplete ? "OK" : undefined },
      { label: "3. Specialist Engine", state: isComplete ? "succeeded" : "pending", duration: isComplete ? "OK" : undefined },
      { label: "4. Spatial Norm", state: isComplete ? "succeeded" : "pending", duration: isComplete ? "OK" : undefined },
      { label: "5. NL Synthesis", state: isComplete ? "succeeded" : "pending", duration: isComplete ? "OK" : undefined },
    ] as Array<{ label: string; state: StageState; duration?: string; sublabel?: string }>;
  }, [structuredStages, legacyTrace, isComplete]);

  return (
    <div className="w-full border border-slate-800/90 bg-[#0c1624]/60 backdrop-blur-md rounded-xl p-5 shadow-[0_4px_24px_rgba(0,0,0,0.3)]">
      <div className="flex items-center justify-between mb-5">
        <p className="text-xs font-semibold text-slate-200 flex items-center gap-1.5 font-mono uppercase tracking-wider">
          <Boxes size={13} className="text-cyan-400" />
          Monotonic Pipeline Execution Stages
        </p>
        <span className="text-[10px] text-slate-400 font-mono">
          {stagesList.length} Stage{stagesList.length === 1 ? "" : "s"} Tracked
        </span>
      </div>

      <div className="overflow-x-auto w-full pb-2">
        <div className="flex items-center min-w-[550px]">
          {stagesList.map((st, i) => (
            <div key={i} className="contents">
              <Stage
                label={st.label}
                state={st.state}
                duration={st.duration}
                sublabel={st.sublabel}
              />
              {i < stagesList.length - 1 && <div className="h-px flex-1 bg-slate-800 -mx-2 mb-6" />}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------
// Specialist Tool Execution Timeline
// ---------------------------------------------
function ToolExecutionTimeline({ investigation }: { investigation: CanonicalInvestigationState | null }) {
  const tools: ToolExecutionRecord[] = useMemo(() => {
    const rawTools = investigation?.response?.structured_trace?.tool_executions;
    if (Array.isArray(rawTools) && rawTools.length > 0) {
      return rawTools;
    }
    // Fallback if structured tools not present but selected_tools are
    const selected = investigation?.response?.selected_tools || investigation?.response?.plan?.target_tools;
    if (Array.isArray(selected) && selected.length > 0) {
      return selected.map((toolName: string) => ({
        tool_name: toolName,
        status: (investigation?.response?.status === "completed" ? "succeeded" : "degraded") as any,
        started_at: "",
        duration_ms: 0,
      }));
    }
    return [];
  }, [investigation]);

  if (tools.length === 0) {
    return null;
  }

  const getStatusBadge = (status: string) => {
    switch (status) {
      case "succeeded":
        return (
          <span className="flex items-center gap-1 text-[10px] font-mono text-emerald-400 bg-emerald-500/10 border border-emerald-500/20 px-2 py-0.5 rounded-full">
            <CheckCircle2 size={11} /> SUCCEEDED
          </span>
        );
      case "failed":
        return (
          <span className="flex items-center gap-1 text-[10px] font-mono text-rose-400 bg-rose-500/10 border border-rose-500/20 px-2 py-0.5 rounded-full">
            <XCircle size={11} /> FAILED
          </span>
        );
      case "unavailable":
        return (
          <span className="flex items-center gap-1 text-[10px] font-mono text-amber-400 bg-amber-500/10 border border-amber-500/20 px-2 py-0.5 rounded-full">
            <AlertTriangle size={11} /> UNAVAILABLE
          </span>
        );
      case "skipped":
        return (
          <span className="flex items-center gap-1 text-[10px] font-mono text-slate-400 bg-slate-800 border border-slate-700 px-2 py-0.5 rounded-full">
            <SkipForward size={11} /> SKIPPED
          </span>
        );
      case "running":
        return (
          <span className="flex items-center gap-1 text-[10px] font-mono text-cyan-300 bg-cyan-500/10 border border-cyan-500/20 px-2 py-0.5 rounded-full animate-pulse">
            <Activity size={11} /> RUNNING
          </span>
        );
      default:
        return (
          <span className="text-[10px] font-mono text-slate-400 bg-slate-800 px-2 py-0.5 rounded-full">
            {status.toUpperCase()}
          </span>
        );
    }
  };

  return (
    <div className="w-full border border-slate-800/90 bg-[#0c1624]/60 backdrop-blur-md rounded-xl p-5 shadow-[0_4px_24px_rgba(0,0,0,0.3)]">
      <div className="flex items-center justify-between mb-4">
        <p className="text-xs font-semibold text-slate-200 flex items-center gap-1.5 font-mono uppercase tracking-wider">
          <Cpu size={13} className="text-cyan-400" />
          Discrete Specialist Tool Execution Timeline
        </p>
        <span className="text-[10px] text-cyan-400 font-mono">
          {tools.length} Specialist Tool{tools.length === 1 ? "" : "s"} Invoked
        </span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
        {tools.map((tool, idx) => (
          <div
            key={idx}
            className="bg-[#08121e]/80 border border-slate-800/80 rounded-lg p-3 space-y-2 flex flex-col justify-between"
          >
            <div>
              <div className="flex items-center justify-between gap-2 mb-1">
                <span className="text-xs font-mono font-semibold text-white truncate">
                  {tool.tool_name}
                </span>
                {getStatusBadge(tool.status)}
              </div>
              <div className="flex items-center gap-2 text-[10px] font-mono text-slate-400">
                <span className="flex items-center gap-1">
                  <Clock size={10} className="text-cyan-400" />
                  {tool.duration_ms ? `${tool.duration_ms.toFixed(1)}ms` : "N/A"}
                </span>
                {tool.error_category && (
                  <span className="text-rose-400 font-mono">
                    [{tool.error_category}]
                  </span>
                )}
              </div>
            </div>

            {tool.error && (
              <div className="text-[10px] font-mono bg-rose-950/40 border border-rose-900/50 text-rose-300 p-2 rounded">
                <strong>Error:</strong> {tool.error}
                {tool.actionable_reason && (
                  <div className="text-slate-300 mt-0.5">
                    <strong>Action:</strong> {tool.actionable_reason}
                  </div>
                )}
              </div>
            )}

            {tool.output_summary && Object.keys(tool.output_summary).length > 0 && (
              <div className="text-[10px] font-mono bg-slate-900/60 p-2 rounded border border-slate-800/60 text-slate-300 space-y-0.5">
                {Object.entries(tool.output_summary).slice(0, 3).map(([k, v]) => (
                  <div key={k} className="flex justify-between gap-2">
                    <span className="text-slate-500 truncate">{k}:</span>
                    <span className="text-cyan-300 font-medium truncate">{String(v)}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

// ---------------------------------------------
// Live kernel output (terminal log & structured events)
// ---------------------------------------------
interface LogLineItem {
  time?: string;
  tag?: string | null;
  text: string;
  color?: string;
  correlationId?: string;
}

function LiveKernelOutput({
  investigation,
  sourceImage,
}: {
  investigation: CanonicalInvestigationState | null;
  sourceImage: CanonicalSourceImage;
}) {
  const [filterLevel, setFilterLevel] = useState<string>("ALL");

  const logs = useMemo<LogLineItem[]>(() => {
    if (!investigation) {
      return [
        { time: "00:00.00", tag: "INFO", text: "Pipeline initialized and listening for investigation requests.", color: "text-slate-400" },
        { time: "00:00.01", tag: "STANDBY", text: `Active raster buffer: ${sourceImage.filename}`, color: "text-slate-500" },
      ];
    }

    // 1. Structured log stream from backend
    const structuredLogs: StructuredLogEvent[] =
      investigation.response?.structured_trace?.logs ||
      investigation.response?.execution_logs ||
      [];

    if (Array.isArray(structuredLogs) && structuredLogs.length > 0) {
      return structuredLogs
        .filter((l) => {
          if (filterLevel === "ALL") return true;
          return l.level === filterLevel;
        })
        .map((l) => {
          const timeStr = l.timestamp ? l.timestamp.split("T")[1]?.slice(0, 12) || l.timestamp : "00:00.00";
          let color = "text-slate-300";
          if (l.level === "ERROR" || l.level === "CRITICAL") color = "text-rose-400 font-semibold";
          else if (l.level === "WARNING") color = "text-amber-300";
          else if (l.level === "INFO") color = "text-emerald-400";
          else if (l.level === "DEBUG") color = "text-slate-400";

          const detailStr = l.details && Object.keys(l.details).length > 0
            ? ` | ${JSON.stringify(l.details)}`
            : "";
          const reasonStr = l.actionable_reason ? ` [Action: ${l.actionable_reason}]` : "";

          return {
            time: timeStr,
            tag: l.level,
            text: `[${l.logger}] ${l.event}${detailStr}${reasonStr}`,
            color,
            correlationId: l.correlation_id,
          };
        });
    }

    // 2. Monotonic stages from execution_trace
    const trace = investigation.response?.execution_trace;
    if (Array.isArray(trace) && trace.length > 0) {
      return trace.map((st: any, idx: number) => {
        const stageNum = (idx + 1).toString().padStart(2, "0");
        const stageName = st.stage || st.name || `Stage ${idx + 1}`;
        const status = st.status || "SUCCESS";
        const duration = typeof st.duration_ms === "number" ? `${st.duration_ms.toFixed(1)}ms` : "";
        const details = st.details || st.message || "";
        const color = status === "SUCCESS" ? "text-emerald-400" : status === "ERROR" ? "text-rose-400" : "text-cyan-300";

        return {
          time: `00:0${stageNum}.00`,
          tag: status,
          text: `[${stageName}] ${details} ${duration ? `(${duration})` : ""}`.trim(),
          color,
        };
      });
    }

    const statusStr = investigation.response?.status || "STANDBY";
    const isFailed = statusStr === "failed" || statusStr === "error";
    return [
      { time: "00:00.00", tag: "INFO", text: `Investigation ID: ${investigation.investigation_id} | Query: "${investigation.query}"`, color: "text-slate-400" },
      { time: "00:00.01", tag: isFailed ? "ERROR" : "INFO", text: `Status: ${statusStr.toUpperCase()} | Monotonic stages complete.`, color: isFailed ? "text-rose-400" : "text-amber-400" },
    ];
  }, [investigation, sourceImage, filterLevel]);

  return (
    <div className="w-full border border-slate-800/90 bg-[#08121e] rounded-xl p-4 font-mono text-[11px] overflow-auto min-h-[320px] shadow-[0_4px_24px_rgba(0,0,0,0.3)]">
      <div className="flex flex-wrap items-center justify-between gap-2 mb-3 pb-2 border-b border-slate-800/80">
        <div className="flex items-center gap-2 text-xs font-semibold text-slate-300">
          <Terminal size={13} className="text-cyan-400" />
          <span>Live Structured Kernel Event Stream</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-[10px] text-slate-400">Filter:</span>
          {(["ALL", "INFO", "WARNING", "ERROR"] as const).map((lvl) => (
            <button
              key={lvl}
              onClick={() => setFilterLevel(lvl)}
              className={`text-[9px] px-2 py-0.5 rounded font-mono transition-all ${
                filterLevel === lvl
                  ? "bg-cyan-500/20 text-cyan-300 border border-cyan-400/40"
                  : "bg-slate-900 text-slate-400 hover:text-slate-200"
              }`}
            >
              {lvl}
            </button>
          ))}
          <span className="text-[10px] text-cyan-400 bg-cyan-500/10 px-2 py-0.5 rounded border border-cyan-500/20">
            PID: AGENT-CONTROLLER
          </span>
        </div>
      </div>
      <div className="space-y-1.5 max-h-[360px] overflow-y-auto">
        {logs.map((line, i) => {
          const timeStr = line?.time || "00:00.00";
          const tagStr = line?.tag;
          const textStr = line?.text || "";
          const colorStr = line?.color || "text-slate-300";

          return (
            <div key={i} className="flex gap-2 items-start leading-relaxed">
              <span className="text-slate-600 shrink-0 select-none">{timeStr}</span>
              {tagStr && (
                <span className={`shrink-0 font-semibold ${colorStr}`}>
                  [{tagStr}]
                </span>
              )}
              {line.correlationId && (
                <span className="text-slate-500 text-[10px] shrink-0 hidden md:inline">
                  [{line.correlationId.slice(0, 8)}]
                </span>
              )}
              <span className={tagStr ? "text-slate-300" : colorStr}>
                {textStr}
                {i === logs.length - 1 && (
                  <span className="inline-block w-1.5 h-3 bg-cyan-400 ml-1 animate-pulse align-middle" />
                )}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ---------------------------------------------
// Total progress ring & Execution Diagnostics
// ---------------------------------------------
function TotalProgress({ investigation }: { investigation: CanonicalInvestigationState | null }) {
  const structured = investigation?.response?.structured_trace;
  const outcome = structured?.overall_outcome || (investigation ? "SUCCEEDED" : "STANDBY");
  const percent = outcome === "SUCCEEDED" ? 100 : outcome === "PARTIAL_SUCCESS" ? 75 : outcome === "FAILED" ? 25 : investigation ? 100 : 0;
  const radius = 46;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (percent / 100) * circumference;

  const ringColor = percent === 100 ? "#10b981" : percent === 75 ? "#f59e0b" : percent === 25 ? "#f43f5e" : "#06b6d4";

  return (
    <div className="w-full border border-slate-800/90 bg-[#0c1624]/60 backdrop-blur-md rounded-xl p-4 flex flex-col justify-between shadow-[0_4px_24px_rgba(0,0,0,0.3)] min-h-[320px]">
      <p className="text-xs font-semibold text-slate-200 mb-2 font-mono uppercase tracking-wider">
        Execution Diagnostics
      </p>

      <div className="flex flex-col items-center">
        <svg width="120" height="120" viewBox="0 0 120 120" className="my-1">
          <circle
            cx="60"
            cy="60"
            r={radius}
            fill="none"
            stroke="rgba(255,255,255,0.06)"
            strokeWidth="8"
          />
          <circle
            cx="60"
            cy="60"
            r={radius}
            fill="none"
            stroke={ringColor}
            strokeWidth="8"
            strokeDasharray={circumference}
            strokeDashoffset={offset}
            strokeLinecap="round"
            transform="rotate(-90 60 60)"
          />
          <text
            x="60"
            y="56"
            textAnchor="middle"
            fill="white"
            fontSize="20"
            fontWeight="700"
            fontFamily="sans-serif"
          >
            {percent}%
          </text>
          <text
            x="60"
            y="74"
            textAnchor="middle"
            fill="#94a3b8"
            fontSize="8"
            fontFamily="monospace"
          >
            {outcome}
          </text>
        </svg>

        {structured?.error_category && (
          <div className="text-[10px] font-mono text-rose-300 bg-rose-950/40 border border-rose-900/50 px-2 py-0.5 rounded mt-1 text-center">
            Category: {structured.error_category}
          </div>
        )}
      </div>

      <div className="grid grid-cols-2 gap-2 w-full mt-3 text-center border-t border-slate-800/80 pt-3">
        <div className="bg-slate-900/60 p-2 rounded border border-slate-800/60">
          <p className="text-[9px] text-slate-400 uppercase font-mono tracking-widest">Detections</p>
          <p className="text-xs text-cyan-400 mt-0.5 font-mono font-semibold">
            {investigation?.response?.grounding?.detections?.length || 0} Targets
          </p>
        </div>
        <div className="bg-slate-900/60 p-2 rounded border border-slate-800/60">
          <p className="text-[9px] text-slate-400 uppercase font-mono tracking-widest">Duration</p>
          <p className="text-xs text-slate-200 mt-0.5 font-mono">
            {structured?.total_duration_ms ? `${structured.total_duration_ms.toFixed(0)}ms` : "Nominal"}
          </p>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------
// Status bar
// ---------------------------------------------
function StatusBar({ investigation }: { investigation: CanonicalInvestigationState | null }) {
  const reqId = investigation?.response?.request_id || investigation?.investigation_id || "NONE";
  return (
    <div className="flex flex-col sm:flex-row items-center justify-between gap-2 px-4 sm:px-8 py-3 text-[10px] font-mono text-emerald-400 border-t border-slate-800/80 bg-[#0d1826]/30">
      <div className="flex items-center gap-1.5">
        <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
        <span>
          Status: {investigation ? "Telemetry Synced · Structured Trace Active" : "Pipeline Idle · Awaiting Query"}
        </span>
        <span className="text-slate-500 ml-2">Trace Correlation: {reqId}</span>
      </div>
      <Link href="/execution-log" className="text-cyan-300 hover:text-cyan-200 underline">
        View Granular Execution Logs &amp; Audit Trail →
      </Link>
    </div>
  );
}

// ---------------------------------------------
// Execution Trace Page
// ---------------------------------------------
export default function ExecutionTracePage() {
  const [investigationState, setInvestigationState] = useState<CanonicalInvestigationState | null>(null);

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
        <Header investigation={investigationState} sourceImage={sourceImage} />

        <main className="flex-1 p-4 sm:p-6 lg:p-8 space-y-5 overflow-y-auto">
          <PipelineTopology investigation={investigationState} />

          <ToolExecutionTimeline investigation={investigationState} />

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
            <div className="lg:col-span-2">
              <LiveKernelOutput investigation={investigationState} sourceImage={sourceImage} />
            </div>
            <div className="lg:col-span-1">
              <TotalProgress investigation={investigationState} />
            </div>
          </div>

          <StatusBar investigation={investigationState} />
        </main>
      </div>
    </div>
  );
}
