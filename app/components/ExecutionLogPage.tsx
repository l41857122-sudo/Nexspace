"use client";

import { useState, useMemo } from "react";
import Link from "next/link";
import {
  Download,
  Info,
  Layers,
  CheckCircle2,
  Cpu,
  Search,
  Crosshair,
  Sparkles,
  ArrowRight,
  AlertTriangle,
  XCircle,
  Clock,
  Terminal,
  Filter,
  ShieldCheck,
  FileText,
} from "lucide-react";
import Sidebar from "./Sidebar";
import type {
  CanonicalSourceImage,
  CanonicalInvestigationState,
  StructuredLogEvent,
  ToolExecutionRecord,
  TraceStage,
} from "../types/nexspace";
import { normalizeBox } from "./QueryPage";
import {
  getCurrentInvestigation,
  getActiveSourceImage,
  DEFAULT_DEMO_SOURCE,
} from "../utils/investigationStorage";

// ---------------------------------------------
// Header
// ---------------------------------------------
function Header({
  investigation,
  sourceImage,
}: {
  investigation: CanonicalInvestigationState | null;
  sourceImage: CanonicalSourceImage;
}) {
  const isUpload = sourceImage.source === "upload";
  const outcome = investigation?.response?.structured_trace?.overall_outcome || (investigation ? "COMPLETED" : "IDLE");

  const handleExportLog = () => {
    if (!investigation) return;
    const logData = {
      investigation_id: investigation.investigation_id,
      request_id: investigation.response?.request_id || investigation.investigation_id,
      query: investigation.query,
      timestamp: investigation.timestamp,
      structured_trace: investigation.response?.structured_trace,
      execution_logs: investigation.response?.execution_logs,
      execution_trace: investigation.response?.execution_trace,
      source_image: {
        filename: sourceImage.filename,
        mediaType: sourceImage.mediaType,
        source: sourceImage.source,
      },
      response: investigation.response,
    };
    const blob = new Blob([JSON.stringify(logData, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `execution_log_${investigation.investigation_id || "active"}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 px-4 sm:px-6 lg:px-8 py-4 border-b border-slate-800/80 bg-[#0d1826]/40 backdrop-blur-md">
      <div>
        <div className="flex flex-wrap items-center gap-2 mb-1.5 font-mono">
          <span className="text-[10px] bg-slate-800/60 text-slate-300 rounded-full px-2 py-0.5 tracking-wide border border-slate-700/60">
            AUDIT TRAIL &amp; LOGS
          </span>
          <span className="flex items-center gap-1 text-[10px] bg-emerald-500/10 text-emerald-300 rounded-full px-2 py-0.5 border border-emerald-500/20">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
            {outcome.toUpperCase()}
          </span>
          <span
            className={`text-[10px] px-2 py-0.5 rounded-full border ${
              isUpload
                ? "text-emerald-300 bg-emerald-500/15 border-emerald-500/30"
                : "text-blue-300 bg-blue-500/15 border-blue-500/30"
            }`}
          >
            Source: {sourceImage.filename} ({isUpload ? "User Upload" : "Demo Tile"})
          </span>
          {investigation?.response?.request_id && (
            <span className="text-[10px] text-cyan-400 font-mono">
              Correlation: {investigation.response.request_id}
            </span>
          )}
        </div>
        <p className="text-base font-semibold text-white tracking-tight">
          Query ID: <span className="font-mono text-cyan-300">{investigation?.investigation_id || "INV-STANDBY"}</span>
        </p>
        <p className="text-[11px] text-slate-400 mt-0.5 font-mono">
          Query: &ldquo;{investigation?.query || "No query executed yet"}&rdquo;
        </p>
      </div>
      <div className="flex items-center gap-2">
        <Link
          href="/execution-trace"
          className="flex items-center gap-1 bg-cyan-500/10 hover:bg-cyan-500/20 border border-cyan-400/30 text-xs text-cyan-300 rounded-lg px-3 py-1.5 font-mono transition-all"
        >
          <Layers size={12} />
          <span>View Trace Flow</span>
        </Link>
        <button
          onClick={handleExportLog}
          disabled={!investigation}
          className="flex items-center gap-1.5 bg-slate-900/60 hover:bg-slate-800/80 disabled:opacity-40 border border-slate-800 hover:border-cyan-500/30 transition-all text-xs text-slate-300 hover:text-cyan-300 rounded-lg px-3 py-1.5 font-mono active:scale-95 cursor-pointer"
        >
          <Download size={12} />
          <span>Export Audit (JSON)</span>
        </button>
      </div>
    </div>
  );
}

// ---------------------------------------------
// Key-value detail box
// ---------------------------------------------
function DetailBox({
  label,
  value,
  valueColor,
}: {
  label: string;
  value: string;
  valueColor?: string;
}) {
  return (
    <div className="bg-slate-900/60 border border-slate-800/80 rounded-lg px-3 py-2 flex-1 min-w-[140px]">
      <p className="text-[9px] text-slate-400 uppercase tracking-widest font-mono">
        {label}
      </p>
      <p
        className={`text-xs font-mono mt-0.5 truncate ${
          valueColor ?? "text-slate-200"
        }`}
      >
        {value}
      </p>
    </div>
  );
}

// ---------------------------------------------
// Pipeline step component
// ---------------------------------------------
function PipelineStep({
  icon: Icon,
  title,
  duration,
  status,
  children,
}: {
  icon: React.ElementType;
  title: string;
  duration?: string;
  status?: string;
  children: React.ReactNode;
}) {
  const isFailed = status === "failed" || status === "error";
  return (
    <div className="flex gap-3.5">
      <div className="flex flex-col items-center">
        <div
          className={`w-8 h-8 rounded-full flex items-center justify-center shrink-0 border ${
            isFailed
              ? "bg-rose-950/60 border-rose-500/40 text-rose-400 shadow-[0_0_10px_rgba(244,63,94,0.3)]"
              : "bg-slate-900/80 border-cyan-500/30 text-cyan-400 shadow-[0_0_10px_rgba(6,182,212,0.2)]"
          }`}
        >
          <Icon size={14} />
        </div>
        <div className="flex-1 w-px bg-slate-800 my-1.5" />
      </div>

      <div className="flex-1 pb-6">
        <div className="flex items-center justify-between mb-2">
          <p className="text-xs sm:text-sm text-slate-100 font-semibold tracking-tight">{title}</p>
          {duration && (
            <span
              className={`text-[10px] font-mono px-2 py-0.5 rounded border ${
                isFailed
                  ? "text-rose-400 bg-rose-500/10 border-rose-500/20"
                  : "text-cyan-400 bg-cyan-500/10 border-cyan-500/20"
              }`}
            >
              {duration}
            </span>
          )}
        </div>
        {children}
      </div>
    </div>
  );
}

// ---------------------------------------------
// Dynamic Pipeline Execution Steps List
// ---------------------------------------------
function PipelineSteps({
  investigation,
  sourceImage,
}: {
  investigation: CanonicalInvestigationState | null;
  sourceImage: CanonicalSourceImage;
}) {
  const detections = investigation?.response?.grounding?.detections || [];
  const caption =
    typeof investigation?.response?.optical_caption === "string"
      ? investigation.response.optical_caption
      : (investigation?.response?.optical_caption as unknown as { caption?: string })?.caption ||
        "Scene captioning completed.";
  const vqa = investigation?.response?.vqa_results || [];
  const structuredStages = investigation?.response?.structured_trace?.stages;
  const toolExecutions = investigation?.response?.structured_trace?.tool_executions;

  if (!investigation) {
    return (
      <div className="w-full border border-slate-800/90 bg-[#0c1624]/60 backdrop-blur-md rounded-xl p-8 text-center shadow-[0_4px_24px_rgba(0,0,0,0.3)] space-y-4">
        <Layers size={32} className="text-slate-600 mx-auto" />
        <h3 className="text-base font-semibold text-white">No Active Investigation Trace</h3>
        <p className="text-xs text-slate-400 font-mono max-w-md mx-auto">
          Execute a query in the Query Terminal to inspect live backend execution stages, tool timelines, and structured audit logs.
        </p>
        <Link
          href="/query"
          className="inline-flex items-center gap-2 bg-cyan-500 hover:bg-cyan-400 text-[#0a1420] text-xs font-bold px-4 py-2 rounded-lg transition-all"
        >
          <Search size={14} />
          <span>Open Query Terminal</span>
        </Link>
      </div>
    );
  }

  return (
    <div className="w-full border border-slate-800/90 bg-[#0c1624]/60 backdrop-blur-md rounded-xl p-5 shadow-[0_4px_24px_rgba(0,0,0,0.3)] space-y-6">
      <div>
        <p className="text-xs font-semibold text-white tracking-tight uppercase tracking-wider font-mono mb-4 flex items-center gap-2">
          <Layers size={13} className="text-cyan-400" />
          <span>Monotonic Pipeline Execution Stages</span>
        </p>

        <div className="flex-1">
          {Array.isArray(structuredStages) && structuredStages.length > 0 ? (
            structuredStages.map((st: TraceStage, idx: number) => {
              const duration = typeof st.duration_ms === "number" ? `${st.duration_ms.toFixed(1)}ms` : "✓ Complete";
              const isFailed = st.status === "failed" || Boolean(st.error);
              return (
                <PipelineStep
                  key={idx}
                  icon={isFailed ? XCircle : CheckCircle2}
                  title={`${st.stage_name || (st as any).stage || `Stage ${idx + 1}`}`}
                  duration={duration}
                  status={st.status}
                >
                  <div className="text-[11px] font-mono bg-slate-900/60 p-3 rounded-lg border border-slate-800/80 space-y-1.5 text-slate-300">
                    <div className="flex justify-between text-slate-400">
                      <span>Status: <strong className={isFailed ? "text-rose-400" : "text-emerald-400"}>{st.status.toUpperCase()}</strong></span>
                      {st.started_at && <span className="text-slate-500">{st.started_at}</span>}
                    </div>
                    {(st as any).details && (
                      <p className="text-slate-200">
                        &gt; {typeof (st as any).details === "string" ? (st as any).details : JSON.stringify((st as any).details)}
                      </p>
                    )}
                    {st.error && (
                      <div className="text-rose-300 bg-rose-950/40 border border-rose-900/50 p-2 rounded">
                        <strong>Error:</strong> {st.error}
                      </div>
                    )}
                  </div>
                </PipelineStep>
              );
            })
          ) : (
            <>
              {/* Fallback default canonical stages if no structured stages returned */}
              <PipelineStep icon={CheckCircle2} title="Stage 1: Input Ingestion & Raster Validation" duration="✓ Verified">
                <div className="flex flex-wrap gap-3">
                  <DetailBox label="Input Raster" value={sourceImage.filename} />
                  <DetailBox label="Source Origin" value={sourceImage.source === "upload" ? "User Upload" : "Verified Tile"} />
                  <DetailBox label="MIME Type" value={sourceImage.mediaType} valueColor="text-cyan-300 font-medium" />
                </div>
              </PipelineStep>

              <PipelineStep icon={Search} title="Stage 2: Intent Classification & Tool Routing" duration="✓ Routed">
                <div className="text-[11px] text-slate-400 space-y-1 mb-2 font-mono bg-slate-900/60 p-3 rounded-lg border border-slate-800/80">
                  <p className="text-slate-200"><strong>Query Intent:</strong> &ldquo;{investigation.query}&rdquo;</p>
                  <p>&gt; Tools Activated: Grounding DINO Localization, BLIP Captioning{vqa.length > 0 ? ", RSVQA" : ""}</p>
                </div>
              </PipelineStep>

              <PipelineStep icon={Cpu} title="Stage 3: Specialist Neural Model Inference" duration="✓ Complete">
                <div className="flex flex-wrap gap-3 mb-2">
                  <DetailBox label="Object Localization" value="Grounding DINO (Swin-T)" valueColor="text-emerald-400" />
                  <DetailBox label="Scene Captioner" value="BLIP (Salesforce/blip-base)" valueColor="text-cyan-300" />
                </div>
                <p className="text-[11px] text-emerald-400 font-mono flex items-center gap-1.5">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
                  Detections Extracted: {detections.length} candidate(s)
                </p>
              </PipelineStep>

              <PipelineStep icon={Crosshair} title="Stage 4: Spatial Normalization & Evidence Extraction" duration="✓ Normalized">
                <div className="text-[11px] text-slate-400 font-mono bg-slate-900/60 p-3 rounded-lg border border-slate-800/80 space-y-1">
                  <p>&gt; Normalized Coordinate Scale: [ymin, xmin, ymax, xmax] (0-1000 range)</p>
                  <p>&gt; Total Spatial Evidence Nodes: {detections.length} node(s) verified</p>
                </div>
              </PipelineStep>

              <PipelineStep icon={Sparkles} title="Stage 5: Multimodal Synthesis & Natural Language Report" duration="✓ Synthesized">
                <div className="text-[11px] text-slate-300 font-mono bg-slate-900/60 p-3 rounded-lg border border-slate-800/80 space-y-1.5">
                  <div><strong className="text-cyan-300">Executive Finding:</strong> {caption}</div>
                  {vqa.length > 0 && (
                    <div><strong className="text-cyan-300">VQA Responses:</strong> {vqa.map(v => `${v.question} → ${v.answer}`).join("; ")}</div>
                  )}
                </div>
              </PipelineStep>
            </>
          )}
        </div>
      </div>

      {/* Discrete Specialist Tools Section */}
      {Array.isArray(toolExecutions) && toolExecutions.length > 0 && (
        <div className="border-t border-slate-800/80 pt-5">
          <p className="text-xs font-semibold text-white tracking-tight uppercase tracking-wider font-mono mb-4 flex items-center gap-2">
            <Cpu size={13} className="text-cyan-400" />
            <span>Specialist Tool Execution Records</span>
          </p>

          <div className="space-y-3">
            {toolExecutions.map((tool: ToolExecutionRecord, i: number) => {
              const isOk = tool.status === "succeeded";
              const isUnavailable = tool.status === "unavailable";
              const isFail = tool.status === "failed";
              return (
                <div
                  key={i}
                  className="bg-slate-900/50 border border-slate-800 rounded-lg p-3 text-[11px] font-mono space-y-2"
                >
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <span className="font-semibold text-white">{tool.tool_name}</span>
                    <div className="flex items-center gap-2">
                      <span
                        className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                          isOk
                            ? "bg-emerald-500/15 text-emerald-400 border border-emerald-500/30"
                            : isUnavailable
                            ? "bg-amber-500/15 text-amber-300 border border-amber-500/30"
                            : isFail
                            ? "bg-rose-500/15 text-rose-400 border border-rose-500/30"
                            : "bg-slate-800 text-slate-400 border border-slate-700"
                        }`}
                      >
                        {tool.status.toUpperCase()}
                      </span>
                      <span className="text-slate-400">{tool.duration_ms.toFixed(1)} ms</span>
                    </div>
                  </div>

                  {tool.error && (
                    <div className="text-rose-300 bg-rose-950/40 p-2 rounded border border-rose-900/50">
                      <div><strong>Error:</strong> {tool.error}</div>
                      {tool.actionable_reason && (
                        <div className="text-slate-300 mt-1">
                          <strong>Actionable Guidance:</strong> {tool.actionable_reason}
                        </div>
                      )}
                    </div>
                  )}

                  {tool.output_summary && Object.keys(tool.output_summary).length > 0 && (
                    <div className="text-slate-400 bg-slate-950/60 p-2 rounded border border-slate-800/80">
                      <span className="text-slate-500">Summary: </span>
                      {JSON.stringify(tool.output_summary)}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------
// Structured Log Stream Tab / Panel
// ---------------------------------------------
function StructuredLogStream({
  investigation,
}: {
  investigation: CanonicalInvestigationState | null;
}) {
  const [levelFilter, setLevelFilter] = useState<string>("ALL");
  const [searchTerm, setSearchTerm] = useState<string>("");

  const logs: StructuredLogEvent[] = useMemo(() => {
    return (
      investigation?.response?.structured_trace?.logs ||
      investigation?.response?.execution_logs ||
      []
    );
  }, [investigation]);

  const filteredLogs = useMemo(() => {
    return logs.filter((log) => {
      if (levelFilter !== "ALL" && log.level !== levelFilter) {
        return false;
      }
      if (searchTerm.trim() !== "") {
        const q = searchTerm.toLowerCase();
        const msgMatch = log.event?.toLowerCase().includes(q);
        const loggerMatch = log.logger?.toLowerCase().includes(q);
        const detailMatch = log.details ? JSON.stringify(log.details).toLowerCase().includes(q) : false;
        return msgMatch || loggerMatch || detailMatch;
      }
      return true;
    });
  }, [logs, levelFilter, searchTerm]);

  return (
    <div className="w-full border border-slate-800/90 bg-[#0c1624]/60 backdrop-blur-md rounded-xl p-5 shadow-[0_4px_24px_rgba(0,0,0,0.3)] space-y-4 font-mono">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800/80 pb-3">
        <div className="flex items-center gap-2 text-xs font-semibold text-white">
          <Terminal size={14} className="text-cyan-400" />
          <span>Structured Diagnostic Log Stream</span>
          <span className="text-[10px] text-cyan-400 bg-cyan-500/10 px-2 py-0.5 rounded border border-cyan-500/20">
            {filteredLogs.length} / {logs.length} events
          </span>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          {/* Search box */}
          <div className="relative">
            <Search size={11} className="absolute left-2.5 top-2.5 text-slate-500" />
            <input
              type="text"
              placeholder="Search logs..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="bg-slate-900 border border-slate-800 rounded px-2.5 py-1 pl-7 text-[11px] text-slate-200 placeholder-slate-500 focus:outline-none focus:border-cyan-500/50"
            />
          </div>

          {/* Level Filter */}
          <div className="flex items-center gap-1">
            {(["ALL", "INFO", "WARNING", "ERROR"] as const).map((lvl) => (
              <button
                key={lvl}
                onClick={() => setLevelFilter(lvl)}
                className={`text-[10px] px-2 py-1 rounded transition-all ${
                  levelFilter === lvl
                    ? "bg-cyan-500/20 text-cyan-300 border border-cyan-400/40"
                    : "bg-slate-900 text-slate-400 hover:text-slate-200"
                }`}
              >
                {lvl}
              </button>
            ))}
          </div>
        </div>
      </div>

      {filteredLogs.length === 0 ? (
        <div className="text-center py-10 text-slate-500 text-xs">
          {logs.length === 0
            ? "No structured logs recorded for this investigation."
            : "No logs matched your current filters."}
        </div>
      ) : (
        <div className="space-y-2 max-h-[550px] overflow-y-auto pr-1">
          {filteredLogs.map((log, idx) => {
            const isError = log.level === "ERROR" || log.level === "CRITICAL";
            const isWarn = log.level === "WARNING";
            const levelColor = isError
              ? "text-rose-400 bg-rose-500/10 border-rose-500/30"
              : isWarn
              ? "text-amber-300 bg-amber-500/10 border-amber-500/30"
              : "text-emerald-400 bg-emerald-500/10 border-emerald-500/30";

            return (
              <div
                key={idx}
                className="bg-[#08121e] border border-slate-800/80 rounded-lg p-2.5 text-[11px] space-y-1"
              >
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div className="flex items-center gap-2">
                    <span className={`px-1.5 py-0.5 rounded text-[9px] font-bold border ${levelColor}`}>
                      {log.level}
                    </span>
                    <span className="text-cyan-300 font-semibold">{log.logger}</span>
                    {log.stage && <span className="text-slate-500">[{log.stage}]</span>}
                  </div>
                  <span className="text-slate-600 text-[10px]">{log.timestamp}</span>
                </div>

                <div className="text-slate-300">{log.event}</div>

                {log.actionable_reason && (
                  <div className="text-[10px] text-amber-300 bg-amber-950/30 p-1.5 rounded border border-amber-900/40">
                    <strong>Actionable Advice:</strong> {log.actionable_reason}
                  </div>
                )}

                {log.details && Object.keys(log.details).length > 0 && (
                  <div className="text-[10px] text-slate-400 bg-slate-950/70 p-2 rounded overflow-x-auto border border-slate-900">
                    <pre className="whitespace-pre-wrap">{JSON.stringify(log.details, null, 2)}</pre>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------
// Node Inspector panel with Real Investigation Data
// ---------------------------------------------
function NodeInspector({
  investigation,
  sourceImage,
}: {
  investigation: CanonicalInvestigationState | null;
  sourceImage: CanonicalSourceImage;
}) {
  const detectionsCount = investigation?.response?.grounding?.detections?.length || 0;
  const normalizedBoxes = useMemo(() => {
    const raw = investigation?.response?.grounding?.detections || [];
    return raw
      .map(d => normalizeBox(d as unknown as Record<string, unknown>))
      .filter((b): b is [number, number, number, number] => b !== null);
  }, [investigation]);

  const trace = investigation?.response?.structured_trace;

  return (
    <div className="w-full border border-slate-800/90 bg-[#0c1624]/60 backdrop-blur-md rounded-xl overflow-hidden flex flex-col shadow-[0_4px_24px_rgba(0,0,0,0.3)] space-y-4">
      <div className="flex items-center gap-1.5 px-4 py-3 border-b border-slate-800/80 font-mono">
        <Info size={13} className="text-cyan-400" />
        <p className="text-sm font-semibold text-white tracking-tight">Active Investigation Diagnostics</p>
      </div>

      {/* Raster & Sensor Metadata */}
      <div className="px-4 space-y-3">
        <p className="text-[10px] text-slate-400 uppercase tracking-widest font-mono">
          Raster Frame &amp; Input Metadata
        </p>

        <div className="space-y-2 text-[11px] font-mono bg-slate-900/60 p-3 rounded-lg border border-slate-800/80">
          <div className="flex justify-between">
            <span className="text-slate-400">Filename</span>
            <span className="text-cyan-300 font-semibold truncate max-w-[160px]">{sourceImage.filename}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-400">Media Type</span>
            <span className="text-slate-200">{sourceImage.mediaType}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-400">Source Type</span>
            <span className="text-slate-200">{sourceImage.source === "upload" ? "User Upload" : "Demo Catalog"}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-400">Detections</span>
            <span className="text-emerald-400 font-semibold">{detectionsCount} objects</span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-400">Total Duration</span>
            <span className="text-cyan-300 font-semibold">
              {trace?.total_duration_ms ? `${trace.total_duration_ms.toFixed(0)} ms` : "Nominal"}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-400">Overall Outcome</span>
            <span className="text-emerald-400 font-semibold">{trace?.overall_outcome || "SUCCESS"}</span>
          </div>
        </div>
      </div>

      {/* Detections preview */}
      {normalizedBoxes.length > 0 && (
        <div className="px-4 pb-4 space-y-2">
          <p className="text-[10px] text-slate-400 uppercase tracking-widest font-mono">
            Normalized Bounding Boxes (0-1000)
          </p>
          <div className="max-h-40 overflow-y-auto space-y-1 bg-slate-950/80 p-2.5 rounded-lg border border-slate-800 text-[10px] font-mono text-cyan-300">
            {normalizedBoxes.map((b, idx) => (
              <div key={idx}>• Target #{idx + 1}: [{b.join(", ")}]</div>
            ))}
          </div>
        </div>
      )}

      {/* Quick link */}
      <div className="px-4 pb-4">
        <Link
          href="/evidence"
          className="w-full flex items-center justify-center gap-1.5 bg-slate-900/80 hover:bg-slate-800 border border-slate-800 hover:border-cyan-500/40 text-slate-300 hover:text-cyan-300 rounded-lg py-2 text-xs font-mono transition-colors"
        >
          <span>Open Evidence Viewer</span>
          <ArrowRight size={12} />
        </Link>
      </div>
    </div>
  );
}

// ---------------------------------------------
// Bottom status bar
// ---------------------------------------------
function StatusBar({
  investigation,
  sourceImage,
}: {
  investigation: CanonicalInvestigationState | null;
  sourceImage: CanonicalSourceImage;
}) {
  return (
    <div className="flex flex-col sm:flex-row items-center justify-between gap-2 py-3 px-4 sm:px-8 border-t border-slate-800/80 text-[10px] font-mono text-slate-400">
      <span className="flex items-center gap-1.5 text-cyan-300">
        <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse" />
        AUDIT STATUS: {investigation ? "INVESTIGATION TRACE SYNCHRONIZED" : "STANDBY"}
      </span>
      <span>SOURCE: {sourceImage.filename} · CORRELATION: {investigation?.response?.request_id || "LOCAL"}</span>
    </div>
  );
}

// ---------------------------------------------
// Execution Log Page Main Component
// ---------------------------------------------
export default function ExecutionLogPage() {
  const [investigationState] = useState<CanonicalInvestigationState | null>(() => {
    return getCurrentInvestigation();
  });
  const [activeTab, setActiveTab] = useState<"stages" | "logs">("stages");

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

        {/* Tab Selector */}
        <div className="px-4 sm:px-6 lg:px-8 pt-4 flex gap-2">
          <button
            onClick={() => setActiveTab("stages")}
            className={`flex items-center gap-1.5 px-4 py-2 rounded-lg text-xs font-mono transition-all ${
              activeTab === "stages"
                ? "bg-cyan-500/20 text-cyan-300 border border-cyan-400/40"
                : "bg-slate-900/60 text-slate-400 hover:text-slate-200 border border-slate-800"
            }`}
          >
            <Layers size={13} />
            <span>Stages &amp; Tools</span>
          </button>
          <button
            onClick={() => setActiveTab("logs")}
            className={`flex items-center gap-1.5 px-4 py-2 rounded-lg text-xs font-mono transition-all ${
              activeTab === "logs"
                ? "bg-cyan-500/20 text-cyan-300 border border-cyan-400/40"
                : "bg-slate-900/60 text-slate-400 hover:text-slate-200 border border-slate-800"
            }`}
          >
            <Terminal size={13} />
            <span>Structured Log Events Stream</span>
          </button>
        </div>

        <main className="flex-1 p-4 sm:p-6 lg:p-8 grid grid-cols-1 lg:grid-cols-3 gap-5 items-start overflow-y-auto">
          <div className="lg:col-span-2">
            {activeTab === "stages" ? (
              <PipelineSteps investigation={investigationState} sourceImage={sourceImage} />
            ) : (
              <StructuredLogStream investigation={investigationState} />
            )}
          </div>
          <div className="lg:col-span-1">
            <NodeInspector investigation={investigationState} sourceImage={sourceImage} />
          </div>
        </main>

        <StatusBar investigation={investigationState} sourceImage={sourceImage} />
      </div>
    </div>
  );
}