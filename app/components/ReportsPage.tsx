"use client";

import { useState, useMemo } from "react";
import Link from "next/link";
import {
  FileText,
  Search,
  Printer,
} from "lucide-react";
import Sidebar from "./Sidebar";
import type { CanonicalSourceImage, CanonicalInvestigationState } from "../types/nexspace";
import { SAMPLE_OPTICAL_PORT } from "../utils/sampleImages";
import { normalizeBox } from "./QueryPage";
import {
  getCurrentInvestigation,
  getActiveSourceImage,
  DEFAULT_DEMO_SOURCE,
} from "../utils/investigationStorage";

function Section({
  number,
  title,
  badge,
  children,
}: {
  number: string;
  title: string;
  badge?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="rounded-xl border border-slate-800/80 bg-[#0c1624]/60 backdrop-blur-md p-5 sm:p-6 shadow-[0_4px_24px_rgba(0,0,0,0.3)]">
      <div className="flex items-center justify-between pb-3 mb-4 border-b border-slate-800/80">
        <div className="flex items-center gap-2">
          <span className="font-mono text-cyan-400 font-bold text-sm sm:text-base">
            {number}
          </span>
          <h2 className="text-sm sm:text-base font-semibold text-white tracking-tight">
            {title}
          </h2>
        </div>
        {badge && (
          <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-bold">
            {badge}
          </span>
        )}
      </div>
      {children}
    </div>
  );
}

export default function ReportsPage() {
  const [investigationState] = useState<CanonicalInvestigationState | null>(() => {
    return getCurrentInvestigation();
  });

  const sourceImage = useMemo<CanonicalSourceImage>(() => {
    if (investigationState?.source_image?.dataUrl) {
      return investigationState.source_image;
    }
    const active = getActiveSourceImage(false);
    if (active) return active;
    return DEFAULT_DEMO_SOURCE;
  }, [investigationState]);

  const normalizedDetections = useMemo(() => {
    const raw = investigationState?.response?.grounding?.detections || [];
    return raw
      .map((det, idx) => {
        const d = det as unknown as Record<string, unknown>;
        const nBox = normalizeBox(d);
        if (!nBox) return null;
        return {
          id: `target-${idx + 1}`,
          label: (d.label as string) || `Target #${idx + 1}`,
          score: typeof d.score === "number" ? d.score : null,
          box: nBox,
        };
      })
      .filter((d): d is NonNullable<typeof d> => d !== null);
  }, [investigationState]);

  const caption = typeof investigationState?.response?.optical_caption === "string"
    ? investigationState.response.optical_caption
    : (investigationState?.response?.optical_caption as unknown as { caption?: string })?.caption || "Scene analysis completed.";

  const vqa = investigationState?.response?.vqa_results || [];
  const isUpload = sourceImage.source === "upload";

  const handlePrint = () => {
    window.print();
  };

  return (
    <div className="flex flex-col lg:flex-row min-h-screen bg-[#0a1420] text-white font-sans overflow-x-hidden">
      <Sidebar />

      <div className="flex-1 flex flex-col min-w-0 overflow-y-auto">
        {/* Header */}
        <header className="flex flex-col sm:flex-row sm:items-start justify-between gap-4 px-4 sm:px-8 py-5 border-b border-slate-800/80 bg-[#09131f]/60 backdrop-blur-md">
          <div>
            <div className="flex flex-wrap items-center gap-2">
              <div className="p-1.5 rounded-lg bg-cyan-500/10 border border-cyan-500/20 text-cyan-400">
                <FileText size={15} />
              </div>
              <h1 className="text-base sm:text-lg font-bold text-white tracking-tight">
                Remote Sensing Intelligence Assessment Report
              </h1>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-cyan-500/15 text-cyan-300 border border-cyan-500/30">
                {investigationState ? "LIVE INVESTIGATION" : "STANDBY"}
              </span>
            </div>
            <p className="mt-1 text-[11px] text-slate-400 font-mono">
              Report ID: <strong className="text-cyan-300">{investigationState?.investigation_id || "INV-STANDBY"}</strong> &nbsp;·&nbsp;
              Target: <strong className="text-white">{sourceImage.filename}</strong> ({isUpload ? "Uploaded by user" : "Demo raster"})
              {investigationState?.timestamp && ` · Generated: ${new Date(investigationState.timestamp).toLocaleTimeString()}`}
            </p>
          </div>

          <div className="flex items-center gap-2.5 shrink-0 self-end sm:self-auto">
            <button
              onClick={handlePrint}
              className="flex items-center gap-1.5 text-xs font-semibold text-slate-300 hover:text-cyan-300 border border-slate-800 hover:border-cyan-500/40 rounded-lg px-3.5 py-2 bg-slate-900/60 hover:bg-slate-900/90 transition-all cursor-pointer shadow-sm"
            >
              <Printer size={13} />
              <span>Print / Save PDF</span>
            </button>
            <Link
              href="/query"
              className="flex items-center gap-1.5 text-xs font-bold text-[#071320] bg-gradient-to-r from-cyan-500 to-sky-400 hover:from-cyan-400 hover:to-sky-300 rounded-lg px-4 py-2 transition-all cursor-pointer"
            >
              <Search size={13} />
              <span>New Query</span>
            </Link>
          </div>
        </header>

        <main className="p-4 sm:p-6 lg:p-8 flex flex-col gap-6">
          {!investigationState ? (
            <div className="rounded-xl border border-slate-800/80 bg-[#0c1624]/60 p-8 text-center space-y-4">
              <FileText size={36} className="text-slate-600 mx-auto" />
              <h2 className="text-base font-semibold text-white">No Active Investigation Report</h2>
              <p className="text-xs text-slate-400 font-mono max-w-md mx-auto">
                Execute a query in the Query Terminal to generate a real-time multimodal intelligence report with dynamic bounding box evidence.
              </p>
              <Link
                href="/query"
                className="inline-flex items-center gap-2 bg-cyan-500 hover:bg-cyan-400 text-[#0a1420] text-xs font-bold px-4 py-2 rounded-lg transition-all"
              >
                <Search size={14} />
                <span>Open Query Terminal</span>
              </Link>
            </div>
          ) : (
            <>
              {/* 1.0 Executive Summary */}
              <Section number="1.0" title="Executive Summary">
                <div className="space-y-3 text-xs sm:text-sm text-slate-300 leading-relaxed font-sans">
                  <p>
                    Automated remote sensing analysis was executed for query:{" "}
                    <strong className="text-cyan-300 font-mono">&ldquo;{investigationState.query}&rdquo;</strong> over source imagery{" "}
                    <strong className="text-white font-mono">{sourceImage.filename}</strong>.
                  </p>
                  <p>
                    <strong className="text-white">Scene Overview:</strong> {caption}
                  </p>
                  {normalizedDetections.length > 0 ? (
                    <p>
                      Neural object localization identified <strong className="text-emerald-400">{normalizedDetections.length} candidate structure(s)</strong> within the active scene.
                    </p>
                  ) : (
                    <p className="text-slate-400">
                      No candidate spatial bounding boxes were localized for the target prompt in this raster frame.
                    </p>
                  )}
                  {vqa.length > 0 && (
                    <div className="pt-2 border-t border-slate-800/60 text-xs font-mono space-y-1">
                      <div className="text-slate-400 font-semibold">Visual Q&amp;A Responses:</div>
                      {vqa.map((v, i) => (
                        <div key={i} className="text-slate-200">
                          • {v.question} → <strong className="text-cyan-300">{v.answer}</strong>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </Section>

              {/* 2.0 Spatial Analysis & Detection Overlay */}
              <Section number="2.0" title="Spatial Analysis & Detection Overlays" badge={`${normalizedDetections.length} Targets`}>
                <div className="relative rounded-xl overflow-hidden border border-slate-800 bg-[#08121e] min-h-[380px] flex flex-col justify-between p-3">
                  <div className="relative w-full flex-1 flex items-center justify-center overflow-hidden min-h-[340px]">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      src={sourceImage.dataUrl}
                      alt={sourceImage.filename}
                      className="max-h-[340px] w-auto object-contain select-none pointer-events-none"
                    />

                    {/* Dynamic Grounding Overlays */}
                    {normalizedDetections.map((d) => {
                      const [xmin, ymin, xmax, ymax] = d.box;
                      const top = (ymin / 1000) * 100;
                      const left = (xmin / 1000) * 100;
                      const width = Math.max(2, ((xmax - xmin) / 1000) * 100);
                      const height = Math.max(2, ((ymax - ymin) / 1000) * 100);

                      return (
                        <div
                          key={d.id}
                          className="absolute border-2 border-cyan-400 bg-cyan-500/20 rounded flex flex-col justify-between p-1 z-10 pointer-events-none shadow-[0_0_12px_rgba(6,182,212,0.5)]"
                          style={{
                            top: `${top}%`,
                            left: `${left}%`,
                            width: `${width}%`,
                            height: `${height}%`,
                          }}
                        >
                          <span className="text-[8px] font-mono font-bold bg-slate-900/90 text-cyan-300 px-1 py-0.5 rounded self-start">
                            {d.label} {d.score !== null ? `(${Math.round(d.score * 100)}%)` : ""}
                          </span>
                        </div>
                      );
                    })}
                  </div>

                  <div className="relative z-10 flex items-center justify-between text-[10px] font-mono text-slate-400 bg-slate-900/85 backdrop-blur-md px-3 py-1.5 rounded-lg border border-slate-800 mt-2">
                    <span>Source: {sourceImage.filename} ({sourceImage.source === "upload" ? "Uploaded by user" : "Demo raster"})</span>
                    <span>Format: {sourceImage.mediaType} · 512x512 Frame</span>
                  </div>
                </div>
              </Section>

              {/* 3.0 Technical Appendix */}
              <Section number="3.0" title="Technical Appendix & Provenance">
                <div className="overflow-x-auto w-full">
                  <table className="w-full text-xs min-w-[580px]">
                    <thead>
                      <tr className="text-left text-slate-400 uppercase tracking-widest font-mono text-[10px] border-b border-slate-800 pb-2">
                        <th className="pb-2.5 pr-4 font-normal">Target Entity</th>
                        <th className="pb-2.5 pr-4 font-normal">Model</th>
                        <th className="pb-2.5 pr-4 font-normal">Normalized Box [ymin, xmin, ymax, xmax]</th>
                        <th className="pb-2.5 font-normal text-right">Confidence</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800/60 font-mono text-[11px]">
                      {normalizedDetections.map((d, i) => (
                        <tr key={i} className="hover:bg-cyan-500/[0.04]">
                          <td className="py-2.5 pr-4 text-cyan-400 font-semibold">{d.label} #{i + 1}</td>
                          <td className="py-2.5 pr-4 text-slate-300">Grounding DINO (Swin-T)</td>
                          <td className="py-2.5 pr-4 text-slate-400">[{d.box.join(", ")}]</td>
                          <td className="py-2.5 font-normal text-right text-emerald-400">
                            {d.score !== null ? `${Math.round(d.score * 100)}%` : "N/A"}
                          </td>
                        </tr>
                      ))}
                      {normalizedDetections.length === 0 && (
                        <tr>
                          <td colSpan={4} className="py-4 text-center text-slate-500 italic">
                            No individual bounding targets generated for this query.
                          </td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>

                <div className="mt-4 pt-3 border-t border-slate-800/60 text-[11px] font-mono text-slate-400 space-y-1">
                  <div>• <strong>Captioning Model:</strong> Salesforce/blip-image-captioning-base</div>
                  <div>• <strong>RSVQA Engine:</strong> {vqa.length > 0 ? "PaliGemma RSVQA / Multimodal Fallback" : "Standby"}</div>
                  <div>• <strong>Geospatial CRS:</strong> {investigationState.response?.geospatial_metadata?.crs || "Pixel frame reference"}</div>
                </div>
              </Section>
            </>
          )}
        </main>
      </div>
    </div>
  );
}