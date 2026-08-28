"use client";

import { motion } from "framer-motion";
import { Download, Share2, TrendingUp, FileText, CheckCircle2, MapPin, Activity, Calendar, Globe2, ShieldCheck } from "lucide-react";
import Sidebar from "./Sidebar";

export default function ReportsPage() {
  return (
    <div className="flex flex-col lg:flex-row min-h-screen bg-[#0a1420] text-white font-sans overflow-x-hidden">
      <Sidebar />

      <div className="flex-1 flex flex-col min-w-0 overflow-y-auto">
        {/* Header */}
        <header className="flex flex-col sm:flex-row sm:items-start justify-between gap-4 px-4 sm:px-8 py-5 border-b border-slate-800/80 bg-[#09131f]/60 backdrop-blur-md">
          <div>
            <div className="flex items-center gap-2">
              <div className="p-1.5 rounded-lg bg-cyan-500/10 border border-cyan-500/20 text-cyan-400">
                <FileText size={15} />
              </div>
              <h1 className="text-base sm:text-lg font-bold text-white tracking-tight">
                Vessel Activity Assessment Report
              </h1>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-cyan-500/15 text-cyan-300 border border-cyan-500/30">
                INTEL REPORT
              </span>
            </div>
            <p className="mt-1 text-[11px] text-slate-400 font-mono">
              ID: SQ-REP-2023-11A &nbsp;·&nbsp; Generated: 2023-10-27T08:14Z
              &nbsp;·&nbsp; Target: Sector 7B (South China Sea)
            </p>
          </div>

          <div className="flex items-center gap-2.5 shrink-0 self-end sm:self-auto">
            <button className="flex items-center gap-1.5 text-xs font-semibold text-slate-300 hover:text-cyan-300 border border-slate-800 hover:border-cyan-500/40 rounded-lg px-3.5 py-2 bg-slate-900/60 hover:bg-slate-900/90 transition-all duration-180 hover:scale-[1.02] active:scale-[0.98] cursor-pointer shadow-sm">
              <Download size={13} />
              <span>Export PDF</span>
            </button>
            <button className="flex items-center gap-1.5 text-xs font-bold text-[#071320] bg-gradient-to-r from-cyan-500 to-sky-400 hover:from-cyan-400 hover:to-sky-300 rounded-lg px-4 py-2 transition-all duration-180 hover:scale-[1.02] active:scale-[0.98] shadow-[0_0_15px_rgba(6,182,212,0.35)] hover:shadow-[0_0_20px_rgba(6,182,212,0.55)] cursor-pointer">
              <Share2 size={13} />
              <span>Share Intel</span>
            </button>
          </div>
        </header>

        <main className="p-4 sm:p-6 lg:p-8 flex flex-col gap-6">
          {/* 1.0 Executive Summary */}
          <Section number="1.0" title="Executive Summary">
            <div className="space-y-3">
              <p className="text-xs sm:text-sm text-slate-300 leading-relaxed font-sans">
                Analysis of multispectral imagery acquired between Oct 15 and Oct
                25 indicates a <strong className="text-white">14.2% increase</strong> in large vessel concentration
                within the designated AoI (Area of Interest). Automated
                neural models identified a clustering pattern consistent with
                recent maritime fleet movements in adjacent international shipping corridors.
              </p>
              <p className="text-xs sm:text-sm text-slate-300 leading-relaxed font-sans">
                Confidence levels for primary detection models exceed <strong className="text-emerald-400 font-mono">94%</strong>, with
                minimal cloud occlusion (avg 3.2%) during the observation
                window. Strategic recommendation involves continued daily
                automated tasking over coordinates{" "}
                <span className="font-mono text-cyan-300 font-semibold px-2 py-0.5 rounded bg-cyan-500/10 border border-cyan-500/25">
                  12°34&apos;N 114°21&apos;E
                </span>
                .
              </p>
            </div>
          </Section>

          {/* 2.0 Spatial Analysis + 3.0 Temporal Trends */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Spatial Analysis */}
            <Section number="2.0" title="Spatial Analysis" badge="High Confidence">
              <div className="relative rounded-xl overflow-hidden border border-slate-800 bg-[#08121e] h-56 flex flex-col justify-between p-3">
                <SatelliteMap />
                <div className="relative z-10 flex items-center justify-between">
                  <div className="text-[10px] font-mono text-cyan-300 bg-slate-900/85 backdrop-blur-md px-2.5 py-1 rounded-md border border-slate-800">
                    Sector 7B Target AOI
                  </div>
                  <div className="text-[10px] font-mono text-slate-400 bg-slate-900/85 backdrop-blur-md px-2.5 py-1 rounded-md border border-slate-800">
                    21:14:07 UTC · Cloud: 3.2%
                  </div>
                </div>
              </div>
            </Section>

            {/* Temporal Trends */}
            <Section number="3.0" title="Temporal Trends">
              <div className="flex gap-8 mb-4">
                <Stat label="Total Detected" value="342" trend="+14%" />
                <Stat label="Avg Size (m)" value="114" trend="-2%" />
              </div>
              <BarChart values={[38, 52, 46, 71, 90, 100, 82]} />
            </Section>
          </div>

          {/* 4.0 Technical Appendix */}
          <Section number="4.0" title="Technical Appendix">
            <div className="overflow-x-auto w-full">
              <table className="w-full text-xs min-w-[580px]">
                <thead>
                  <tr className="text-left text-slate-400 uppercase tracking-widest font-mono text-[10px] border-b border-slate-800 pb-2">
                    <th className="pb-2.5 pr-4 font-normal">Dataset ID</th>
                    <th className="pb-2.5 pr-4 font-normal">Timestamp (Z)</th>
                    <th className="pb-2.5 pr-4 font-normal">Sensor</th>
                    <th className="pb-2.5 pr-4 font-normal">Coordinates (Center)</th>
                    <th className="pb-2.5 pr-4 font-normal">Resolution</th>
                    <th className="pb-2.5 font-normal text-right">Confidence</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 font-mono text-[11px]">
                  {[
                    {
                      id: "SQ-IMG-0812",
                      ts: "2023-10-25T14:32:11",
                      sensor: "SAR-X Band",
                      coords: "12.56N, 114.35E",
                      res: "0.5m GSD",
                      conf: "98.2%",
                    },
                    {
                      id: "SQ-IMG-0813",
                      ts: "2023-10-26T02:11:45",
                      sensor: "Sentinel-2 MSI",
                      coords: "12.55N, 114.33E",
                      res: "10m GSD",
                      conf: "94.1%",
                    },
                    {
                      id: "SQ-IMG-0814",
                      ts: "2023-10-27T08:00:02",
                      sensor: "WorldView-3",
                      coords: "12.58N, 114.37E",
                      res: "0.3m GSD",
                      conf: "99.0%",
                    },
                  ].map((row) => (
                    <tr
                      key={row.id}
                      className="hover:bg-cyan-500/[0.04] transition-colors duration-150 group cursor-default"
                    >
                      <td className="py-3 pr-4 text-cyan-400 font-semibold group-hover:text-cyan-300">
                        {row.id}
                      </td>
                      <td className="py-3 pr-4 text-slate-400">{row.ts}</td>
                      <td className="py-3 pr-4 text-slate-200">{row.sensor}</td>
                      <td className="py-3 pr-4 text-slate-400">{row.coords}</td>
                      <td className="py-3 pr-4 text-slate-400">{row.res}</td>
                      <td className="py-3 text-right text-emerald-400 font-semibold">
                        <span className="bg-emerald-500/10 border border-emerald-500/20 px-2 py-0.5 rounded">
                          {row.conf}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>
        </main>
      </div>
    </div>
  );
}

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
    <div className="w-full border border-slate-800/80 bg-[#0c1624]/60 backdrop-blur-md rounded-xl p-4 sm:p-5 shadow-[0_4px_24px_rgba(0,0,0,0.3)]">
      <div className="flex items-center justify-between mb-4 border-b border-slate-800/60 pb-2.5">
        <h2 className="text-xs font-bold tracking-wide text-slate-200 uppercase font-mono flex items-center">
          <span className="text-cyan-400 mr-2">{number}</span>
          {title}
        </h2>
        {badge && (
          <span className="text-[10px] font-mono font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 px-2.5 py-0.5 rounded-full flex items-center gap-1">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
            {badge}
          </span>
        )}
      </div>
      {children}
    </div>
  );
}

function Stat({
  label,
  value,
  trend,
}: {
  label: string;
  value: string;
  trend: string;
}) {
  const positive = trend.startsWith("+");
  return (
    <div>
      <div className="text-[10px] text-slate-400 uppercase tracking-widest font-mono mb-1">
        {label}
      </div>
      <div className="text-2xl font-bold text-white tracking-tight">{value}</div>
      <div
        className={`flex items-center gap-1 text-[11px] font-mono mt-0.5 ${
          positive ? "text-emerald-400" : "text-rose-400"
        }`}
      >
        <TrendingUp size={11} className={positive ? "" : "rotate-180"} />
        {trend}
      </div>
    </div>
  );
}

function BarChart({ values }: { values: number[] }) {
  const max = Math.max(...values);
  return (
    <div className="flex items-end gap-2 h-28 pt-2">
      {values.map((v, i) => (
        <motion.div
          key={i}
          initial={{ height: 0, opacity: 0 }}
          animate={{ height: `${(v / max) * 100}%`, opacity: 1 }}
          transition={{ duration: 0.4, delay: i * 0.05, ease: "easeOut" }}
          className="flex-1 rounded-t bg-gradient-to-t from-cyan-600/30 via-cyan-500/60 to-cyan-400 hover:to-cyan-300 transition-colors cursor-pointer group relative"
        >
          <div className="opacity-0 group-hover:opacity-100 transition-opacity absolute -top-6 left-1/2 -translate-x-1/2 bg-slate-900 text-cyan-300 font-mono text-[9px] px-1 rounded border border-slate-700 pointer-events-none">
            {v}
          </div>
        </motion.div>
      ))}
    </div>
  );
}

function SatelliteMap() {
  return (
    <div className="absolute inset-0 overflow-hidden">
      {/* HUD Grid */}
      <div
        className="absolute inset-0 opacity-10 pointer-events-none"
        style={{
          backgroundImage:
            "linear-gradient(rgba(56,189,248,0.2) 1px, transparent 1px), linear-gradient(90deg, rgba(56,189,248,0.2) 1px, transparent 1px)",
          backgroundSize: "24px 24px",
        }}
      />

      {/* Slow continuous scan line */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div
          className="w-full h-[2px] bg-gradient-to-r from-transparent via-cyan-400/40 to-transparent animate-scan"
          style={{ animationDuration: "5s" }}
        />
      </div>

      <svg viewBox="0 0 400 200" className="w-full h-full" preserveAspectRatio="xMidYMid slice">
        <rect width="400" height="200" fill="#08121e" />
        {Array.from({ length: 10 }).map((_, i) => (
          <line
            key={`h${i}`}
            x1="0"
            y1={i * 20}
            x2="400"
            y2={i * 20}
            stroke="#1e293b"
            strokeWidth="0.5"
          />
        ))}
        {Array.from({ length: 20 }).map((_, i) => (
          <line
            key={`v${i}`}
            x1={i * 20}
            y1="0"
            x2={i * 20}
            y2="200"
            stroke="#1e293b"
            strokeWidth="0.5"
          />
        ))}

        {/* Target 1 with pulsing indicator */}
        <g>
          <rect
            x="150"
            y="70"
            width="28"
            height="20"
            fill="rgba(52,211,153,0.15)"
            stroke="#34d399"
            strokeWidth="1.5"
            strokeDasharray="2 1"
          />
          <circle cx="164" cy="80" r="3" fill="#34d399" />
          <text x="150" y="64" fontSize="8" fill="#34d399" fontFamily="monospace" fontWeight="bold">
            SQ-011 (1.1t)
          </text>
        </g>

        {/* Target 2 with pulsing indicator */}
        <g>
          <rect
            x="255"
            y="110"
            width="22"
            height="16"
            fill="rgba(251,191,36,0.15)"
            stroke="#fbbf24"
            strokeWidth="1.5"
            strokeDasharray="2 1"
          />
          <circle cx="266" cy="118" r="2.5" fill="#fbbf24" />
          <text x="255" y="104" fontSize="8" fill="#fbbf24" fontFamily="monospace" fontWeight="bold">
            SQ-004 (0.9t)
          </text>
        </g>
      </svg>
    </div>
  );
}