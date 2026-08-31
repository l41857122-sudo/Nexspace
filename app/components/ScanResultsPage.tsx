"use client";

import { useState } from "react";
import {
  Bell,
  HelpCircle,
  ChevronRight,
  Search,
  Ship,
  Building2,
  BarChart3,
  FileOutput,
  FileDown,
  Download,
  Scan,
  Crosshair,
  MapPin,
  Layers,
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Navigation,
  CheckCircle2,
  Loader2,
} from "lucide-react";

import Sidebar from "./Sidebar";

// ---------------------------------------------
// Top bar
// ---------------------------------------------
function TopBar() {
  return (
    <header className="flex items-center justify-between px-4 sm:px-6 lg:px-8 py-4 border-b border-slate-800/80 bg-[#09131f]/60 backdrop-blur-md">
      <div className="flex items-center gap-1.5 text-sm">
        <span className="text-white font-bold tracking-tight">SatQuery AI</span>
        <ChevronRight size={12} className="text-slate-600" />
        <span className="text-slate-400">Queries</span>
        <ChevronRight size={12} className="text-slate-600" />
        <span className="text-cyan-400 font-mono text-xs">Q-778-DELTA</span>
      </div>
      <div className="flex items-center gap-2">
        <button
          className="p-2 rounded-lg bg-slate-900/60 border border-slate-800 text-slate-400 hover:text-cyan-300 hover:border-cyan-500/30 transition-all duration-180 hover:scale-[1.02] active:scale-[0.98] cursor-pointer shadow-sm"
          aria-label="Notifications"
        >
          <Bell size={14} />
        </button>
        <button
          className="p-2 rounded-lg bg-slate-900/60 border border-slate-800 text-slate-400 hover:text-cyan-300 hover:border-cyan-500/30 transition-all duration-180 hover:scale-[1.02] active:scale-[0.98] cursor-pointer shadow-sm"
          aria-label="Help"
        >
          <HelpCircle size={14} />
        </button>
      </div>
    </header>
  );
}

// ---------------------------------------------
// Detected Entity Definitions
// ---------------------------------------------
interface DetectedEntity {
  id: string;
  name: string;
  type: string;
  meta: string;
  confidence: string;
  statusColor: string;
  badgeBorder: string;
  badgeBg: string;
  badgeText: string;
  x: number; // percentage on map
  y: number; // percentage on map
  width: number; // in px
  height: number; // in px
  icon: typeof Ship;
}

const detectedEntities: DetectedEntity[] = [
  {
    id: "entity-1",
    name: "Vessel_Panamax_01",
    type: "Cargo/Container",
    meta: "294 × 32m",
    confidence: "94%",
    statusColor: "bg-emerald-400",
    badgeBorder: "border-emerald-500/30",
    badgeBg: "bg-emerald-500/10",
    badgeText: "text-emerald-400",
    x: 48,
    y: 38,
    width: 140,
    height: 48,
    icon: Ship,
  },
  {
    id: "entity-2",
    name: "Infra_Tank_Farm_B",
    type: "Storage/Liquid",
    meta: "14,500 m²",
    confidence: "82%",
    statusColor: "bg-amber-400",
    badgeBorder: "border-amber-500/30",
    badgeBg: "bg-amber-500/10",
    badgeText: "text-amber-300",
    x: 64,
    y: 56,
    width: 110,
    height: 60,
    icon: Building2,
  },
  {
    id: "entity-3",
    name: "Vessel_Feeder_12",
    type: "Cargo/Breakbulk",
    meta: "142 × 22m",
    confidence: "91%",
    statusColor: "bg-emerald-400",
    badgeBorder: "border-emerald-500/30",
    badgeBg: "bg-emerald-500/10",
    badgeText: "text-emerald-400",
    x: 28,
    y: 28,
    width: 120,
    height: 44,
    icon: Ship,
  },
];

// ---------------------------------------------
// Map area (Mission Control Viewport)
// ---------------------------------------------
function MapArea({
  hoveredEntityId,
  selectedEntityId,
  onHoverEntity,
  onSelectEntity,
}: {
  hoveredEntityId: string | null;
  selectedEntityId: string | null;
  onHoverEntity: (id: string | null) => void;
  onSelectEntity: (id: string | null) => void;
}) {
  const [zoom, setZoom] = useState(1);

  const labels = [
    { text: "Atlantikwall Museum", className: "top-[18%] left-[8%]" },
    { text: "Hook of Holland", className: "top-[22%] left-[34%]" },
    { text: "Europoort Rotterdam", className: "top-[62%] left-[58%]" },
    { text: "Oostvoorne", className: "top-[78%] left-[20%]" },
  ];

  // Calculate transform center based on selected entity
  const selectedEntity = detectedEntities.find((e) => e.id === selectedEntityId);
  const transformOrigin = selectedEntity
    ? `${selectedEntity.x}% ${selectedEntity.y}%`
    : "center center";

  return (
    <div className="flex-1 relative bg-[#07111c] overflow-hidden min-h-[480px] lg:min-h-0 flex flex-col justify-between p-4 select-none">
      {/* Background HUD Coordinate Grid */}
      <div
        className="absolute inset-0 opacity-15 pointer-events-none"
        style={{
          backgroundImage:
            "linear-gradient(rgba(56,189,248,0.2) 1px, transparent 1px), linear-gradient(90deg, rgba(56,189,248,0.2) 1px, transparent 1px)",
          backgroundSize: "32px 32px",
        }}
      />

      {/* Top HUD Telemetry Banner */}
      <div className="relative z-20 flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2 bg-[#091522]/90 backdrop-blur-md px-3 py-1.5 rounded-lg border border-slate-800 font-mono text-[10px] text-cyan-300 shadow-sm">
          <Scan size={13} className="text-cyan-400 animate-pulse" />
          <span className="font-semibold">SAR COMPOSITE PASS · POL: HH+HV</span>
          <span className="text-slate-600">|</span>
          <span className="text-slate-400">FPS: 60Hz REAL-TIME</span>
        </div>

        {/* Map Control Actions */}
        <div className="flex items-center gap-1 bg-[#091522]/90 backdrop-blur-md p-1 rounded-lg border border-slate-800 font-mono text-[10px] text-slate-300">
          <button
            onClick={() => setZoom((z) => Math.min(Number((z + 0.25).toFixed(2)), 2.25))}
            className="p-1 hover:bg-cyan-500/10 hover:text-cyan-300 rounded transition-colors cursor-pointer"
            title="Zoom In"
          >
            <ZoomIn size={13} />
          </button>
          <span className="text-cyan-400 px-1 font-semibold">{zoom.toFixed(2)}×</span>
          <button
            onClick={() => setZoom((z) => Math.max(Number((z - 0.25).toFixed(2)), 0.75))}
            className="p-1 hover:bg-cyan-500/10 hover:text-cyan-300 rounded transition-colors cursor-pointer"
            title="Zoom Out"
          >
            <ZoomOut size={13} />
          </button>
          <button
            onClick={() => {
              setZoom(1);
              onSelectEntity(null);
            }}
            className="p-1 hover:bg-cyan-500/10 hover:text-cyan-300 rounded text-slate-500 hover:text-slate-300 border-l border-slate-800 ml-0.5 transition-colors cursor-pointer"
            title="Reset Viewport"
          >
            <RotateCcw size={11} />
          </button>
        </div>
      </div>

      {/* Main Interactive Map Viewport Canvas */}
      <div
        className="absolute inset-0 overflow-hidden transition-all duration-300 ease-out"
        style={{
          transform: `scale(${selectedEntityId ? Math.max(zoom, 1.35) : zoom})`,
          transformOrigin: transformOrigin,
        }}
      >
        {/* Real High-Resolution Satellite Raster Imagery Background */}
        <div
          className="absolute inset-0 bg-cover bg-center opacity-85 contrast-125 brightness-95 transition-all duration-300 pointer-events-none"
          style={{
            backgroundImage:
              "url('https://images.unsplash.com/photo-1541185933-ef5d8ed016c2?q=80&w=1600&auto=format&fit=crop')",
          }}
        />
        <div className="absolute inset-0 bg-gradient-to-t from-[#07111c]/90 via-[#07111c]/30 to-[#07111c]/80 pointer-events-none" />

        {/* Synthetic Nautical Bathymetry / Coastal Contours Overlay */}
        <svg
          className="absolute inset-0 w-full h-full pointer-events-none opacity-55"
          preserveAspectRatio="none"
          viewBox="0 0 1000 600"
        >
          <defs>
            <linearGradient id="waterGrad" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#081b2e" stopOpacity="0.85" />
              <stop offset="100%" stopColor="#05101a" stopOpacity="0.95" />
            </linearGradient>
            <linearGradient id="channelGrad" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="rgba(6,182,212,0.18)" />
              <stop offset="100%" stopColor="rgba(6,182,212,0.06)" />
            </linearGradient>
          </defs>

          {/* Background water layer */}
          <rect width="1000" height="600" fill="url(#waterGrad)" />

          {/* Shipping Channel Navigation Lanes */}
          <path
            d="M 50,220 C 250,230 450,260 650,340 C 800,400 950,480 1000,520 L 1000,600 L 0,600 Z"
            fill="url(#channelGrad)"
            stroke="rgba(6,182,212,0.3)"
            strokeWidth="1.5"
            strokeDasharray="8 4"
          />

          {/* Bathymetric Depth Contours */}
          <path
            d="M 0,160 Q 300,190 550,280 T 1000,420"
            fill="none"
            stroke="rgba(56,189,248,0.22)"
            strokeWidth="1"
          />
          <path
            d="M 0,280 Q 280,310 500,380 T 1000,560"
            fill="none"
            stroke="rgba(56,189,248,0.15)"
            strokeWidth="1"
          />

          {/* Port Maasvlakte Quay & Industrial Outlines */}
          <path
            d="M 550,300 L 620,290 L 680,340 L 780,350 L 850,440 L 800,480 L 720,440 L 640,460 Z"
            fill="rgba(30,41,59,0.45)"
            stroke="rgba(148,163,184,0.35)"
            strokeWidth="1.5"
          />

          {/* Sector Boundary Polygon */}
          <polygon
            points="220,160 480,180 820,320 720,520 180,420"
            fill="rgba(6,182,212,0.05)"
            stroke="rgba(6,182,212,0.45)"
            strokeWidth="1.5"
            strokeDasharray="4 3"
          />
        </svg>

        {/* Active Scan-Line Radar Sweep Beam Animation */}
        <div className="absolute inset-0 overflow-hidden pointer-events-none z-15">
          <div
            className="w-full h-12 bg-gradient-to-b from-cyan-400/30 via-cyan-500/10 to-transparent border-b-2 border-cyan-400 animate-scan shadow-[0_0_15px_rgba(6,182,212,0.6)]"
            style={{ animationDuration: "3.5s" }}
          />
        </div>

        {/* Geographic Labels */}
        {labels.map((l) => (
          <span
            key={l.text}
            className={`absolute text-[10px] font-mono text-slate-300/80 bg-[#091522]/85 px-2 py-0.5 rounded border border-slate-800 shadow-sm z-10 ${l.className}`}
          >
            {l.text}
          </span>
        ))}

        {/* Interactive Detected Entity Markers & Bounding Boxes */}
        {detectedEntities.map((entity) => {
          const isHovered = hoveredEntityId === entity.id;
          const isSelected = selectedEntityId === entity.id;
          const isHighlighted = isHovered || isSelected;

          return (
            <div
              key={entity.id}
              onClick={() => onSelectEntity(isSelected ? null : entity.id)}
              onMouseEnter={() => onHoverEntity(entity.id)}
              onMouseLeave={() => onHoverEntity(null)}
              style={{
                top: `${entity.y}%`,
                left: `${entity.x}%`,
                transform: "translate(-50%, -50%)",
              }}
              className={`absolute cursor-pointer transition-all duration-180 z-20 ${
                isHighlighted ? "scale-110 z-30" : "hover:scale-105"
              }`}
            >
              {/* Bounding Box Container */}
              <div
                style={{ width: `${entity.width}px` }}
                className={`p-2 rounded-lg border-2 backdrop-blur-sm transition-all duration-180 ${
                  isHighlighted
                    ? `${
                        entity.statusColor === "bg-amber-400"
                          ? "border-amber-300 bg-amber-500/25 shadow-[0_0_25px_rgba(251,191,36,0.7)]"
                          : "border-cyan-300 bg-cyan-500/25 shadow-[0_0_25px_rgba(6,182,212,0.7)]"
                      }`
                    : "border-cyan-500/40 bg-slate-900/75 hover:border-cyan-400/80"
                }`}
              >
                {/* Corner reticle tick marks */}
                <div className="flex items-center justify-between gap-1 mb-1">
                  <div className="flex items-center gap-1.5">
                    <span className={`w-2 h-2 rounded-full ${entity.statusColor} animate-pulse`} />
                    <span className="text-[10px] font-mono font-bold text-white truncate max-w-[90px]">
                      {entity.name.replace("Vessel_", "").replace("Infra_", "")}
                    </span>
                  </div>
                  <span
                    className={`text-[8px] font-mono font-bold px-1 py-0.5 rounded border ${entity.badgeBg} ${entity.badgeBorder} ${entity.badgeText}`}
                  >
                    {entity.confidence}
                  </span>
                </div>

                <div className="flex items-center justify-between text-[8px] font-mono text-slate-300 pt-1 border-t border-slate-700/60">
                  <span>{entity.type.split("/")[0]}</span>
                  <span className="text-cyan-300">{entity.meta}</span>
                </div>
              </div>

              {/* Pulsing Target Point Anchor */}
              <div className="absolute -top-1 -right-1 w-2.5 h-2.5 rounded-full bg-cyan-400 shadow-[0_0_8px_rgba(6,182,212,0.9)] animate-ping" />
            </div>
          );
        })}
      </div>

      {/* Bottom Telemetry Footer Strip */}
      <div className="relative z-20 flex flex-wrap items-center justify-between gap-2 text-[10px] font-mono text-slate-400 bg-[#091522]/90 backdrop-blur-md px-3.5 py-2 rounded-lg border border-slate-800 shadow-sm mt-auto">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5 text-emerald-400">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
            <span>AI SCAN ACTIVE · EPSG:3857</span>
          </div>
          <span className="text-slate-700 hidden sm:inline">|</span>
          <span className="text-slate-400 hidden sm:inline">RESOLUTION: 0.5m GSD</span>
        </div>
        <div className="flex items-center gap-2 text-cyan-300">
          <Navigation size={12} className="text-cyan-400" />
          <span>LAT 51.9852° N · LON 4.1287° E · NORTH-ORIENTED</span>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------
// Stat block
// ---------------------------------------------
function StatBlock({
  icon: Icon,
  label,
  value,
  sub,
  color,
}: {
  icon: React.ElementType;
  label: string;
  value: string;
  sub: string;
  color: string;
}) {
  return (
    <div className="flex-1 bg-slate-900/60 border border-slate-800/80 rounded-xl p-3 shadow-sm">
      <div className="flex items-center gap-1.5 text-[10px] text-slate-400 uppercase tracking-widest font-mono">
        <Icon size={11} className={color} />
        {label}
      </div>
      <p className="text-xl font-bold text-white tracking-tight mt-1">{value}</p>
      <p className="text-[10px] font-mono text-slate-400 mt-0.5">{sub}</p>
    </div>
  );
}

// ---------------------------------------------
// Entity row
// ---------------------------------------------
function EntityRow({
  entity,
  isHovered,
  isSelected,
  onHover,
  onSelect,
}: {
  entity: DetectedEntity;
  isHovered: boolean;
  isSelected: boolean;
  onHover: (id: string | null) => void;
  onSelect: (id: string | null) => void;
}) {
  const isHighlighted = isHovered || isSelected;

  return (
    <div
      onMouseEnter={() => onHover(entity.id)}
      onMouseLeave={() => onHover(null)}
      onClick={() => onSelect(isSelected ? null : entity.id)}
      className={`flex items-center justify-between border-b border-slate-800/60 py-2.5 px-2 rounded-lg last:border-b-0 transition-all duration-150 cursor-pointer ${
        isHighlighted
          ? "bg-cyan-500/15 border-cyan-500/40 translate-x-0.5 shadow-sm"
          : "hover:bg-slate-800/50"
      }`}
    >
      <div className="flex items-start gap-2.5 min-w-0">
        <span className={`w-1.5 h-1.5 rounded-full mt-1.5 shrink-0 ${entity.statusColor}`} />
        <div className="min-w-0">
          <p
            className={`text-xs font-mono font-medium truncate transition-colors ${
              isHighlighted ? "text-cyan-300 font-bold" : "text-slate-200"
            }`}
          >
            {entity.name}
          </p>
          <p className="text-[10px] text-slate-400 font-mono mt-0.5">
            {entity.type} · {entity.meta}
          </p>
        </div>
      </div>

      <div className="flex items-center gap-1.5 shrink-0 ml-2">
        <span
          className={`text-[10px] font-mono font-semibold px-2 py-0.5 rounded border ${entity.badgeBg} ${entity.badgeBorder} ${entity.badgeText}`}
        >
          {entity.confidence} CONF
        </span>
      </div>
    </div>
  );
}

// ---------------------------------------------
// Scan Results side panel
// ---------------------------------------------
function ScanResultsPanel({
  hoveredEntityId,
  selectedEntityId,
  onHoverEntity,
  onSelectEntity,
}: {
  hoveredEntityId: string | null;
  selectedEntityId: string | null;
  onHoverEntity: (id: string | null) => void;
  onSelectEntity: (id: string | null) => void;
}) {
  const [downloading, setDownloading] = useState(false);
  const [downloadSuccess, setDownloadSuccess] = useState(false);

  const handleGeneratePDF = async () => {
    if (downloading) return;
    setDownloading(true);
    try {
      window.open("/api/reports/SQ-REP-2023-11A/pdf", "_blank");
      setDownloadSuccess(true);
      setTimeout(() => setDownloadSuccess(false), 3000);
    } catch (e) {
      console.error(e);
    } finally {
      setDownloading(false);
    }
  };

  return (
    <div className="w-full lg:w-80 shrink-0 bg-[#0c1624]/90 backdrop-blur-md border-t lg:border-t-0 lg:border-l border-slate-800/80 flex flex-col justify-between">
      <div>
        <div className="flex items-start justify-between p-4 border-b border-slate-800/80">
          <div>
            <div className="flex items-center gap-1.5">
              <Search size={14} className="text-cyan-400" />
              <h2 className="text-sm font-semibold text-white tracking-tight">Scan Results</h2>
            </div>
            <p className="text-[10px] text-slate-400 font-mono mt-0.5">
              Area: Maasvlakte Industrial Zone
            </p>
          </div>
          <span className="text-[10px] font-mono text-cyan-400 bg-cyan-500/10 px-2 py-0.5 rounded border border-cyan-500/20">
            3 Targets
          </span>
        </div>

        <div className="p-4 space-y-4 overflow-y-auto">
          {/* Top KPI Metrics */}
          <div className="flex gap-3">
            <StatBlock
              icon={Ship}
              label="Cargo Ships"
              value="12"
              sub="+2 from last scan"
              color="text-cyan-400"
            />
            <StatBlock
              icon={Building2}
              label="Infra"
              value="4"
              sub="New construction"
              color="text-amber-400"
            />
          </div>

          {/* Confidence Summary */}
          <div className="bg-slate-900/60 border border-slate-800/80 rounded-xl p-3 shadow-sm">
            <div className="flex items-center justify-between mb-1">
              <div className="flex items-center gap-1.5 text-[10px] text-slate-400 uppercase tracking-widest font-mono">
                <BarChart3 size={11} className="text-cyan-400" />
                <span>Confidence Distribution</span>
              </div>
              <span className="text-[10px] font-mono text-emerald-400">High Accuracy</span>
            </div>
            <p className="text-lg font-bold text-white tracking-tight">
              88%{" "}
              <span className="text-xs font-normal font-mono text-slate-400">
                · Mean Precision
              </span>
            </p>
          </div>

          {/* Detected Entities List */}
          <div>
            <p className="text-[10px] text-slate-400 uppercase tracking-widest font-mono mb-2">
              Detected Entities (Click to Focus)
            </p>
            <div className="bg-slate-900/40 border border-slate-800/80 rounded-xl p-2 space-y-0.5">
              {detectedEntities.map((entity) => (
                <EntityRow
                  key={entity.id}
                  entity={entity}
                  isHovered={hoveredEntityId === entity.id}
                  isSelected={selectedEntityId === entity.id}
                  onHover={onHoverEntity}
                  onSelect={onSelectEntity}
                />
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Report download & Export Actions */}
      <div className="p-4 border-t border-slate-800/80 space-y-2">
        <button className="w-full flex items-center justify-center gap-2 bg-slate-900/60 hover:bg-slate-800/80 border border-slate-800 hover:border-cyan-500/30 transition-all duration-180 hover:scale-[1.01] active:scale-[0.98] text-xs text-slate-300 hover:text-cyan-300 rounded-lg py-2 font-mono cursor-pointer shadow-sm">
          <FileOutput size={13} />
          <span>Export GeoJSON</span>
        </button>
        <button className="w-full flex items-center justify-center gap-2 bg-slate-900/60 hover:bg-slate-800/80 border border-slate-800 hover:border-cyan-500/30 transition-all duration-180 hover:scale-[1.01] active:scale-[0.98] text-xs text-slate-300 hover:text-cyan-300 rounded-lg py-2 font-mono cursor-pointer shadow-sm">
          <FileDown size={13} />
          <span>Download CSV</span>
        </button>
        <button
          onClick={handleGeneratePDF}
          disabled={downloading}
          className="w-full flex items-center justify-center gap-2 bg-gradient-to-r from-cyan-500 to-sky-400 hover:from-cyan-400 hover:to-sky-300 disabled:opacity-75 transition-all duration-180 hover:scale-[1.02] active:scale-[0.98] text-[#071320] text-xs font-bold rounded-lg py-2.5 shadow-[0_0_12px_rgba(6,182,212,0.35)] hover:shadow-[0_0_18px_rgba(6,182,212,0.55)] cursor-pointer"
        >
          {downloading ? (
            <>
              <Loader2 size={13} className="animate-spin text-[#071320]" />
              <span>Generating Report...</span>
            </>
          ) : downloadSuccess ? (
            <>
              <CheckCircle2 size={13} className="text-[#071320]" />
              <span>PDF Generated!</span>
            </>
          ) : (
            <>
              <Download size={13} />
              <span>Generate PDF Report</span>
            </>
          )}
        </button>
      </div>
    </div>
  );
}

// ---------------------------------------------
// Scan Results Page Root
// ---------------------------------------------
export default function ScanResultsPage() {
  const [hoveredEntityId, setHoveredEntityId] = useState<string | null>(null);
  const [selectedEntityId, setSelectedEntityId] = useState<string | null>(null);

  return (
    <div className="flex flex-col lg:flex-row min-h-screen bg-[#0a1420] text-white font-sans overflow-x-hidden">
      <Sidebar />

      <div className="flex-1 flex flex-col min-w-0">
        <TopBar />
        <div className="flex-1 flex flex-col lg:flex-row min-w-0 overflow-y-auto">
          <MapArea
            hoveredEntityId={hoveredEntityId}
            selectedEntityId={selectedEntityId}
            onHoverEntity={setHoveredEntityId}
            onSelectEntity={setSelectedEntityId}
          />
          <ScanResultsPanel
            hoveredEntityId={hoveredEntityId}
            selectedEntityId={selectedEntityId}
            onHoverEntity={setHoveredEntityId}
            onSelectEntity={setSelectedEntityId}
          />
        </div>
      </div>
    </div>
  );
}