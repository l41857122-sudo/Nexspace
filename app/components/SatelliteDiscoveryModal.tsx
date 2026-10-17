"use client";

import React, { useState, useEffect } from "react";
import {
  Search,
  Sparkles,
  MapPin,
  Calendar,
  Cloud,
  Layers,
  ChevronRight,
  Loader2,
  AlertTriangle,
  X,
  ExternalLink,
  ShieldCheck,
  CheckCircle2,
  Info,
  Compass,
  ArrowRight,
  Globe,
} from "lucide-react";
import GoogleMapView from "./GoogleMapView";
import type {
  DiscoveredSatelliteScene,
  DiscoveryPlan,
  ImagerySearchResponse,
  ImageryRetrieveResponse,
  NexSpaceQueryResponse,
} from "../types/nexspace";

interface SatelliteDiscoveryModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSceneSelected: (
    scene: DiscoveredSatelliteScene,
    retrievedImageBase64: string,
    investigationResult?: NexSpaceQueryResponse | null
  ) => void;
  initialQuery?: string;
}

export default function SatelliteDiscoveryModal({
  isOpen,
  onClose,
  onSceneSelected,
  initialQuery = "",
}: SatelliteDiscoveryModalProps) {
  const [nlQuery, setNlQuery] = useState(initialQuery);
  const [prevInitialQuery, setPrevInitialQuery] = useState(initialQuery);
  if (initialQuery !== prevInitialQuery) {
    setPrevInitialQuery(initialQuery);
    setNlQuery(initialQuery);
  }

  // Location state is completely dynamic with zero hardcoding
  const [locationInput, setLocationInput] = useState("");
  const [isLocationManuallyEdited, setIsLocationManuallyEdited] = useState(false);
  const [selectedMapScene, setSelectedMapScene] = useState<DiscoveredSatelliteScene | null>(null);
  const [dateRange, setDateRange] = useState("latest");
  const [cloudThreshold, setCloudThreshold] = useState(20.0);
  const [sensorSelection, setSensorSelection] = useState("Sentinel-2");
  const [resolutionSelection, setResolutionSelection] = useState("10m");

  const [isSearching, setIsSearching] = useState(false);
  const [searchError, setSearchError] = useState<{
    status: string;
    reason: string;
    actions?: string[];
  } | null>(null);

  const [discoveredScenes, setDiscoveredScenes] = useState<DiscoveredSatelliteScene[]>([]);
  const [discoveryPlan, setDiscoveryPlan] = useState<DiscoveryPlan | null>(null);
  const [selectedPreviewScene, setSelectedPreviewScene] = useState<DiscoveredSatelliteScene | null>(null);
  const [isRetrieving, setIsRetrieving] = useState(false);
  const [retrievalStage, setRetrievalStage] = useState<string>("");

  if (!isOpen) return null;

  // --------------------------------------------------------------------------
  // Execute Search (Strict Investigation Isolation - Requirement 17)
  // --------------------------------------------------------------------------
  const handleSearch = async () => {
    setIsSearching(true);
    setSearchError(null);
    setDiscoveredScenes([]);
    setDiscoveryPlan(null);
    setSelectedMapScene(null);

    try {
      const payload: Record<string, unknown> = {
        max_cloud_cover: cloudThreshold,
        sensor: sensorSelection,
        collection: "sentinel-2-c1-l2a",
        limit: 10,
      };

      if (nlQuery.trim()) {
        payload.query = nlQuery.trim();
      }
      // Only include structured location if manually entered by user or if no NL query was given
      if (locationInput.trim() && (isLocationManuallyEdited || !nlQuery.trim())) {
        payload.location = locationInput.trim();
      }
      if (dateRange === "last_month") {
        const now = new Date();
        const start = new Date(now.getTime() - 60 * 86400000).toISOString();
        const end = new Date(now.getTime() - 15 * 86400000).toISOString();
        payload.datetime_range = `${start}/${end}`;
      } else if (dateRange === "last_week") {
        const now = new Date();
        const start = new Date(now.getTime() - 7 * 86400000).toISOString();
        const end = now.toISOString();
        payload.datetime_range = `${start}/${end}`;
      }

      const res = await fetch("/api/imagery/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const data: ImagerySearchResponse = await res.json();

      if (!res.ok || data.status === "SATELLITE_DATA_UNAVAILABLE") {
        setSearchError({
          status: data.status || "SATELLITE_DATA_UNAVAILABLE",
          reason: data.reason || "Copernicus catalog request failed.",
          actions: data.actions || ["Retry", "Upload imagery instead"],
        });
        return;
      }

      if (data.status === "location_unresolved") {
        setSearchError({
          status: "LOCATION_UNRESOLVED",
          reason: data.message || "Geographic location could not be identified or geocoded. Please enter a valid city or coordinate pair.",
          actions: ["Check spelling", "Enter coordinates (e.g. 26.8467, 80.9462)"],
        });
        return;
      }

      if (data.plan) {
        setDiscoveryPlan(data.plan);
        if (data.plan.location_raw) {
          setLocationInput(data.plan.location_raw);
          setIsLocationManuallyEdited(false);
        }
      }

      const scenesList = data.scenes || [];
      setDiscoveredScenes(scenesList);
      if (scenesList.length > 0) {
        setSelectedMapScene(scenesList[0]);
      }
    } catch (err: unknown) {
      const errMsg = err instanceof Error ? err.message : "Network error";
      setSearchError({
        status: "SATELLITE_DATA_UNAVAILABLE",
        reason: "Failed to communicate with satellite catalog: " + errMsg,
        actions: ["Retry", "Upload imagery instead"],
      });
    } finally {
      setIsSearching(false);
    }
  };

  // --------------------------------------------------------------------------
  // Execute Retrieval & Downstream Analysis
  // --------------------------------------------------------------------------
  const handleAnalyzeScene = async (scene: DiscoveredSatelliteScene) => {
    setIsRetrieving(true);
    setRetrievalStage("Retrieving satellite scene raster (preserving CRS & affine metadata)...");

    try {
      const payload = {
        scene_id: scene.scene_id,
        prefer_preview: true, // bandwidth & latency conservation
        query: nlQuery || `Detect targets in ${scene.platform} scene`,
        auto_investigate: true,
      };

      setRetrievalStage("Running NexSpace agentic investigation pipeline...");
      const res = await fetch("/api/imagery/retrieve", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const data: ImageryRetrieveResponse = await res.json();

      if (!res.ok || data.status === "SATELLITE_DATA_UNAVAILABLE") {
        setSearchError({
          status: data.status || "SATELLITE_DATA_UNAVAILABLE",
          reason: data.reason || "Scene retrieval failed.",
          actions: data.actions || ["Retry", "Upload imagery instead"],
        });
        return;
      }

      // Notify parent component and close modal
      onSceneSelected(scene, data.scene.image_base64, data.investigation);
      onClose();
    } catch (err: unknown) {
      const errMsg = err instanceof Error ? err.message : "Unknown error";
      setSearchError({
        status: "RETRIEVAL_FAILED",
        reason: "Scene retrieval failed: " + errMsg,
        actions: ["Retry", "Upload imagery instead"],
      });
    } finally {
      setIsRetrieving(false);
      setRetrievalStage("");
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-md overflow-y-auto">
      <div className="relative w-full max-w-5xl bg-[#0a1424] border border-slate-700/80 rounded-2xl shadow-2xl overflow-hidden flex flex-col max-h-[90vh]">
        {/* Modal Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800 bg-[#0c182b]">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-cyan-500/10 border border-cyan-500/30 text-cyan-400">
              <Compass className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-lg font-bold text-white tracking-wide flex items-center gap-2">
                <span>FIND SATELLITE IMAGERY</span>
                <span className="text-xs px-2 py-0.5 rounded bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 font-mono">
                  Sentinel-2 L2A BOA
                </span>
              </h2>
              <p className="text-xs text-slate-400">
                Direct Earth-observation catalog search with CRS & affine preservation.
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Modal Content */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {/* Natural Language Prompt Search Bar */}
          <div className="bg-[#0e1c31] border border-slate-700/60 rounded-xl p-4 space-y-3">
            <div className="flex items-center justify-between text-xs text-slate-300 font-semibold">
              <span className="flex items-center gap-1.5 text-cyan-400">
                <Sparkles className="w-4 h-4" />
                Natural Language Investigation Prompt
              </span>
              <span className="text-slate-400 text-[11px]">
                Semantic extraction separates Location, Time, Sensor, and Target
              </span>
            </div>
            <div className="flex items-center gap-2">
              <div className="relative flex-1">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
                <input
                  type="text"
                  value={nlQuery}
                  onChange={(e) => {
                    setNlQuery(e.target.value);
                    if (!isLocationManuallyEdited) {
                      setLocationInput("");
                    }
                  }}
                  onKeyDown={(e) => e.key === "Enter" && handleSearch()}
                  placeholder='e.g. "Find recent satellite imagery of London and locate buildings"'
                  className="w-full bg-[#08111e] border border-slate-700 rounded-lg pl-9 pr-4 py-2.5 text-sm text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500"
                />
              </div>
              <button
                onClick={handleSearch}
                disabled={isSearching || isRetrieving}
                className="flex items-center gap-2 px-5 py-2.5 rounded-lg bg-gradient-to-r from-cyan-500 to-blue-600 text-slate-950 font-bold text-sm hover:from-cyan-400 hover:to-blue-500 transition-all shadow-lg disabled:opacity-50"
              >
                {isSearching ? <Loader2 className="w-4 h-4 animate-spin" /> : <Search className="w-4 h-4" />}
                <span>SEARCH IMAGERY</span>
              </button>
            </div>
          </div>

          {/* Structured Controls Accordion / Secondary Inputs */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3 bg-[#0c182b] p-4 rounded-xl border border-slate-800 text-xs">
            <div>
              <label className="text-slate-400 font-medium block mb-1.5 flex items-center gap-1">
                <MapPin className="w-3.5 h-3.5 text-cyan-400" />
                Location / Coordinates
              </label>
              <input
                type="text"
                value={locationInput}
                onChange={(e) => {
                  setLocationInput(e.target.value);
                  setIsLocationManuallyEdited(true);
                }}
                placeholder="e.g. London, Tokyo, 40.7128, -74.0060"
                className="w-full bg-[#070f1a] border border-slate-700 rounded px-2.5 py-1.5 text-slate-200 focus:outline-none focus:border-cyan-500"
              />
            </div>

            <div>
              <label className="text-slate-400 font-medium block mb-1.5 flex items-center gap-1">
                <Calendar className="w-3.5 h-3.5 text-cyan-400" />
                Acquisition Date
              </label>
              <select
                value={dateRange}
                onChange={(e) => setDateRange(e.target.value)}
                className="w-full bg-[#070f1a] border border-slate-700 rounded px-2.5 py-1.5 text-slate-200 focus:outline-none focus:border-cyan-500"
              >
                <option value="latest">Latest available</option>
                <option value="last_month">Last month</option>
                <option value="last_week">Past 7 days</option>
              </select>
            </div>

            <div>
              <label className="text-slate-400 font-medium block mb-1.5 flex items-center gap-1">
                <Cloud className="w-3.5 h-3.5 text-cyan-400" />
                Cloud Cover
              </label>
              <select
                value={cloudThreshold}
                onChange={(e) => setCloudThreshold(Number(e.target.value))}
                className="w-full bg-[#070f1a] border border-slate-700 rounded px-2.5 py-1.5 text-slate-200 focus:outline-none focus:border-cyan-500"
              >
                <option value={5}>&lt; 5% (Clear sky)</option>
                <option value={10}>&lt; 10% (Low cloud)</option>
                <option value={20}>&lt; 20% (Standard)</option>
                <option value={40}>&lt; 40% (Permissive)</option>
              </select>
            </div>

            <div>
              <label className="text-slate-400 font-medium block mb-1.5 flex items-center gap-1">
                <Layers className="w-3.5 h-3.5 text-cyan-400" />
                Sensor
              </label>
              <select
                value={sensorSelection}
                onChange={(e) => setSensorSelection(e.target.value)}
                className="w-full bg-[#070f1a] border border-slate-700 rounded px-2.5 py-1.5 text-slate-200 focus:outline-none focus:border-cyan-500"
              >
                <option value="Sentinel-2">Sentinel-2 Optical (10m)</option>
                <option value="Sentinel-1">Sentinel-1 SAR Radar</option>
              </select>
            </div>

            <div>
              <label className="text-slate-400 font-medium block mb-1.5 flex items-center gap-1">
                <ShieldCheck className="w-3.5 h-3.5 text-cyan-400" />
                Resolution
              </label>
              <select
                value={resolutionSelection}
                onChange={(e) => setResolutionSelection(e.target.value)}
                className="w-full bg-[#070f1a] border border-slate-700 rounded px-2.5 py-1.5 text-slate-200 focus:outline-none focus:border-cyan-500"
              >
                <option value="10m">Best available (10m)</option>
                <option value="20m">Medium (20m)</option>
              </select>
            </div>
          </div>

          {/* Loading Indicator */}
          {isSearching && (
            <div className="flex flex-col items-center justify-center py-12 space-y-3">
              <Loader2 className="w-8 h-8 text-cyan-400 animate-spin" />
              <p className="text-sm text-slate-300 font-medium">
                Querying Earth-observation STAC catalog for Sentinel-2 scenes...
              </p>
              <p className="text-xs text-slate-500">
                Filtering by geographic AOI, temporal window, and cloud cover.
              </p>
            </div>
          )}

          {/* Retrieval Overlay */}
          {isRetrieving && (
            <div className="flex flex-col items-center justify-center py-12 space-y-3 bg-cyan-950/20 border border-cyan-800/40 rounded-xl">
              <Loader2 className="w-8 h-8 text-cyan-400 animate-spin" />
              <p className="text-sm text-cyan-300 font-bold">{retrievalStage}</p>
              <p className="text-xs text-slate-400">
                Preserving CRS, affine geotransform, GSD, and multispectral bands.
              </p>
            </div>
          )}

          {/* Error Banner: SATELLITE DATA UNAVAILABLE */}
          {searchError && (
            <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-300 space-y-3">
              <div className="flex items-center gap-2 font-bold text-sm">
                <AlertTriangle className="w-5 h-5 text-rose-400 flex-shrink-0" />
                <span>{searchError.status}</span>
              </div>
              <p className="text-xs text-slate-300">{searchError.reason}</p>
              {searchError.actions && (
                <div className="flex items-center gap-2 pt-1">
                  <button
                    onClick={handleSearch}
                    className="px-3 py-1 rounded bg-rose-500/20 hover:bg-rose-500/30 text-xs font-semibold text-rose-200 border border-rose-500/40"
                  >
                    Retry
                  </button>
                  <button
                    onClick={onClose}
                    className="px-3 py-1 rounded bg-slate-800 hover:bg-slate-700 text-xs text-slate-300 border border-slate-700"
                  >
                    Upload imagery instead
                  </button>
                </div>
              )}
            </div>
          )}

          {/* Search Results Grid & Geographic Context */}
          {!isSearching && (discoveredScenes.length > 0 || discoveryPlan?.geocoding) && (
            <div className="space-y-4">
              <div className="flex items-center justify-between text-xs text-slate-400">
                <span>
                  {discoveredScenes.length > 0 ? (
                    <>Found <strong className="text-white">{discoveredScenes.length}</strong> verified Sentinel-2 scenes</>
                  ) : (
                    <>Target AOI resolved (<strong className="text-cyan-400">0</strong> scenes under current cloud filter)</>
                  )}
                </span>
                {discoveryPlan?.geocoding && (
                  <span className="text-cyan-400 font-mono text-[11px]">
                    AOI: [{(() => {
                      const b: any = discoveryPlan.geocoding.bbox || (discoveryPlan.geocoding as any).bbox_list;
                      if (!b) return "";
                      if (Array.isArray(b)) return b.map((v: any) => Number(v).toFixed(3)).join(", ");
                      if (typeof b === "object") return `${Number(b.west).toFixed(3)}, ${Number(b.south).toFixed(3)}, ${Number(b.east).toFixed(3)}, ${Number(b.north).toFixed(3)}`;
                      return String(b);
                    })()}]
                  </span>
                )}
              </div>

              {/* Dedicated GEOGRAPHIC CONTEXT Section (Requirement 21) */}
              <div className="bg-[#0b1526] border border-cyan-500/40 rounded-xl p-4 space-y-3 shadow-2xl">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Globe className="w-4 h-4 text-cyan-400" />
                    <span className="font-bold text-white text-xs tracking-wider uppercase">
                      Geographic Context & Satellite Scene Footprint
                    </span>
                    {selectedMapScene && (
                      <span className="text-[10px] font-mono text-cyan-300 bg-cyan-950/80 px-2 py-0.5 rounded border border-cyan-800">
                        🛰 {selectedMapScene.platform} ({selectedMapScene.acquisition_date.substring(0, 10)})
                      </span>
                    )}
                  </div>
                  {discoveryPlan?.geocoding && (
                    <span className="text-[11px] font-mono text-cyan-400">
                      AOI: [{(() => {
                        const b: any = discoveryPlan.geocoding.bbox || (discoveryPlan.geocoding as any).bbox_list;
                        if (!b) return "";
                        if (Array.isArray(b)) return b.map((v: any) => Number(v).toFixed(3)).join(", ");
                        if (typeof b === "object") return `${Number(b.west).toFixed(3)}, ${Number(b.south).toFixed(3)}, ${Number(b.east).toFixed(3)}, ${Number(b.north).toFixed(3)}`;
                        return String(b);
                      })()}]
                    </span>
                  )}
                </div>

                <div className="rounded-lg overflow-hidden border border-slate-800">
                  <GoogleMapView
                    aoiBbox={discoveryPlan?.geocoding?.bbox}
                    scene={selectedMapScene || (discoveredScenes.length > 0 ? discoveredScenes[0] : undefined)}
                    location={discoveryPlan?.geocoding as any}
                    height="280px"
                  />
                </div>
              </div>

              {discoveredScenes.length === 0 && (
                <div className="p-3 bg-slate-900/60 border border-slate-800 rounded-xl text-center text-xs text-slate-400">
                  <span>No scenes in catalog met the strict cloud cover filter for this timeframe. The geographic AOI footprint is mapped above.</span>
                </div>
              )}

              {discoveredScenes.length > 0 && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {discoveredScenes.map((scene) => {
                  const acqDate = scene.acquisition_date.substring(0, 10);
                  const isTopPick = scene.ranking_score > 0.7;

                  return (
                    <div
                      key={scene.scene_id}
                      className="flex flex-col bg-[#0b1526] border border-slate-800 hover:border-cyan-500/60 rounded-xl overflow-hidden shadow-lg transition-all"
                    >
                      {/* Thumbnail & Badges */}
                      <div className="relative h-44 bg-slate-950 flex items-center justify-center overflow-hidden">
                        {scene.thumbnail_url ? (
                          <img
                            src={scene.thumbnail_url}
                            alt={scene.scene_id}
                            className="w-full h-full object-cover"
                            loading="lazy"
                          />
                        ) : (
                          <div className="flex flex-col items-center text-slate-600 text-xs space-y-1">
                            <Compass className="w-6 h-6" />
                            <span>Footprint Available</span>
                          </div>
                        )}

                        {/* Top Badges */}
                        <div className="absolute top-2 left-2 flex items-center gap-1.5">
                          <span className="px-2 py-0.5 rounded bg-slate-900/90 backdrop-blur-md border border-slate-700 text-cyan-300 font-mono text-[11px] font-semibold">
                            🛰 {scene.platform}
                          </span>
                          {isTopPick && (
                            <span className="px-2 py-0.5 rounded bg-emerald-500/90 text-slate-950 font-bold text-[10px]">
                              TOP MATCH
                            </span>
                          )}
                        </div>

                        <div className="absolute top-2 right-2">
                          <span className="px-2 py-0.5 rounded bg-slate-900/90 backdrop-blur-md border border-slate-700 text-amber-300 text-[11px] font-mono">
                            ☁ {scene.cloud_cover.toFixed(1)}% cloud
                          </span>
                        </div>

                        <div className="absolute bottom-2 left-2">
                          <span className="px-2 py-0.5 rounded bg-slate-900/90 backdrop-blur-md text-slate-300 text-[11px]">
                            {acqDate} ({scene.sensing_time})
                          </span>
                        </div>
                      </div>

                      {/* Card Body */}
                      <div className="p-4 flex-1 flex flex-col justify-between space-y-3 text-xs">
                        <div className="space-y-2">
                          <div className="flex items-center justify-between text-slate-400">
                            <span>Processing: {scene.processing_level.split("(")[0]}</span>
                            <span className="text-cyan-400 font-mono">
                              Res: {scene.spatial_resolution_m}m
                            </span>
                          </div>

                          <div className="p-2 rounded bg-slate-950/60 border border-slate-800/80 text-[11px] text-slate-300 leading-relaxed">
                            <strong className="text-cyan-400 block mb-0.5">Selection Rationale:</strong>
                            {scene.selected_reason}
                          </div>

                          <div className="flex flex-wrap gap-1 pt-1">
                            {scene.available_bands.slice(0, 4).map((b, i) => (
                              <span
                                key={i}
                                className="px-1.5 py-0.5 rounded bg-slate-900 border border-slate-800 text-slate-400 text-[10px]"
                              >
                                {b.split(" ")[0]}
                              </span>
                            ))}
                          </div>
                        </div>

                        {/* Action Buttons: PREVIEW SCENE | MAP | ANALYZE THIS SCENE */}
                        <div className="flex items-center gap-2 pt-2 border-t border-slate-800/80">
                          <button
                            onClick={() => setSelectedPreviewScene(scene)}
                            className="py-2 px-2.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 font-semibold text-xs border border-slate-700 transition-colors"
                          >
                            PREVIEW
                          </button>
                          <button
                            onClick={() => setSelectedMapScene(scene)}
                            className={`py-2 px-3 rounded-lg text-xs font-semibold flex items-center justify-center gap-1 transition-all ${
                              selectedMapScene?.scene_id === scene.scene_id
                                ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/60"
                                : "bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700"
                            }`}
                            title="Focus scene footprint on Google Map"
                          >
                            <MapPin className="w-3.5 h-3.5" />
                            <span>MAP</span>
                          </button>
                          <button
                            onClick={() => handleAnalyzeScene(scene)}
                            disabled={isRetrieving}
                            className="flex-1 py-2 rounded-lg bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold text-xs flex items-center justify-center gap-1.5 transition-all shadow-md"
                          >
                            <span>ANALYZE THIS SCENE</span>
                            <ArrowRight className="w-3.5 h-3.5" />
                          </button>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
            </div>
          )}
        </div>

        {/* Modal Footer */}
        <div className="flex items-center justify-between px-6 py-3 border-t border-slate-800 bg-[#0c182b] text-xs text-slate-400">
          <div className="flex items-center gap-2">
            <ShieldCheck className="w-4 h-4 text-emerald-400" />
            <span>Forensic Georeferencing: EPSG:4326 WGS84 GeoJSON generation verified</span>
          </div>
          <button
            onClick={onClose}
            className="px-4 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 font-medium"
          >
            Close
          </button>
        </div>
      </div>

      {/* Detail Preview Sub-Modal */}
      {selectedPreviewScene && (
        <div className="fixed inset-0 z-60 flex items-center justify-center p-4 bg-slate-950/90 backdrop-blur-md">
          <div className="relative w-full max-w-2xl bg-[#0b1526] border border-cyan-500/40 rounded-2xl p-6 shadow-2xl space-y-4">
            <div className="flex items-center justify-between pb-3 border-b border-slate-800">
              <h3 className="font-bold text-white text-base flex items-center gap-2">
                <span>🛰 {selectedPreviewScene.platform} Scene Preview</span>
                <span className="text-xs px-2 py-0.5 rounded bg-cyan-500/20 text-cyan-300 font-mono">
                  {selectedPreviewScene.acquisition_date.substring(0, 10)}
                </span>
              </h3>
              <button
                onClick={() => setSelectedPreviewScene(null)}
                className="p-1 rounded text-slate-400 hover:text-white"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {selectedPreviewScene.thumbnail_url && (
              <div className="h-64 rounded-xl overflow-hidden bg-black border border-slate-800 flex items-center justify-center">
                <img
                  src={selectedPreviewScene.thumbnail_url}
                  alt={selectedPreviewScene.scene_id}
                  className="w-full h-full object-contain"
                />
              </div>
            )}

            <div className="grid grid-cols-2 gap-3 text-xs bg-slate-950 p-3 rounded-lg border border-slate-800">
              <div>
                <span className="text-slate-400">Scene ID:</span>
                <p className="font-mono text-slate-200 truncate">{selectedPreviewScene.scene_id}</p>
              </div>
              <div>
                <span className="text-slate-400">Cloud Coverage:</span>
                <p className="text-amber-400 font-bold">{selectedPreviewScene.cloud_cover.toFixed(1)}%</p>
              </div>
              <div>
                <span className="text-slate-400">Native Resolution:</span>
                <p className="text-cyan-300">{selectedPreviewScene.spatial_resolution_m} meters GSD</p>
              </div>
              <div>
                <span className="text-slate-400">CRS:</span>
                <p className="text-emerald-400 font-mono">{selectedPreviewScene.crs}</p>
              </div>
            </div>

            <div className="flex items-center justify-end gap-2 pt-2">
              <button
                onClick={() => setSelectedPreviewScene(null)}
                className="px-4 py-2 rounded-lg bg-slate-800 text-slate-300 text-xs font-semibold"
              >
                Back to results
              </button>
              <button
                onClick={() => {
                  setSelectedPreviewScene(null);
                  handleAnalyzeScene(selectedPreviewScene);
                }}
                className="px-5 py-2 rounded-lg bg-cyan-500 hover:bg-cyan-400 text-slate-950 text-xs font-bold shadow-lg"
              >
                Analyze this scene →
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
