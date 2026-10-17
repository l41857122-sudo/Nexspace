"use client";

import React, { useEffect, useRef, useState, useCallback, useMemo } from "react";
import {
  MapPin,
  ExternalLink,
  Layers,
  ZoomIn,
  ZoomOut,
  Maximize2,
  Compass,
  AlertCircle,
  ShieldCheck,
  Globe,
  Info,
  CheckCircle2,
  Navigation,
  Eye,
  Crosshair,
} from "lucide-react";
import type {
  GeoMetadata,
  GeoJSONFeatureCollection,
  GeoJSONFeature,
  EvidenceNode,
  DiscoveredSatelliteScene,
  ResolvedLocation,
  MapState,
  MapStatus,
} from "../types/nexspace";

export interface GoogleMapViewProps {
  geoMeta?: GeoMetadata | null;
  geojson?: GeoJSONFeatureCollection | null;
  evidence?: EvidenceNode[];
  selectedDetectionId?: string | null;
  onSelectDetection?: (id: string) => void;
  aoiBbox?: number[] | null; // [min_lon, min_lat, max_lon, max_lat] in EPSG:4326
  location?: ResolvedLocation | null;
  scene?: DiscoveredSatelliteScene | null;
  overlayUrl?: string | null;
  overlayBounds?: { north: number; south: number; east: number; west: number } | null;
  osmGeojson?: GeoJSONFeatureCollection | null;
  height?: string;
  className?: string;
}

declare global {
  interface Window {
    google?: any;
    __googleMapsLoadingPromise?: Promise<void>;
  }
}

/**
 * Derives the optimal initial zoom level dynamically based on geographic extent
 */
function calculateDynamicZoom(bbox?: number[] | null, source?: string): number {
  if (source === "EXPLICIT_COORDINATES") return 15;
  if (!bbox || bbox.length < 4) return 12;

  const lonDelta = Math.abs(bbox[2] - bbox[0]);
  const latDelta = Math.abs(bbox[3] - bbox[1]);
  const maxDelta = Math.max(lonDelta, latDelta);

  if (maxDelta > 15.0) return 4;   // Continent / Multi-country
  if (maxDelta > 5.0) return 6;    // Country / Large region
  if (maxDelta > 1.5) return 8;    // State / Sub-region
  if (maxDelta > 0.4) return 11;   // Metropolitan area
  if (maxDelta > 0.1) return 13;   // City / Municipality
  if (maxDelta > 0.02) return 15;  // Neighborhood / Local AOI
  return 16;                       // Street / Target facility
}

export default function GoogleMapView({
  geoMeta,
  geojson,
  evidence = [],
  selectedDetectionId,
  onSelectDetection,
  aoiBbox,
  location,
  scene,
  overlayUrl,
  overlayBounds,
  osmGeojson,
  height = "520px",
  className = "",
}: GoogleMapViewProps) {
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<any>(null);
  const markersRef = useRef<Map<string, any>>(new Map());
  const polygonsRef = useRef<Map<string, any>>(new Map());
  const polylinesRef = useRef<Map<string, any>>(new Map());
  const osmMarkersRef = useRef<Map<string, any>>(new Map());
  const osmPolygonsRef = useRef<Map<string, any>>(new Map());
  const osmPolylinesRef = useRef<Map<string, any>>(new Map());
  const groundOverlayRef = useRef<any>(null);
  const infoWindowRef = useRef<any>(null);

  const [mapsLoaded, setMapsLoaded] = useState<boolean>(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [mapType, setMapType] = useState<"hybrid" | "satellite" | "roadmap">("hybrid");
  const [activeFeature, setActiveFeature] = useState<GeoJSONFeature | null>(null);
  const [showOsmLayer, setShowOsmLayer] = useState<boolean>(true);

  const apiKey = process.env.NEXT_PUBLIC_GOOGLE_MAPS_API_KEY || "";

  // --------------------------------------------------------------------------
  // 1. Resolve Dynamic Bounding Extent & Center (Strictly No Hardcoded Fallback)
  // --------------------------------------------------------------------------
  const effectiveBbox: number[] | null = useMemo(() => {
    // Priority: scene bounds_world -> scene.bbox -> aoiBbox -> location.bbox_list
    if (
      geoMeta?.bounds_world &&
      typeof geoMeta.bounds_world.min_x === "number" &&
      typeof geoMeta.bounds_world.min_y === "number"
    ) {
      return [
        geoMeta.bounds_world.min_x,
        geoMeta.bounds_world.min_y,
        geoMeta.bounds_world.max_x,
        geoMeta.bounds_world.max_y,
      ];
    }
    if (scene?.bbox && scene.bbox.length === 4) {
      return scene.bbox;
    }
    if (aoiBbox) {
      if (Array.isArray(aoiBbox) && aoiBbox.length === 4) {
        return aoiBbox;
      }
      if (typeof aoiBbox === "object" && "west" in (aoiBbox as any)) {
        const b = aoiBbox as any;
        return [b.west, b.south, b.east, b.north];
      }
    }
    if (location?.bbox_list && location.bbox_list.length === 4) {
      return location.bbox_list;
    }
    if (location?.bbox) {
      return [
        location.bbox.west,
        location.bbox.south,
        location.bbox.east,
        location.bbox.north,
      ];
    }
    if (
      typeof location?.latitude === "number" &&
      typeof location?.longitude === "number" &&
      (location.latitude !== 0 || location.longitude !== 0)
    ) {
      const d = 0.05;
      return [
        location.longitude - d,
        location.latitude - d,
        location.longitude + d,
        location.latitude + d,
      ];
    }
    return null;
  }, [geoMeta, scene, aoiBbox, location]);

  const dynamicCenter = useMemo<{ lat: number; lng: number } | null>(() => {
    if (effectiveBbox && effectiveBbox.length === 4) {
      return {
        lat: (effectiveBbox[1] + effectiveBbox[3]) / 2.0,
        lng: (effectiveBbox[0] + effectiveBbox[2]) / 2.0,
      };
    }
    if (
      typeof location?.latitude === "number" &&
      typeof location?.longitude === "number" &&
      (location.latitude !== 0 || location.longitude !== 0)
    ) {
      return { lat: location.latitude, lng: location.longitude };
    }
    return null;
  }, [effectiveBbox, location]);

  const dynamicZoom = useMemo(() => {
    return calculateDynamicZoom(effectiveBbox, location?.source);
  }, [effectiveBbox, location]);

  // Determine explicit MapStatus for UX inspection
  const mapStatus: MapStatus = useMemo(() => {
    if (!apiKey) return "NO_API_KEY";
    if (loadError) return "API_ERROR";
    if (!mapsLoaded) return "LOADING";
    if (!dynamicCenter && !effectiveBbox) return "NO_LOCATION";
    const featuresList = geojson?.features || [];
    if (featuresList.length > 0) return "DETECTIONS_VISIBLE";
    if (effectiveBbox) return "SCENE_ONLY";
    return "NO_GEOMETRY";
  }, [apiKey, loadError, mapsLoaded, dynamicCenter, effectiveBbox, geojson]);

  // --------------------------------------------------------------------------
  // 2. Google Maps Script Loader
  // --------------------------------------------------------------------------
  useEffect(() => {
    if (!apiKey) {
      setLoadError("NEXT_PUBLIC_GOOGLE_MAPS_API_KEY is not configured in environment.");
      return;
    }

    if (window.google?.maps) {
      setMapsLoaded(true);
      return;
    }

    if (!window.__googleMapsLoadingPromise) {
      window.__googleMapsLoadingPromise = new Promise<void>((resolve, reject) => {
        const script = document.createElement("script");
        script.src = `https://maps.googleapis.com/maps/api/js?key=${apiKey}&libraries=geometry`;
        script.async = true;
        script.defer = true;
        script.onload = () => resolve();
        script.onerror = () => reject(new Error("Failed to load Google Maps JavaScript API"));
        document.head.appendChild(script);
      });
    }

    window.__googleMapsLoadingPromise
      .then(() => {
        setMapsLoaded(true);
        setLoadError(null);
      })
      .catch((err) => {
        setLoadError(err.message || "Failed to load Google Maps");
      });
  }, [apiKey]);

  // --------------------------------------------------------------------------
  // 3. Initialize Interactive Google Map Instance
  // --------------------------------------------------------------------------
  useEffect(() => {
    if (!mapsLoaded || !mapContainerRef.current || !window.google?.maps || !dynamicCenter) return;

    if (!mapInstanceRef.current) {
      mapInstanceRef.current = new window.google.maps.Map(mapContainerRef.current, {
        center: dynamicCenter,
        zoom: dynamicZoom,
        mapTypeId: mapType,
        tilt: 0,
        mapTypeControl: false,
        streetViewControl: false,
        fullscreenControl: false,
        zoomControl: false,
        styles: [
          {
            featureType: "all",
            elementType: "labels.text.fill",
            stylers: [{ color: "#ffffff" }],
          },
        ],
      });

      infoWindowRef.current = new window.google.maps.InfoWindow();
    } else {
      mapInstanceRef.current.setCenter(dynamicCenter);
    }
  }, [mapsLoaded, dynamicCenter, dynamicZoom, mapType]);

  // Helper to open stylized InfoWindow
  const openInfoWindowForFeature = useCallback((feature: GeoJSONFeature, position: any) => {
    if (!infoWindowRef.current || !mapInstanceRef.current) return;

    const props = feature.properties || {};
    const lat = typeof position.lat === "function" ? position.lat() : position.lat;
    const lng = typeof position.lng === "function" ? position.lng() : position.lng;
    const gmapsUrl = `https://www.google.com/maps?q=${lat.toFixed(6)},${lng.toFixed(6)}`;

    const contentString = `
      <div style="background-color: #0b1320; color: #f8fafc; padding: 12px; border-radius: 8px; font-family: ui-sans-serif, system-ui, sans-serif; font-size: 13px; max-width: 290px; border: 1px solid #1e293b;">
        <div style="font-weight: 700; color: #38bdf8; font-size: 14px; margin-bottom: 4px; display: flex; align-items: center; justify-content: space-between;">
          <span>${props.label || "Detected Target"}</span>
          <span style="font-size: 11px; background: rgba(16, 185, 129, 0.2); color: #34d399; padding: 2px 6px; border-radius: 4px;">
            ${props.confidence ? Math.round(Number(props.confidence) * 100) + "% conf" : "Verified"}
          </span>
        </div>
        <div style="color: #94a3b8; font-size: 11px; margin-bottom: 8px;">
          Target ID: <code style="color: #cbd5e1;">${props.evidence_id || feature.id || "N/A"}</code>
          <span style="margin-left: 6px; color: #38bdf8; text-transform: uppercase; font-size: 10px;">
            [${props.feature_category || "discrete"}]
          </span>
        </div>
        <div style="background: rgba(15, 23, 42, 0.7); padding: 6px 8px; border-radius: 6px; border: 1px solid #1e293b; margin-bottom: 8px; font-size: 11px; line-height: 1.5;">
          <div style="color: #94a3b8;">WGS84 Coordinates:</div>
          <div>Lat: <strong style="color: #f1f5f9;">${lat.toFixed(6)}°</strong></div>
          <div>Lon: <strong style="color: #f1f5f9;">${lng.toFixed(6)}°</strong></div>
          <div style="color: #38bdf8; font-size: 10px; margin-top: 2px;">
            Georeferencing: ${props.native_crs ? "GeoTIFF / " + props.native_crs : "Sentinel-2 Exact"}
          </div>
          ${
            props.ground_area
              ? `<div style="color: #10b981; font-size: 10px;">Ground Area: ${props.ground_area} ${props.ground_area_unit || "m²"}</div>`
              : ""
          }
        </div>
        <a href="${gmapsUrl}" target="_blank" rel="noopener noreferrer" style="display: flex; align-items: center; justify-content: center; gap: 4px; background: #0284c7; color: #ffffff; text-decoration: none; padding: 6px 10px; border-radius: 6px; font-size: 11px; font-weight: 600; text-align: center;">
          Open in Google Maps ↗
        </a>
      </div>
    `;

    infoWindowRef.current.setContent(contentString);
    infoWindowRef.current.setPosition(position);
    infoWindowRef.current.open(mapInstanceRef.current);
  }, []);

  // --------------------------------------------------------------------------
  // 4. Render Dynamic Content Hierarchy (Markers, Polygons, Polylines, Footprint)
  // --------------------------------------------------------------------------
  useEffect(() => {
    if (!mapsLoaded || !mapInstanceRef.current || !window.google?.maps) return;

    const map = mapInstanceRef.current;
    const bounds = new window.google.maps.LatLngBounds();
    let hasGeom = false;

    // Clear prior layers to guarantee zero stale coordinate leakage
    markersRef.current.forEach((m) => m.setMap(null));
    markersRef.current.clear();
    polygonsRef.current.forEach((p) => p.setMap(null));
    polygonsRef.current.clear();
    polylinesRef.current.forEach((l) => l.setMap(null));
    polylinesRef.current.clear();
    osmMarkersRef.current.forEach((m) => m.setMap(null));
    osmMarkersRef.current.clear();
    osmPolygonsRef.current.forEach((p) => p.setMap(null));
    osmPolygonsRef.current.clear();
    osmPolylinesRef.current.forEach((l) => l.setMap(null));
    osmPolylinesRef.current.clear();
    if (groundOverlayRef.current) {
      groundOverlayRef.current.setMap(null);
      groundOverlayRef.current = null;
    }

    // Render Satellite Raster GroundOverlay if provided (Requirement 10: Map Synchronization)
    if (overlayUrl && window.google?.maps?.GroundOverlay) {
      let north = overlayBounds?.north;
      let south = overlayBounds?.south;
      let east = overlayBounds?.east;
      let west = overlayBounds?.west;

      if (north === undefined || south === undefined || east === undefined || west === undefined) {
        if (effectiveBbox && effectiveBbox.length === 4) {
          west = effectiveBbox[0];
          south = effectiveBbox[1];
          east = effectiveBbox[2];
          north = effectiveBbox[3];
        }
      }

      if (
        typeof north === "number" &&
        typeof south === "number" &&
        typeof east === "number" &&
        typeof west === "number"
      ) {
        // Enforce strict orientation: west < east and south < north
        if (south > north) {
          const tmp = south;
          south = north;
          north = tmp;
        }
        if (west > east) {
          const tmp = west;
          west = east;
          east = tmp;
        }

        try {
          const sw = new window.google.maps.LatLng(south, west);
          const ne = new window.google.maps.LatLng(north, east);
          const gBounds = new window.google.maps.LatLngBounds(sw, ne);
          const overlay = new window.google.maps.GroundOverlay(overlayUrl, gBounds, {
            opacity: 0.90,
            clickable: false,
          });
          overlay.setMap(map);
          groundOverlayRef.current = overlay;
          bounds.extend(sw);
          bounds.extend(ne);
          hasGeom = true;
        } catch (err) {
          console.warn("[GoogleMapView] Failed to instantiate GroundOverlay:", err);
        }
      }
    }

    // A. Render AOI Bounding Box (if requested)
    if (aoiBbox && aoiBbox.length === 4) {
      const [minLon, minLat, maxLon, maxLat] = aoiBbox;
      const aoiCoords = [
        { lat: minLat, lng: minLon },
        { lat: maxLat, lng: minLon },
        { lat: maxLat, lng: maxLon },
        { lat: minLat, lng: maxLon },
      ];

      const aoiPolygon = new window.google.maps.Polygon({
        paths: aoiCoords,
        strokeColor: "#06b6d4",
        strokeOpacity: 0.8,
        strokeWeight: 1.5,
        fillColor: "#06b6d4",
        fillOpacity: 0.06,
        map: map,
      });
      polygonsRef.current.set("aoi_boundary", aoiPolygon);
      aoiCoords.forEach((c) => bounds.extend(c));
      hasGeom = true;
    }

    // B. Render Satellite Scene Footprint (Requirement 8 & 13)
    // Always preserved even when detections = 0
    const footprintBounds = geoMeta?.bounds_world
      ? {
          minLon: geoMeta.bounds_world.min_x,
          minLat: geoMeta.bounds_world.min_y,
          maxLon: geoMeta.bounds_world.max_x,
          maxLat: geoMeta.bounds_world.max_y,
        }
      : scene?.bbox && scene.bbox.length === 4
      ? {
          minLon: scene.bbox[0],
          minLat: scene.bbox[1],
          maxLon: scene.bbox[2],
          maxLat: scene.bbox[3],
        }
      : null;

    if (footprintBounds) {
      const footprintCoords = [
        { lat: footprintBounds.minLat, lng: footprintBounds.minLon },
        { lat: footprintBounds.maxLat, lng: footprintBounds.minLon },
        { lat: footprintBounds.maxLat, lng: footprintBounds.maxLon },
        { lat: footprintBounds.minLat, lng: footprintBounds.maxLon },
      ];

      const footprintPoly = new window.google.maps.Polygon({
        paths: footprintCoords,
        strokeColor: "#38bdf8",
        strokeOpacity: 0.9,
        strokeWeight: 2,
        fillColor: "#38bdf8",
        fillOpacity: 0.05,
        map: map,
      });
      polygonsRef.current.set("scene_footprint", footprintPoly);
      footprintCoords.forEach((c) => bounds.extend(c));
      hasGeom = true;
    }

    // C. Render GeoJSON Evidence Targets according to Content Hierarchy (Requirement 7)
    if (geojson?.features && geojson.features.length > 0) {
      geojson.features.forEach((feature: GeoJSONFeature, idx: number) => {
        const featId = feature.id || feature.properties?.evidence_id || `feat_${idx}`;
        const geom = feature.geometry;
        const props = feature.properties || {};
        const category = props.feature_category || "discrete";
        const isSelected = selectedDetectionId === featId;

        if (geom.type === "Polygon" && Array.isArray(geom.coordinates)) {
          const rawRing = geom.coordinates[0] as number[][];
          if (!rawRing || rawRing.length < 3) return;

          const path = rawRing.map(([lon, lat]) => ({ lat, lng: lon }));
          path.forEach((pt) => bounds.extend(pt));
          hasGeom = true;

          // Compute Centroid
          const centerLat = path.reduce((sum, p) => sum + p.lat, 0) / path.length;
          const centerLng = path.reduce((sum, p) => sum + p.lng, 0) / path.length;
          const centerPt = { lat: centerLat, lng: centerLng };

          // Hierarchy 1: Linear Features -> Polyline
          if (category === "linear") {
            const polyline = new window.google.maps.Polyline({
              path: path,
              strokeColor: isSelected ? "#10b981" : "#06b6d4",
              strokeOpacity: isSelected ? 1.0 : 0.9,
              strokeWeight: isSelected ? 4 : 3,
              map: map,
            });

            polyline.addListener("click", () => {
              if (onSelectDetection) onSelectDetection(featId);
              setActiveFeature(feature);
              openInfoWindowForFeature(feature, centerPt);
            });
            polylinesRef.current.set(featId, polyline);
          }
          // Hierarchy 2: Region Features -> Polygon
          else if (category === "region") {
            const poly = new window.google.maps.Polygon({
              paths: path,
              strokeColor: isSelected ? "#10b981" : "#38bdf8",
              strokeOpacity: isSelected ? 1.0 : 0.85,
              strokeWeight: isSelected ? 3 : 2,
              fillColor: isSelected ? "#10b981" : "#38bdf8",
              fillOpacity: isSelected ? 0.35 : 0.15,
              map: map,
            });

            poly.addListener("click", () => {
              if (onSelectDetection) onSelectDetection(featId);
              setActiveFeature(feature);
              openInfoWindowForFeature(feature, centerPt);
            });
            polygonsRef.current.set(featId, poly);
          }
          // Hierarchy 3: Discrete Objects -> Marker + Bounding Polygon
          else {
            const poly = new window.google.maps.Polygon({
              paths: path,
              strokeColor: isSelected ? "#10b981" : "#06b6d4",
              strokeOpacity: isSelected ? 1.0 : 0.8,
              strokeWeight: isSelected ? 2.5 : 1.5,
              fillColor: isSelected ? "#10b981" : "#06b6d4",
              fillOpacity: isSelected ? 0.25 : 0.1,
              map: map,
            });

            poly.addListener("click", () => {
              if (onSelectDetection) onSelectDetection(featId);
              setActiveFeature(feature);
              openInfoWindowForFeature(feature, centerPt);
            });
            polygonsRef.current.set(featId, poly);

            // Centroid Pin Marker
            const marker = new window.google.maps.Marker({
              position: centerPt,
              map: map,
              title: `${props.label || "Target"} (#${featId})`,
              icon: {
                path: window.google.maps.SymbolPath.CIRCLE,
                scale: isSelected ? 8 : 5.5,
                fillColor: isSelected ? "#10b981" : "#06b6d4",
                fillOpacity: 1,
                strokeColor: "#ffffff",
                strokeWeight: 1.5,
              },
            });

            marker.addListener("click", () => {
              if (onSelectDetection) onSelectDetection(featId);
              setActiveFeature(feature);
              openInfoWindowForFeature(feature, centerPt);
            });
            markersRef.current.set(featId, marker);
          }
        }
      });
    }

    // D. Render OpenStreetMap Cartographic Vector Layer (Phase 2 Requirement)
    if (showOsmLayer && osmGeojson?.features && osmGeojson.features.length > 0) {
      osmGeojson.features.forEach((feat: GeoJSONFeature, idx: number) => {
        const osmId: string = String(feat.id || feat.properties?.osm_id || `osm_${idx}`);
        const geom = feat.geometry;
        const props = feat.properties || {};
        const featClass = String(props.feature_class || "other").toLowerCase();

        // Distinct category styling
        let strokeColor = "#818cf8"; // roads: violet
        let fillColor = "#818cf8";
        let fillOpacity = 0.15;
        let strokeWeight = 2;

        if (featClass === "buildings") {
          strokeColor = "#f59e0b"; // amber-500
          fillColor = "#fbbf24";   // amber-400
          fillOpacity = 0.25;
          strokeWeight = 1.5;
        } else if (featClass === "bridges") {
          strokeColor = "#14b8a6"; // teal-500
          fillColor = "#2dd4bf";
          fillOpacity = 0.25;
          strokeWeight = 3.5;
        } else if (featClass === "waterways") {
          strokeColor = "#0284c7"; // sky-600
          fillColor = "#38bdf8";
          fillOpacity = 0.25;
          strokeWeight = 2.5;
        } else if (featClass === "infrastructure") {
          strokeColor = "#ec4899"; // pink-500
          fillColor = "#f472b6";
          fillOpacity = 0.20;
          strokeWeight = 2;
        }

        const openOsmInfoWindow = (pos: any) => {
          if (!infoWindowRef.current || !mapInstanceRef.current) return;
          const lat = typeof pos.lat === "function" ? pos.lat() : pos.lat;
          const lng = typeof pos.lng === "function" ? pos.lng() : pos.lng;
          const content = `
            <div style="background-color: #0b1320; color: #f8fafc; padding: 12px; border-radius: 8px; font-family: ui-sans-serif, system-ui, sans-serif; font-size: 13px; max-width: 300px; border: 1px solid #d97706;">
              <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 4px;">
                <span style="font-weight: 700; color: #fbbf24; font-size: 13px;">${props.name || "OSM " + featClass.toUpperCase()}</span>
                <span style="font-size: 10px; background: rgba(245, 158, 11, 0.2); color: #fbbf24; padding: 2px 6px; border-radius: 4px; font-family: monospace;">
                  ${featClass.toUpperCase()}
                </span>
              </div>
              <div style="color: #94a3b8; font-size: 10px; margin-bottom: 6px; font-family: monospace;">
                OSM ID: ${props.osm_id || feat.id || "N/A"}
              </div>
              <div style="background: rgba(15, 23, 42, 0.8); padding: 6px 8px; border-radius: 6px; border: 1px solid #1e293b; margin-bottom: 8px; font-size: 11px;">
                <div>Lat: <strong>${lat.toFixed(6)}°</strong>, Lon: <strong>${lng.toFixed(6)}°</strong></div>
                <div style="color: #fbbf24; font-size: 10px; margin-top: 4px;">
                  ⚠️ Crowdsourced cartographic record — not proof of current satellite visibility.
                </div>
              </div>
              <div style="color: #64748b; font-size: 10px; text-align: right;">
                © OpenStreetMap contributors
              </div>
            </div>
          `;
          infoWindowRef.current.setContent(content);
          infoWindowRef.current.setPosition(pos);
          infoWindowRef.current.open(mapInstanceRef.current);
        };

        if (geom.type === "Polygon" && Array.isArray(geom.coordinates)) {
          const rawRing = geom.coordinates[0] as number[][];
          if (!rawRing || rawRing.length < 3) return;
          const path = rawRing.map(([lon, lat]) => ({ lat, lng: lon }));
          path.forEach((pt) => bounds.extend(pt));
          hasGeom = true;

          const centerLat = path.reduce((sum, p) => sum + p.lat, 0) / path.length;
          const centerLng = path.reduce((sum, p) => sum + p.lng, 0) / path.length;
          const centerPt = { lat: centerLat, lng: centerLng };

          const poly = new window.google.maps.Polygon({
            paths: path,
            strokeColor: strokeColor,
            strokeOpacity: 0.9,
            strokeWeight: strokeWeight,
            fillColor: fillColor,
            fillOpacity: fillOpacity,
            map: map,
          });
          poly.addListener("click", () => openOsmInfoWindow(centerPt));
          osmPolygonsRef.current.set(osmId, poly);
        } else if (geom.type === "LineString" && Array.isArray(geom.coordinates)) {
          const rawCoords = geom.coordinates as number[][];
          if (!rawCoords || rawCoords.length < 2) return;
          const path = rawCoords.map(([lon, lat]) => ({ lat, lng: lon }));
          path.forEach((pt) => bounds.extend(pt));
          hasGeom = true;
          const midPt = path[Math.floor(path.length / 2)];

          const polyline = new window.google.maps.Polyline({
            path: path,
            strokeColor: strokeColor,
            strokeOpacity: 0.85,
            strokeWeight: strokeWeight,
            map: map,
          });
          polyline.addListener("click", () => openOsmInfoWindow(midPt));
          osmPolylinesRef.current.set(osmId, polyline);
        } else if (geom.type === "Point" && Array.isArray(geom.coordinates)) {
          const [lon, lat] = geom.coordinates as number[];
          const pt = { lat, lng: lon };
          bounds.extend(pt);
          hasGeom = true;

          const marker = new window.google.maps.Marker({
            position: pt,
            map: map,
            title: `OSM ${featClass}: ${props.name || osmId}`,
            icon: {
              path: window.google.maps.SymbolPath.CIRCLE,
              scale: 4.5,
              fillColor: strokeColor,
              fillOpacity: 0.9,
              strokeColor: "#ffffff",
              strokeWeight: 1,
            },
          });
          marker.addListener("click", () => openOsmInfoWindow(pt));
          osmMarkersRef.current.set(osmId, marker);
        }
      });
    }

    // Dynamic fitBounds execution (Requirement 6)
    if (hasGeom && !bounds.isEmpty()) {
      map.fitBounds(bounds);
      const listener = window.google.maps.event.addListener(map, "idle", () => {
        if (map.getZoom() > 17) map.setZoom(16);
        window.google.maps.event.removeListener(listener);
      });
    } else if (dynamicCenter) {
      map.setCenter(dynamicCenter);
      map.setZoom(dynamicZoom);
    }
  }, [
    mapsLoaded,
    geojson,
    osmGeojson,
    showOsmLayer,
    aoiBbox,
    geoMeta,
    scene,
    dynamicCenter,
    dynamicZoom,
    selectedDetectionId,
    onSelectDetection,
    openInfoWindowForFeature,
    overlayUrl,
    overlayBounds,
    effectiveBbox,
  ]);

  // --------------------------------------------------------------------------
  // 5. Two-Way Selection Sync
  // --------------------------------------------------------------------------
  useEffect(() => {
    if (!mapsLoaded || !mapInstanceRef.current || !selectedDetectionId) return;

    const map = mapInstanceRef.current;
    const marker = markersRef.current.get(selectedDetectionId);
    const poly = polygonsRef.current.get(selectedDetectionId);
    const polyline = polylinesRef.current.get(selectedDetectionId);

    if (marker) {
      map.panTo(marker.getPosition());
      const feat = geojson?.features?.find(
        (f) => (f.id || f.properties?.evidence_id) === selectedDetectionId
      );
      if (feat) openInfoWindowForFeature(feat, marker.getPosition());
    } else if (poly) {
      const paths = poly.getPath().getArray();
      if (paths.length > 0) {
        map.panTo(paths[0]);
        const feat = geojson?.features?.find(
          (f) => (f.id || f.properties?.evidence_id) === selectedDetectionId
        );
        if (feat) openInfoWindowForFeature(feat, paths[0]);
      }
    } else if (polyline) {
      const path = polyline.getPath().getArray();
      if (path.length > 0) {
        map.panTo(path[0]);
        const feat = geojson?.features?.find(
          (f) => (f.id || f.properties?.evidence_id) === selectedDetectionId
        );
        if (feat) openInfoWindowForFeature(feat, path[0]);
      }
    }
  }, [selectedDetectionId, mapsLoaded, geojson, openInfoWindowForFeature]);

  // Map Type Toggle Handler
  const handleMapTypeChange = (newType: "hybrid" | "satellite" | "roadmap") => {
    setMapType(newType);
    if (mapInstanceRef.current) {
      mapInstanceRef.current.setMapTypeId(newType);
    }
  };

  // --------------------------------------------------------------------------
  // 6. Resilient Keyless Fallback View (Requirement 12 & 20)
  // --------------------------------------------------------------------------
  if (loadError || !apiKey) {
    const featuresList = geojson?.features || [];
    const hasSceneFootprint = Boolean(effectiveBbox && effectiveBbox.length === 4);
    const centerLat = dynamicCenter?.lat ?? null;
    const centerLon = dynamicCenter?.lng ?? null;
    const gmapsSceneUrl =
      centerLat && centerLon
        ? `https://www.google.com/maps?q=${centerLat.toFixed(6)},${centerLon.toFixed(6)}`
        : null;

    // Geolocation Source Label (Requirement 24)
    const sourceLabel =
      geoMeta?.crs || scene?.crs
        ? "📍 Exact — GeoTIFF / Scene Metadata"
        : location?.source === "EXPLICIT_COORDINATES"
        ? "📍 Exact — Explicit Coordinates"
        : location?.source === "GEOCODED_QUERY"
        ? "📍 Geocoded — Location Query"
        : location?.source === "MODEL_ESTIMATE"
        ? "📍 Estimated — Model-derived"
        : "📍 Unknown — Insufficient Evidence";

    return (
      <div
        className={`relative flex flex-col bg-[#070f1a] border border-slate-800 rounded-xl overflow-hidden shadow-2xl ${className}`}
        style={{ height }}
      >
        {/* Header Bar */}
        <div className="flex items-center justify-between px-4 py-2.5 bg-[#0b1524] border-b border-slate-800/80 text-xs">
          <div className="flex items-center gap-2 text-cyan-400 font-semibold">
            <Globe className="w-4 h-4" />
            <span>Universal Geospatial Evidence Inspector</span>
            <span className="px-2 py-0.5 rounded bg-cyan-500/10 border border-cyan-500/30 text-[10px] text-cyan-300 font-mono">
              WGS84 EPSG:4326
            </span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-slate-400 text-[11px] font-mono">{sourceLabel}</span>
          </div>
        </div>

        {/* Notice Bar */}
        <div className="bg-sky-500/10 border-b border-sky-500/20 px-4 py-2 flex items-center justify-between text-xs text-sky-300">
          <div className="flex items-center gap-2">
            <Info className="w-4 h-4 text-sky-400 flex-shrink-0" />
            <span>
              Direct geospatial WGS84 coordinate inspector with external Google Maps synchronization.
            </span>
          </div>
          {gmapsSceneUrl && (
            <a
              href={gmapsSceneUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1.5 px-3 py-1 rounded bg-sky-600/30 hover:bg-sky-600/50 text-xs text-white border border-sky-400/40 transition-colors font-medium"
            >
              <span>Open in Google Maps</span>
              <ExternalLink className="w-3 h-3" />
            </a>
          )}
        </div>

        {/* Coordinate & Feature List */}
        <div className="flex-1 overflow-y-auto p-4 space-y-3">
          {/* Real Sentinel-2 Raster Display in Geospatial Inspector */}
          {overlayUrl && (
            <div className="rounded-xl overflow-hidden border border-cyan-500/40 bg-black/60 p-2 shadow-xl">
              <div className="flex items-center justify-between px-2 py-1 mb-2 border-b border-cyan-500/20">
                <span className="text-xs font-mono text-cyan-300 font-bold flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
                  Retrieved Sentinel-2 Optical Raster (Georeferenced)
                </span>
                <span className="text-[10px] font-mono text-emerald-300 bg-emerald-950/60 px-2 py-0.5 rounded border border-emerald-500/30">
                  {scene?.platform || "Sentinel-2"} • 10m GSD
                </span>
              </div>
              <div className="relative aspect-video max-h-56 w-full flex items-center justify-center bg-slate-950 rounded-lg overflow-hidden border border-slate-800">
                <img
                  src={overlayUrl}
                  alt={scene?.scene_id || "Sentinel-2 Satellite Imagery"}
                  className="w-full h-full object-contain"
                />
              </div>
            </div>
          )}

          {/* Section 23: Pure Satellite Discovery or Zero-Detection UX */}
          {featuresList.length === 0 ? (
            <div className="p-4 rounded-xl border border-cyan-500/40 bg-cyan-950/20 shadow-lg space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 animate-pulse" />
                  <span className="font-bold text-slate-100 text-sm">
                    {overlayUrl || scene ? "Satellite Image Found & Georeferenced" : "Geographic Context Verified (0 Detections)"}
                  </span>
                  <span className="text-[10px] font-mono text-cyan-300 bg-cyan-500/20 px-2 py-0.5 rounded-full border border-cyan-500/30">
                    {overlayUrl || scene ? "SATELLITE IMAGE FOUND" : "Scene AOI"}
                  </span>
                </div>
              </div>

              <div className="text-xs text-slate-300 space-y-1">
                {overlayUrl || scene ? (
                  <p className="font-semibold text-emerald-300">SATELLITE IMAGE FOUND — Sentinel-2 Optical Surface Reflectance</p>
                ) : (
                  <p className="font-semibold text-slate-200">No matching targets detected.</p>
                )}
                <div className="text-slate-400 space-y-0.5 pl-2 border-l border-slate-700">
                  <p>✓ Location resolved: <strong className="text-cyan-300">{location?.displayName || location?.query || scene?.platform || "Verified"}</strong></p>
                  <p>✓ Satellite scene georeferenced: <strong className="text-slate-200">{geoMeta?.crs || scene?.crs || "EPSG:4326"}</strong></p>
                  <p>✓ Scene footprint available: <strong className="text-emerald-300 font-mono">[{effectiveBbox?.map((v) => v.toFixed(3)).join(", ")}]</strong></p>
                  {overlayUrl || scene ? (
                    <p className="text-cyan-400 font-mono text-xs">✓ Real optical raster retrieved and registered on geospatial extent.</p>
                  ) : (
                    <p className="text-slate-500 italic">No verified target geometry is available for this specific query.</p>
                  )}
                </div>
              </div>

              {dynamicCenter && (
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5 text-[11px] text-slate-300 bg-slate-950/70 p-3 rounded-lg border border-slate-800">
                  <div>
                    <span className="text-slate-500 block">Center Latitude:</span>
                    <strong className="text-cyan-300">{dynamicCenter.lat.toFixed(6)}°</strong>
                  </div>
                  <div>
                    <span className="text-slate-500 block">Center Longitude:</span>
                    <strong className="text-cyan-300">{dynamicCenter.lng.toFixed(6)}°</strong>
                  </div>
                  <div>
                    <span className="text-slate-500 block">Native CRS:</span>
                    <span className="text-slate-200">{geoMeta?.crs || scene?.crs || "EPSG:4326"}</span>
                  </div>
                  <div>
                    <span className="text-slate-500 block">Bounding Extent:</span>
                    <span className="text-emerald-300 font-mono text-[10px]">
                      [{effectiveBbox?.[0]?.toFixed(3)}, {effectiveBbox?.[1]?.toFixed(3)}]
                    </span>
                  </div>
                </div>
              )}
            </div>
          ) : (
            featuresList.map((feat, idx) => {
              const featId = feat.id || feat.properties?.evidence_id || `feat_${idx}`;
              const isSelected = selectedDetectionId === featId;
              const geom = feat.geometry;
              const coords =
                geom?.type === "Polygon" && Array.isArray(geom.coordinates)
                  ? (geom.coordinates[0] as number[][])
                  : [];
              const firstPt = coords[0] || [0, 0];
              const lon = firstPt[0];
              const lat = firstPt[1];
              const gmapsUrl = `https://www.google.com/maps?q=${lat.toFixed(6)},${lon.toFixed(6)}`;

              return (
                <div
                  key={featId}
                  onClick={() => onSelectDetection && onSelectDetection(featId)}
                  className={`p-3 rounded-lg border transition-all cursor-pointer ${
                    isSelected
                      ? "bg-emerald-500/10 border-emerald-500/40 shadow-lg"
                      : "bg-[#0c1626] border-slate-800 hover:border-slate-700"
                  }`}
                >
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <span
                        className={`w-2 h-2 rounded-full ${
                          isSelected ? "bg-emerald-400 animate-pulse" : "bg-cyan-400"
                        }`}
                      />
                      <span className="font-semibold text-slate-200 text-sm">
                        {feat.properties?.label || "Target"}
                      </span>
                      <code className="text-xs text-slate-400 bg-slate-900 px-1.5 py-0.5 rounded">
                        #{featId}
                      </code>
                      <span className="text-[10px] uppercase font-mono px-1.5 py-0.5 rounded bg-slate-800 text-cyan-300">
                        {String(feat.properties?.feature_category || "discrete")}
                      </span>
                    </div>
                    <a
                      href={gmapsUrl}
                      target="_blank"
                      rel="noopener noreferrer"
                      onClick={(e) => e.stopPropagation()}
                      className="flex items-center gap-1 text-xs text-sky-400 hover:text-sky-300 bg-sky-500/10 hover:bg-sky-500/20 px-2.5 py-1 rounded border border-sky-500/30 transition-colors"
                    >
                      <span>Open Target ↗</span>
                      <ExternalLink className="w-3 h-3" />
                    </a>
                  </div>

                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-[11px] text-slate-400 bg-slate-950/60 p-2 rounded">
                    <div>
                      <span className="text-slate-500 block">Latitude:</span>
                      <strong className="text-slate-200">{lat.toFixed(6)}°</strong>
                    </div>
                    <div>
                      <span className="text-slate-500 block">Longitude:</span>
                      <strong className="text-slate-200">{lon.toFixed(6)}°</strong>
                    </div>
                    <div>
                      <span className="text-slate-500 block">Source:</span>
                      <span className="text-cyan-300">
                        {feat.properties?.native_crs || "Sentinel-2 / WGS84"}
                      </span>
                    </div>
                    <div>
                      <span className="text-slate-500 block">Ground Area:</span>
                      <span className="text-emerald-300">
                        {feat.properties?.ground_area
                          ? `${feat.properties.ground_area} ${feat.properties.ground_area_unit || "m²"}`
                          : "Calculated from bounds"}
                      </span>
                    </div>
                  </div>
                </div>
              );
            })
          )}
        </div>
      </div>
    );
  }

  // --------------------------------------------------------------------------
  // 7. Interactive Google Map View (When API Key is Present)
  // --------------------------------------------------------------------------
  return (
    <div
      className={`relative flex flex-col bg-[#070f1a] border border-slate-800 rounded-xl overflow-hidden shadow-2xl ${className}`}
      style={{ height }}
    >
      {/* Top Map Type Control Bar */}
      <div className="absolute top-3 left-3 z-10 flex items-center gap-1.5 bg-[#0b1524]/90 backdrop-blur-md px-2 py-1.5 rounded-lg border border-slate-700/60 shadow-xl">
        <button
          onClick={() => handleMapTypeChange("hybrid")}
          className={`px-2.5 py-1 rounded text-xs font-medium transition-all ${
            mapType === "hybrid"
              ? "bg-cyan-500 text-slate-950 shadow-md font-semibold"
              : "text-slate-300 hover:text-white hover:bg-slate-800"
          }`}
        >
          Hybrid
        </button>
        <button
          onClick={() => handleMapTypeChange("satellite")}
          className={`px-2.5 py-1 rounded text-xs font-medium transition-all ${
            mapType === "satellite"
              ? "bg-cyan-500 text-slate-950 shadow-md font-semibold"
              : "text-slate-300 hover:text-white hover:bg-slate-800"
          }`}
        >
          Satellite
        </button>
        <button
          onClick={() => handleMapTypeChange("roadmap")}
          className={`px-2.5 py-1 rounded text-xs font-medium transition-all ${
            mapType === "roadmap"
              ? "bg-cyan-500 text-slate-950 shadow-md font-semibold"
              : "text-slate-300 hover:text-white hover:bg-slate-800"
          }`}
        >
          Roadmap
        </button>

        {osmGeojson?.features && osmGeojson.features.length > 0 && (
          <button
            onClick={() => setShowOsmLayer(!showOsmLayer)}
            className={`flex items-center gap-1.5 px-2.5 py-1 rounded text-xs font-mono transition-all ml-1 ${
              showOsmLayer
                ? "bg-amber-500/20 text-amber-300 border border-amber-400/40 shadow-sm"
                : "text-slate-400 hover:text-white hover:bg-slate-800 border border-transparent"
            }`}
            title="Toggle OpenStreetMap Vector Layer (© OpenStreetMap contributors)"
          >
            <Layers className="w-3.5 h-3.5" />
            <span>OSM Layer ({osmGeojson.features.length})</span>
          </button>
        )}
      </div>

      {/* Top Right Status & CRS Badge */}
      <div className="absolute top-3 right-3 z-10 flex items-center gap-2 bg-[#0b1524]/90 backdrop-blur-md px-3 py-1.5 rounded-lg border border-slate-700/60 shadow-xl text-xs">
        <ShieldCheck className="w-4 h-4 text-emerald-400" />
        <span className="text-slate-300">
          CRS: <strong className="text-cyan-300">{geoMeta?.crs || scene?.crs || "EPSG:4326 (WGS84)"}</strong>
        </span>
      </div>

      {/* Zero Detections Floating Context Banner */}
      {geojson && geojson.features.length === 0 && effectiveBbox && (
        <div className="absolute bottom-3 left-3 z-10 bg-[#0b1524]/90 backdrop-blur-md px-3 py-2 rounded-lg border border-cyan-500/40 shadow-xl text-xs text-slate-300 max-w-sm space-y-1">
          <div className="flex items-center gap-1.5 font-bold text-cyan-300">
            <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
            <span>Scene Footprint Verified (0 Detections)</span>
          </div>
          <p className="text-[11px] text-slate-400 leading-snug">
            Geographic scene extent is displayed on map. No discrete targets matched the query criteria.
          </p>
        </div>
      )}

      {/* Google Map Div */}
      <div ref={mapContainerRef} className="w-full h-full" />
    </div>
  );
}
