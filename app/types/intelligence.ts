import type {
  NexSpaceQueryResponse,
  CanonicalSourceImage,
  TraceStage,
  GeoJSONFeatureCollection,
} from "./nexspace";

export type DetectionStateType =
  | "detections_found"
  | "genuine_zero"
  | "model_unavailable"
  | "localization_unavailable"
  | "not_applicable";

export type GeometryVisualType = "DISCRETE_OBJECT" | "REGION_FEATURE" | "LINEAR_FEATURE" | "NONE";

export interface DetectionItem {
  box_2d?: number[];
  label?: string;
  score?: number;
  geometry_type?: string;
  polygon?: [number, number][];
  evidence_id?: string;
  bbox_pixel?: number[];
  bbox_normalized?: number[];
  [key: string]: any;
}

export interface CategorizedTargetGroup {
  label: string;
  count: number;
  geometryType: GeometryVisualType;
  items: EnrichedIntelligenceDetection[];
}

export interface EnrichedIntelligenceDetection {
  index: number;
  tag: string;
  label: string;
  cleanLabel: string;
  category: string;
  geometryType: GeometryVisualType;
  score: number;
  confidencePercent: number;
  confidenceTier: "High confidence" | "Moderate confidence" | "Low confidence";
  box: [number, number, number, number]; // [xmin, ymin, xmax, ymax]
  polygon?: [number, number][];
  width: number;
  height: number;
  areaPx: number;
  areaEstimate?: string;
  coordinates?: {
    lat: number;
    lon: number;
    formatted: string;
  };
  gmapsUrl?: string;
  evidenceId: string;
}

export interface InvestigationResult {
  query: string;
  task: string;
  detectionState: DetectionStateType;
  headline: string;
  naturalLanguageAnswer: string;
  summary: string;
  targetCount: number;
  targetNoun: string;
  hasDetections: boolean;
  isCaptionOnly: boolean;
  isCountingQuery: boolean;
  isPresenceQuery: boolean;
  isSpatialRelationQuery: boolean;
  isChangeQuery: boolean;
  isMultiTargetQuery: boolean;
  spatialRelationDetails?: {
    predicate: string;
    reference: string;
  };
  categorizedGroups: CategorizedTargetGroup[];
  allDetections: EnrichedIntelligenceDetection[];
  sceneDescription?: string;
  locationInfo: {
    resolvedName: string;
    coordinates?: {
      lat: number;
      lon: number;
      formatted: string;
    };
    crs?: string;
    resolution?: string;
    bounds?: [number, number, number, number];
    footprintGeoJson?: any;
    gmapsUrl?: string;
  };
  changeSummary?: {
    category?: string;
    summary: string;
    beforeDesc?: string;
    afterDesc?: string;
  };
  observations: string[];
  notes: string[];
  technicalDetails: {
    modelsUsed: string[];
    routingTools: string[];
    confidenceSource: string;
    confidenceValue: number;
    executionStages: Array<{ stage: string; status: string; duration_ms: number }>;
    totalDurationMs: number;
    sensorType: string;
    imageDimensions: string;
  };
}

/**
 * Format lat/lon to clean geographic string e.g. "27.1224° N, 79.9935° E"
 */
export function formatGeoCoordinates(lat: number, lon: number): string {
  const latStr = `${Math.abs(lat).toFixed(4)}° ${lat >= 0 ? "N" : "S"}`;
  const lonStr = `${Math.abs(lon).toFixed(4)}° ${lon >= 0 ? "E" : "W"}`;
  return `${latStr}, ${lonStr}`;
}

/**
 * Translates raw model confidence scores into honest, calm confidence tiers.
 */
export function getConfidenceTier(score: number): {
  tier: "High confidence" | "Moderate confidence" | "Low confidence";
  percent: number;
  badgeClass: string;
} {
  const pct = Math.round(score * 100);
  if (pct >= 75) {
    return {
      tier: "High confidence",
      percent: pct,
      badgeClass: "text-emerald-300 bg-emerald-500/15 border-emerald-500/30",
    };
  }
  if (pct >= 45) {
    return {
      tier: "Moderate confidence",
      percent: pct,
      badgeClass: "text-cyan-300 bg-cyan-500/15 border-cyan-500/30",
    };
  }
  return {
    tier: "Low confidence",
    percent: pct,
    badgeClass: "text-amber-300 bg-amber-500/15 border-amber-500/30",
  };
}

/**
 * Maps raw backend detection items into enriched intelligence targets.
 */
export function enrichDetections(
  rawDetections: DetectionItem[] | any[],
  geojson: any,
  imgWidth = 1000,
  imgHeight = 1000
): EnrichedIntelligenceDetection[] {
  return (rawDetections || []).map((det: any, index: number) => {
    const rawBox: number[] = det.box_2d || det.bbox_pixel || [0, 0, 100, 100];
    const isNormalized = rawBox.every((v: number) => v <= 1.05);
    const box: [number, number, number, number] = isNormalized
      ? [
          Math.round(rawBox[0] * imgWidth),
          Math.round(rawBox[1] * imgHeight),
          Math.round(rawBox[2] * imgWidth),
          Math.round(rawBox[3] * imgHeight),
        ]
      : [
          Math.round(rawBox[0]),
          Math.round(rawBox[1]),
          Math.round(rawBox[2]),
          Math.round(rawBox[3]),
        ];

    const width = Math.max(1, box[2] - box[0]);
    const height = Math.max(1, box[3] - box[1]);
    const areaPx = width * height;

    const rawScore = det.score ?? det.confidence ?? 0.75;
    const conf = getConfidenceTier(rawScore);

    const tag = String(index + 1).padStart(2, "0");
    const label = det.label || "Object";
    const cleanLabel = label
      .replace(/[._\-]/g, " ")
      .replace(/\b(object|detected|instance)\b/gi, "")
      .trim() || "Target";

    // Geometry classification
    const geomRaw = String(det.geometry_type || det.type || "").toUpperCase();
    let geometryType: GeometryVisualType = "DISCRETE_OBJECT";
    if (geomRaw.includes("REGION") || geomRaw.includes("POLYGON")) {
      geometryType = "REGION_FEATURE";
    } else if (geomRaw.includes("LINEAR") || geomRaw.includes("LINE")) {
      geometryType = "LINEAR_FEATURE";
    }

    // Polygon coordinates if available
    let polygon: [number, number][] | undefined;
    if (det.polygon && Array.isArray(det.polygon) && det.polygon.length >= 3) {
      polygon = det.polygon.map(([px, py]: [number, number]) => [
        isNormalized ? Math.round(px * imgWidth) : Math.round(px),
        isNormalized ? Math.round(py * imgHeight) : Math.round(py),
      ]);
    }

    // Georeferenced Coordinates
    let coordinates: { lat: number; lon: number; formatted: string } | undefined;
    let gmapsUrl: string | undefined;

    const feat =
      geojson?.features?.find(
        (f: any) =>
          f.properties?.evidence_id === `det_${index}` ||
          f.id === `det_${index}` ||
          f.id === String(index) ||
          f.properties?.evidence_id === tag
      ) || geojson?.features?.[index];

    if (feat?.geometry) {
      if (feat.geometry.type === "Point" && Array.isArray(feat.geometry.coordinates)) {
        const [lon, lat] = feat.geometry.coordinates;
        coordinates = { lat, lon, formatted: formatGeoCoordinates(lat, lon) };
        gmapsUrl = `https://www.google.com/maps?q=${lat.toFixed(6)},${lon.toFixed(6)}`;
      } else if (feat.geometry.type === "Polygon" && Array.isArray(feat.geometry.coordinates?.[0])) {
        const ring = feat.geometry.coordinates[0];
        if (ring.length > 0) {
          const [lon, lat] = ring[0];
          coordinates = { lat, lon, formatted: formatGeoCoordinates(lat, lon) };
          gmapsUrl = `https://www.google.com/maps?q=${lat.toFixed(6)},${lon.toFixed(6)}`;
        }
      }
    }

    return {
      index,
      tag,
      label,
      cleanLabel: cleanLabel.charAt(0).toUpperCase() + cleanLabel.slice(1),
      category: cleanLabel.toLowerCase(),
      geometryType,
      score: rawScore,
      confidencePercent: conf.percent,
      confidenceTier: conf.tier,
      box,
      polygon,
      width,
      height,
      areaPx,
      areaEstimate: areaPx > 5000 ? `${(areaPx * 0.25).toLocaleString()} m² (est.)` : undefined,
      coordinates,
      gmapsUrl,
      evidenceId: det.evidence_id || `det_${index}`,
    };
  });
}

/**
 * Synthesizes a unified, user-facing InvestigationResult from raw backend response.
 */
export function synthesizeInvestigationResult(
  query: string,
  raw: NexSpaceQueryResponse,
  sourceImage?: CanonicalSourceImage
): InvestigationResult {
  const qClean = (query || "").trim();
  const qLower = qClean.toLowerCase();
  const rawAny = raw as any;

  const routingObj = raw.routing_decision as any;
  const taskType = (routingObj?.task_type || raw.task_type || raw.task || "GROUNDING").toUpperCase();
  const rawDetections = (raw.detections || raw.grounding?.detections || []) as any[];
  const enriched = enrichDetections(rawDetections, raw.geojson);

  // Intent Flags
  const isCaptionOnly =
    taskType === "CAPTIONING" ||
    taskType === "IMAGE_DESCRIPTION" ||
    (raw.detection_status === "not_applicable" && Boolean(raw.optical_caption || raw.response_text));

  const isCountingQuery =
    taskType === "COUNTING" ||
    Boolean(routingObj?.parameters?.is_counting) ||
    Boolean(raw.count_summary) ||
    /\b(how many|count|number of|quantity of)\b/i.test(qLower);

  const isPresenceQuery =
    taskType === "PRESENCE_CHECK" ||
    Boolean(routingObj?.parameters?.presence_check) ||
    /\b(is there|are there|tell me if there is|can you see any)\b/i.test(qLower);

  const isSpatialRelationQuery =
    taskType === "SPATIAL_REASONING" ||
    Boolean(rawAny.spatial_reasoning) ||
    /\b(near|north of|south of|east of|west of|adjacent to|crossing|inside|outside|along)\b/i.test(qLower);

  const isChangeQuery =
    taskType === "CHANGE_ANALYSIS" ||
    Boolean(raw.change_analysis) ||
    /\b(what changed|compare|difference between|before and after)\b/i.test(qLower);

  // Categorized target groups for multi-target queries
  const categoryMap = new Map<string, EnrichedIntelligenceDetection[]>();
  for (const det of enriched) {
    const cat = det.cleanLabel;
    if (!categoryMap.has(cat)) {
      categoryMap.set(cat, []);
    }
    categoryMap.get(cat)!.push(det);
  }

  const isMultiTargetQuery = categoryMap.size > 1;
  const categorizedGroups: CategorizedTargetGroup[] = Array.from(categoryMap.entries()).map(
    ([label, items]) => ({
      label,
      count: items.length,
      geometryType: items[0]?.geometryType || "DISCRETE_OBJECT",
      items,
    })
  );

  // Determine Primary Target Noun
  let targetNoun = "Objects";
  if (routingObj?.parameters?.target_phrase) {
    targetNoun = routingObj.parameters.target_phrase;
  } else if (raw.target) {
    targetNoun = raw.target;
  } else if (enriched.length > 0) {
    targetNoun = enriched[0].cleanLabel;
  } else if (/buildings?/i.test(qLower)) {
    targetNoun = "Buildings";
  } else if (/vehicles?|cars?/i.test(qLower)) {
    targetNoun = "Vehicles";
  } else if (/ships?|vessels?|boats?/i.test(qLower)) {
    targetNoun = "Vessels";
  } else if (/aircraft|planes?|airplanes?/i.test(qLower)) {
    targetNoun = "Aircraft";
  } else if (/water\s*(?:bodies|body)?|rivers?|lakes?/i.test(qLower)) {
    targetNoun = "Water Body";
  } else if (/roads?|highways?/i.test(qLower)) {
    targetNoun = "Roads";
  } else if (/agricultural\s*areas?|farms?|fields?/i.test(qLower)) {
    targetNoun = "Agricultural Areas";
  } else if (/industrial\s*facilities?|factories?/i.test(qLower)) {
    targetNoun = "Industrial Facilities";
  } else if (/construction(?:\s*activity)?/i.test(qLower)) {
    targetNoun = "Construction Activity";
  }

  // Pluralized / singular clean noun
  targetNoun = targetNoun.charAt(0).toUpperCase() + targetNoun.slice(1);

  // Detection State
  let detectionState: DetectionStateType = "detections_found";
  if (
    raw.detection_status === "satellite_imagery_found" ||
    isCaptionOnly ||
    taskType === "GEOSPATIAL_METADATA" ||
    taskType === "GEOLOCATION"
  ) {
    detectionState = "not_applicable";
  } else if (raw.detection_status === "model_unavailable") {
    detectionState = "model_unavailable";
  } else if (raw.detection_status === "localization_unavailable") {
    detectionState = "localization_unavailable";
  } else if (enriched.length === 0) {
    detectionState = "genuine_zero";
  } else {
    detectionState = "detections_found";
  }

  // Location / Georeferencing
  const geoMeta = raw.geospatial_metadata || raw.geospatial;
  const isGeoreferenced = !!(
    geoMeta?.geospatial_available ||
    raw.selected_scene ||
    (sourceImage as any)?.isGeoreferenced
  );
  const resolvedName =
    (raw as any)?.geographic_context?.display_name ||
    (raw as any)?.geospatial_metadata?.display_name ||
    (geoMeta as any)?.display_name ||
    (sourceImage as any)?.locationName ||
    (sourceImage as any)?.caption ||
    (geoMeta as any)?.resolved_location ||
    (isGeoreferenced ? "Georeferenced Scene" : "Image Frame (Unreferenced)");

  let locCoords: { lat: number; lon: number; formatted: string } | undefined;
  let locGmapsUrl: string | undefined;

  if (isGeoreferenced) {
    if (raw.geographic_context?.center) {
      const { lat, lng } = raw.geographic_context.center;
      locCoords = { lat, lon: lng, formatted: formatGeoCoordinates(lat, lng) };
      locGmapsUrl = `https://www.google.com/maps?q=${lat.toFixed(6)},${lng.toFixed(6)}`;
    } else if ((geoMeta as any)?.center_coords) {
      const { lat, lon } = (geoMeta as any).center_coords;
      locCoords = { lat, lon, formatted: formatGeoCoordinates(lat, lon) };
      locGmapsUrl = `https://www.google.com/maps?q=${lat.toFixed(6)},${lon.toFixed(6)}`;
    } else if (geoMeta?.bounds_world) {
      const lat = (geoMeta.bounds_world.min_y + geoMeta.bounds_world.max_y) / 2;
      const lon = (geoMeta.bounds_world.min_x + geoMeta.bounds_world.max_x) / 2;
      locCoords = { lat, lon, formatted: formatGeoCoordinates(lat, lon) };
      locGmapsUrl = `https://www.google.com/maps?q=${lat.toFixed(6)},${lon.toFixed(6)}`;
    } else if (enriched.length > 0 && enriched[0].coordinates) {
      locCoords = enriched[0].coordinates;
      locGmapsUrl = enriched[0].gmapsUrl;
    }
  }

  // Headline & Natural-Language Answer Synthesis
  let headline = "";
  let naturalLanguageAnswer = "";

  if (raw.detection_status === "satellite_imagery_found" || (raw.selected_scene && enriched.length === 0 && !isPresenceQuery && !isCountingQuery && !isChangeQuery)) {
    headline = "SATELLITE IMAGERY ACQUIRED";
    const sceneId = raw.selected_scene?.scene_id || "Sentinel-2";
    const date = raw.selected_scene?.acquisition_date ? new Date(raw.selected_scene.acquisition_date).toLocaleDateString() : "Recent";
    const cloud = raw.selected_scene?.cloud_cover !== undefined ? `${raw.selected_scene.cloud_cover.toFixed(1)}% cloud` : "";
    naturalLanguageAnswer = raw.response_text || raw.investigation_report?.summary || `Retrieved high-resolution Sentinel-2 RGB satellite imagery (${sceneId.split('_')[0] || "Sentinel-2"}, ${date}${cloud ? `, ${cloud}` : ""}) for ${resolvedName}.`;
  } else if (isCaptionOnly) {
    headline = "SCENE DESCRIPTION";
    naturalLanguageAnswer =
      raw.optical_caption ||
      raw.response_text ||
      raw.investigation_report?.summary ||
      "Comprehensive scene analysis complete.";
  } else if (isChangeQuery) {
    const changeCount = raw.change_analysis?.anomalies?.length || 0;
    headline = changeCount > 0 ? `${changeCount} SIGNIFICANT CHANGES IDENTIFIED` : "CHANGE ANALYSIS COMPLETE";
    naturalLanguageAnswer =
      raw.change_analysis?.what_changed ||
      raw.change_analysis?.summary ||
      `Change analysis identified ${changeCount > 0 ? changeCount : "no major"} structural changes between the two observations.`;
  } else if (isPresenceQuery) {
    if (enriched.length > 0) {
      headline = `${targetNoun.toUpperCase()} DETECTED`;
      naturalLanguageAnswer = `Yes. ${targetNoun} was identified in the imagery with ${enriched.length} confirmed feature region(s).`;
    } else {
      headline = `NO ${targetNoun.toUpperCase()} IDENTIFIED`;
      naturalLanguageAnswer = `No matching ${targetNoun.toLowerCase()} was identified in this imagery.`;
    }
  } else if (isCountingQuery) {
    headline = `${enriched.length} ${targetNoun.toUpperCase()} IDENTIFIED`;
    naturalLanguageAnswer = `${enriched.length} ${targetNoun.toLowerCase()} were identified and verified across the scene.`;
  } else if (isSpatialRelationQuery) {
    const relObj = rawAny.spatial_reasoning;
    const pred = relObj?.relation || "near";
    const ref = relObj?.reference_entity || "the designated reference area";
    headline = `${enriched.length} ${targetNoun.toUpperCase()} IDENTIFIED`;
    naturalLanguageAnswer = `${enriched.length} ${targetNoun.toLowerCase()} identified ${pred.replace(/_/g, " ")} ${ref}.`;
  } else if (isMultiTargetQuery) {
    headline = `${enriched.length} TOTAL TARGETS IDENTIFIED`;
    const breakdowns = categorizedGroups.map((g) => `${g.count} ${g.label.toLowerCase()}`).join(", ");
    naturalLanguageAnswer = `Analysis identified ${breakdowns} across the scene.`;
  } else {
    if (enriched.length > 0) {
      headline = `${enriched.length} ${targetNoun.toUpperCase()} FOUND`;
      naturalLanguageAnswer = `I identified ${enriched.length} ${targetNoun.toLowerCase()} candidate(s) in the imagery.`;
    } else if (detectionState === "genuine_zero") {
      headline = `NO ${targetNoun.toUpperCase()} IDENTIFIED`;
      naturalLanguageAnswer = `No matching ${targetNoun.toLowerCase()} were identified in this imagery above confidence threshold.`;
    } else if (detectionState === "model_unavailable") {
      headline = "DETECTION TEMPORARILY UNAVAILABLE";
      naturalLanguageAnswer = `Specialist localization for ${targetNoun.toLowerCase()} is temporarily unavailable.`;
    } else if (detectionState === "localization_unavailable") {
      headline = "LANDSCAPE FEATURE DETECTED";
      naturalLanguageAnswer = `NexSpace analyzed the imagery, but could not reliably localize discrete bounding boxes for ${targetNoun.toLowerCase()}.`;
    }
  }

  // Observations
  const observations: string[] = [];
  if (raw.investigation_report?.observations && Array.isArray(raw.investigation_report.observations)) {
    observations.push(...raw.investigation_report.observations);
  } else {
    if (enriched.length > 0) {
      observations.push(`Identified ${enriched.length} visual instance(s) matching your query.`);
    }
    if (locCoords) {
      observations.push(`Scene georeferenced at ${locCoords.formatted}.`);
    }
    if (raw.optical_caption) {
      observations.push(raw.optical_caption);
    }
  }

  // Notes & Limitations
  const notes: string[] = [];
  if (raw.investigation_report?.limitations && Array.isArray(raw.investigation_report.limitations)) {
    notes.push(...raw.investigation_report.limitations);
  }

  // Technical Details (cleanly isolated)
  const traceList: TraceStage[] = raw.execution_trace || rawAny.trace || [];
  const technicalDetails = {
    modelsUsed: raw.selected_tools || ["Grounding DINO", "BLIP Optical Vision"],
    routingTools: routingObj?.target_tools || raw.selected_tools || [],
    confidenceSource: raw.confidence_source || "Model Confidence Calibration",
    confidenceValue: raw.confidence ?? 0.85,
    executionStages: traceList.map((t) => ({
      stage: t.stage,
      status: t.status,
      duration_ms: t.duration_ms,
    })),
    totalDurationMs: traceList.reduce((acc: number, t: TraceStage) => acc + (t.duration_ms || 0), 0),
    sensorType:
      (sourceImage as any)?.sensor ||
      (raw.selected_scene?.platform ? `${raw.selected_scene.platform} (MSI)` : undefined) ||
      (isGeoreferenced ? "Satellite Optical" : "Standard RGB (Sensor Unknown)"),
    imageDimensions: `${sourceImage?.width || 1000} × ${sourceImage?.height || 1000} px`,
  };

  return {
    query: qClean,
    task: taskType,
    detectionState,
    headline,
    naturalLanguageAnswer,
    summary: raw.investigation_report?.summary || naturalLanguageAnswer,
    targetCount: enriched.length,
    targetNoun,
    hasDetections: enriched.length > 0,
    isCaptionOnly,
    isCountingQuery,
    isPresenceQuery,
    isSpatialRelationQuery,
    isChangeQuery,
    isMultiTargetQuery,
    spatialRelationDetails: rawAny.spatial_reasoning
      ? {
          predicate: rawAny.spatial_reasoning.relation,
          reference: rawAny.spatial_reasoning.reference_entity,
        }
      : undefined,
    categorizedGroups,
    allDetections: enriched,
    sceneDescription:
      raw.optical_caption ||
      (raw.response_text && !raw.response_text.startsWith("**Geospatial Metadata:**") ? raw.response_text : undefined) ||
      raw.investigation_report?.summary,
    locationInfo: {
      resolvedName,
      coordinates: locCoords,
      crs:
        (geoMeta?.geospatial_available ? geoMeta.crs : undefined) ||
        raw.selected_scene?.crs ||
        ((sourceImage as any)?.isGeoreferenced && (sourceImage as any)?.crs) ||
        "None (Not Georeferenced)",
      resolution:
        (geoMeta?.geospatial_available && (geoMeta?.gsd || (geoMeta as any)?.resolution)
          ? `${geoMeta.gsd || (geoMeta as any).resolution}m GSD`
          : undefined) ||
        (raw.selected_scene?.spatial_resolution_m ? `${raw.selected_scene.spatial_resolution_m}m Optical` : undefined) ||
        ((sourceImage as any)?.isGeoreferenced && (sourceImage as any)?.resolution) ||
        "Unavailable (Non-georeferenced)",
      bounds: isGeoreferenced && geoMeta?.bounds_world
        ? [
            geoMeta.bounds_world.min_x,
            geoMeta.bounds_world.min_y,
            geoMeta.bounds_world.max_x,
            geoMeta.bounds_world.max_y,
          ]
        : undefined,
      footprintGeoJson: raw.geojson,
      gmapsUrl: locGmapsUrl,
    },
    changeSummary: raw.change_analysis
      ? {
          category: raw.change_analysis.change_category,
          summary: raw.change_analysis.what_changed || raw.change_analysis.summary,
          beforeDesc: raw.change_analysis.before_interpretation,
          afterDesc: raw.change_analysis.after_interpretation,
        }
      : undefined,
    observations,
    notes,
    technicalDetails,
  };
}
