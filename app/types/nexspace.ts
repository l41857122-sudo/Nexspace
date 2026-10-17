/**
 * app/types/nexspace.ts
 * ---------------------
 * Centralized TypeScript type definitions mirroring the NexSpace FastAPI Backend Schemas.
 */

export interface GeoSpatialResolution {
  x: number;
  y: number;
  unit: string;
}

export interface GeoSpatialBounds {
  min_x: number;
  min_y: number;
  max_x: number;
  max_y: number;
}

export interface GeoMetadata {
  geospatial_available: boolean;
  crs?: string | null;
  crs_type?: "geographic" | "projected" | null;
  crs_epsg?: number | null;
  crs_units?: string | null;
  transform?: {
    a: number;
    b: number;
    c: number;
    d: number;
    e: number;
    f: number;
  } | null;
  resolution?: GeoSpatialResolution | null;
  bounds_world?: GeoSpatialBounds | null;
  dimensions_pixel?: {
    width: number;
    height: number;
  } | null;
  reason?: string | null;
  gsd?: number | string | null;
}

export interface GeoJSONGeometry {
  type: "Polygon" | "Point" | "MultiPolygon" | "LineString" | "MultiLineString";
  coordinates: any;
}

export interface GeoJSONFeature {
  type: "Feature";
  id?: string;
  geometry: GeoJSONGeometry;
  properties: {
    evidence_id?: string;
    type?: string;
    label?: string;
    source?: string;
    confidence?: number;
    ground_area?: number;
    ground_area_unit?: string;
    native_crs?: string;
    bbox_pixel?: number[];
    [key: string]: unknown;
  };
}

export interface GeoJSONFeatureCollection {
  type: "FeatureCollection";
  geospatial_available: boolean;
  crs?: {
    type: string;
    properties: {
      name: string;
    };
  };
  features: GeoJSONFeature[];
}

export interface TraceStage {
  stage: string;
  stage_id?: string;
  stage_name?: string;
  status: "started" | "completed" | "failed" | "skipped" | "partial_success" | "unavailable" | "pending" | "running";
  started_at: string;
  completed_at?: string | null;
  timestamp?: string;
  duration_ms: number;
  metadata?: Record<string, unknown>;
  details?: Record<string, unknown>;
  provenance?: Record<string, unknown>;
  error?: string | null;
  error_category?: string | null;
}

export interface ToolExecutionRecord {
  tool_name: string;
  status: "pending" | "running" | "succeeded" | "failed" | "skipped" | "unavailable" | "degraded";
  started_at: string;
  completed_at?: string | null;
  duration_ms: number;
  input_ref?: Record<string, unknown>;
  output_summary?: Record<string, unknown>;
  provenance?: Record<string, unknown>;
  error?: string | null;
  error_category?: string | null;
  actionable_reason?: string | null;
}

export interface StructuredLogEvent {
  timestamp: string;
  level: "DEBUG" | "INFO" | "WARNING" | "ERROR" | "CRITICAL";
  logger: string;
  event: string;
  correlation_id: string;
  stage?: string;
  error_category?: string;
  actionable_reason?: string;
  details?: Record<string, unknown>;
}

export interface StructuredExecutionTrace {
  investigation_id: string;
  request_id: string;
  query_id?: string;
  query?: string;
  started_at: string;
  completed_at?: string;
  total_duration_ms: number;
  overall_outcome: string;
  error_category?: string;
  actionable_reason?: string;
  validation_outcome?: { passed: boolean; reason?: string | null };
  query_metadata?: Record<string, unknown>;
  data_sources?: Array<{ source_id: string; sensor_type: string; metadata?: Record<string, unknown> }>;
  stages: TraceStage[];
  tool_executions: ToolExecutionRecord[];
  logs: StructuredLogEvent[];
}

export interface EvidenceNode {
  evidence_id: string;
  type: "object_detection" | "change_region" | "vqa_finding" | "optical_caption" | "sar_caption" | "multimodal_fusion" | string;
  source_tool: string;
  source_model: string;
  derived_from: string[];
  payload: {
    box?: number[];
    bbox_pixel?: number[];
    bbox_normalized?: number[];
    bbox_world?: {
      min_x: number;
      min_y: number;
      max_x: number;
      max_y: number;
      crs: string;
      polygon_world: Array<[number, number]>;
    };
    ground_area?: number;
    label?: string;
    score?: number;
    confidence?: number;
    text?: string;
    answer?: string;
    question?: string;
    [key: string]: unknown;
  };
  confidence?: number | null;
  confidence_type?: "model" | "heuristic" | "unavailable";
  confidence_source?: string;
  validation_status: "valid" | "invalid" | "unverified";
  validation_error?: string | null;
}

export interface EvidenceGraph {
  query_id: string;
  nodes: EvidenceNode[];
  edges: Array<{
    source_id: string;
    target_id: string;
    relation: string;
  }>;
}

export interface SpatialSummary {
  geospatial_available: boolean;
  crs?: string | null;
  evidence_with_coordinates?: number;
  total_ground_area?: number;
  total_ground_area_unit?: string;
  area_unit?: string;
}

export interface ExecutionSummary {
  total_duration_ms?: number;
  stages_executed?: number;
  tools_executed?: string[];
  fallback_count?: number;
}

export interface InvestigationReport {
  summary: string;
  observations: string[];
  interpretations: string[];
  evidence_references: string[];
  limitations: string[];
  spatial_summary?: SpatialSummary;
  execution_summary?: ExecutionSummary;
  plan?: {
    task_type: string;
    selected_tools: string[];
    parameters: Record<string, unknown>;
  };
}

export interface VQAFinding {
  question: string;
  answer: string;
  confidence: number | null;
  low_confidence?: boolean;
}

export interface GroundingDetection {
  box_2d: number[];
  label: string;
  score: number;
  bbox_pixel?: number[];
  bbox_normalized?: number[];
  bbox_world?: {
    min_x: number;
    min_y: number;
    max_x: number;
    max_y: number;
    crs: string;
  };
}

export interface GroundingResult {
  target_phrase: string;
  detections: GroundingDetection[];
  num_detections: number;
}

export interface AnomalyRegion {
  id: string;
  label: string;
  type?: string;
  bbox_pixel: number[];
  bbox_normalized: number[];
  bbox_world?: {
    min_x: number;
    min_y: number;
    max_x: number;
    max_y: number;
    crs: string;
  };
  ground_area?: number;
  area_unit?: string;
  severity?: "high" | "medium" | "low" | string;
  confidence?: number;
  mean_diff?: number;
  max_diff?: number;
}

export interface AnomalySummary {
  total_anomalies: number;
  high_severity: number;
  medium_severity: number;
  low_severity: number;
  total_area_m2?: number;
  threshold_used?: number;
}

export interface ChangeAnalysisResult {
  summary: string;
  what_changed?: string;
  change_category?: "PIXEL CHANGE" | "OBJECT CHANGE" | "SEMANTIC CHANGE" | string;
  before_interpretation?: string;
  after_interpretation?: string;
  change_vqa_answer?: string;
  changed_fraction: number;
  mean_intensity_delta: number;
  overlay_image?: string | null;
  heatmap_image?: string | null;
  anomalies?: AnomalyRegion[];
  anomaly_summary?: AnomalySummary;
  model_provenance?: string;
  scene_a_id?: string;
  scene_b_id?: string;
  acquisition_date_a?: string;
  acquisition_date_b?: string;
  sensor_a?: string;
  sensor_b?: string;
  valid_pixels_fraction?: number;
  nodata_pixels_count?: number;
  cloud_contamination_detected?: boolean;
  cloud_pixels_fraction?: number;
  alignment_status?: string;
  limitations?: string[];
}

export interface OpticalSARAnalysisResult {
  optical_features_dim?: number;
  sar_features_dim?: number;
  joint_representation_dim?: number;
  fused_feature_dim?: number;
  cosine_similarity?: number;
  fusion_type?: string;
  fusion_operation?: string;
  fusion_learning?: string;
  optical_evidence?: string;
  sar_evidence?: string;
  fused_conclusion?: string;
  cross_modal_summary?: string;
  correlation_summary?: string;
  alignment_status?: string;
  is_trained_model?: boolean;
  model_provenance?: string;
  limitations?: string[];
  optical_only_metrics?: Record<string, unknown>;
  sar_only_metrics?: Record<string, unknown>;
  combined_metrics?: Record<string, unknown>;
  source_contributions?: Record<string, string>;
  preprocessing_status?: Record<string, unknown>;
}

export interface NexSpaceQueryResponse {
  request_id: string;
  status: "completed" | "error" | "validation_error" | "internal_error" | "degraded" | "failed";
  provenance?: "MODEL_DERIVED" | "ALGORITHMIC" | "HEURISTIC" | "FALLBACK" | "UNAVAILABLE" | string;
  model_loaded?: boolean;
  fallback_used?: boolean;
  provenance_contract?: Record<string, unknown>;
  query: string;
  task_type?: string;
  intent: string;
  parameters?: Record<string, unknown>;
  plan?: {
    task_type: string;
    target_tools: string[];
    parameters: Record<string, unknown>;
    execution_strategy: string;
  };
  selected_tools?: string[];
  routing_decision?: {
    target_tools: string[];
    restructured_vqa_queries?: string[];
    requires_count_warning?: boolean;
    execution_reasoning?: string;
  };
  results?: Record<string, unknown>;
  evidence?: EvidenceNode[];
  evidence_graph?: EvidenceGraph;
  investigation_report?: InvestigationReport;
  spatial_summary?: SpatialSummary;
  execution_trace?: TraceStage[];
  structured_trace?: StructuredExecutionTrace;
  execution_logs?: StructuredLogEvent[];
  confidence?: number | null;
  confidence_type?: "model" | "heuristic" | "unavailable";
  confidence_source?: string;
  model_provenance?: string;
  registration_quality?: string;
  fallback_count?: number;
  limitations?: string[];
  geojson?: GeoJSONFeatureCollection;
  geospatial?: GeoMetadata;
  geospatial_metadata?: GeoMetadata;
  grounding?: GroundingResult;
  detections?: unknown[];
  optical_caption?: string | null;
  sar_caption?: string | null;
  vqa_results?: VQAFinding[];
  change_analysis?: ChangeAnalysisResult | null;
  optical_sar_analysis?: OpticalSARAnalysisResult | null;
  semantic_change?: Record<string, unknown> | null;
  response_text?: string;
  source_image?: CanonicalSourceImage | null;
  satellite_image?: CanonicalSourceImage | null;
  backend_status?: "live_backend" | "offline_fallback";
  task?: string;
  target?: string | null;
  detection_status?: string;
  count?: number | null;
  count_summary?: {
    count: number;
    target: string;
    reliability: string;
    message: string;
  } | null;
  semantic_plan?: Record<string, unknown> | null;
  geographic_context?: {
    bounds_world?: GeoSpatialBounds | null;
    center?: { lat: number; lng: number } | null;
    primary_source?: string;
    has_detections?: boolean;
    features_count?: number;
  } | null;
  selected_scene?: DiscoveredSatelliteScene | null;
  scenes?: DiscoveredSatelliteScene[];
  satellite_image_url?: string | null;
  satellite_image_base64?: string | null;
  google_maps_overlay?: {
    image_url?: string | null;
    image_base64?: string | null;
    bounds?: { west: number; south: number; east: number; north: number } | null;
    crs?: string;
    aoi_clipped?: boolean;
    cloud_cover?: number | null;
    quality_warning?: string | null;
  } | null;
  result_category?: string;
  resolution_status?: string;
  exact_locality_verified?: boolean;
}


export interface NexSpaceChangeAnalysisResponse {
  request_id?: string;
  status: "success" | "error";
  summary: string;
  changed_fraction: number;
  mean_intensity_delta?: number;
  overlay_image?: string | null;
  heatmap_image?: string | null;
  anomalies: AnomalyRegion[];
  anomaly_summary?: AnomalySummary;
  evidence?: EvidenceNode[];
  geojson?: GeoJSONFeatureCollection;
  geospatial?: GeoMetadata;
  backend_status?: "live_backend" | "offline_fallback" | "online" | string;
  scene_a_id?: string;
  scene_b_id?: string;
  acquisition_date_a?: string;
  acquisition_date_b?: string;
  sensor_a?: string;
  sensor_b?: string;
  valid_pixels_fraction?: number;
  nodata_pixels_count?: number;
  cloud_contamination_detected?: boolean;
  cloud_pixels_fraction?: number;
  alignment_status?: string;
  limitations?: string[];
}

export interface NexSpaceHealthResponse {
  request_id?: string;
  service?: string;
  version?: string;
  ml_backend_url?: string;
  status: "ok" | "degraded" | "error";
  timestamp?: string;
  uptime_seconds?: number;
  capabilities: {
    captioning: string;
    grounding: string;
    vqa: string;
    change_analysis: string;
    anomaly_extraction: string;
    optical_sar_fusion: string;
    geospatial: string;
  };
  models?: {
    blip_captioning: string;
    grounding_dino: string;
    paligemma_vqa: string;
  };
}

export interface CanonicalSourceImage {
  id: string;
  filename: string;
  mediaType: string;
  dataUrl: string;
  source: "upload" | "demo" | "satellite";
  sha256?: string;
  uploadedAt?: string;
  width?: number;
  height?: number;
  caption?: string;
  locationName?: string;
}

export interface CanonicalInvestigationState {
  investigation_id: string;
  query: string;
  inputMode?: "SINGLE_IMAGE" | "BI_TEMPORAL" | "OPTICAL_SAR" | "SATELLITE_DISCOVERY";
  source_image: CanonicalSourceImage;
  sar_image?: string | null;
  change_image_b?: CanonicalSourceImage | null;
  selectedTargetId?: string | null;
  timestamp: string;
  response: NexSpaceQueryResponse | null;
  selectedSatelliteScene?: DiscoveredSatelliteScene | null;
}

export interface GeocodingResult {
  query: string;
  display_name: string;
  latitude: number;
  longitude: number;
  bbox: number[];
  source: string;
  geocoding_available: boolean;
  error?: string | null;
}

export interface DiscoveredSatelliteScene {
  scene_id: string;
  provider: string;
  platform: string;
  collection: string;
  acquisition_date: string;
  sensing_time: string;
  cloud_cover: number;
  spatial_resolution_m: number;
  processing_level: string;
  bbox: number[];
  geometry: GeoJSONGeometry;
  crs: string;
  available_bands: string[];
  spectral_source: string;
  thumbnail_url?: string | null;
  visual_asset_url?: string | null;
  selected_reason: string;
  ranking_score: number;
}

export interface DiscoveryPlan {
  query: string;
  location_raw?: string | null;
  geocoding?: GeocodingResult | null;
  temporal: {
    raw_text: string;
    is_bitemporal: boolean;
    datetime_range?: string | null;
    datetime_before?: string | null;
    datetime_after?: string | null;
    description: string;
  };
  imagery: {
    sensor: string;
    modality: string;
    max_cloud_cover: number;
    resolution_preference: string;
    collection: string;
  };
  analysis_task: string;
  target_concepts: string[];
  spatial_relation?: {
    relation: string;
    reference_entity: string;
    raw_phrase: string;
  } | null;
}

export interface ImagerySearchResponse {
  request_id: string;
  status: "success" | "location_unresolved" | "SATELLITE_DATA_UNAVAILABLE" | string;
  plan?: DiscoveryPlan;
  count: number;
  scenes: DiscoveredSatelliteScene[];
  trace?: TraceStage[];
  reason?: string;
  actions?: string[];
  message?: string;
}

export interface ImageryRetrieveResponse {
  request_id: string;
  status: "success" | "SATELLITE_DATA_UNAVAILABLE" | string;
  scene: {
    scene_id: string;
    image_path: string;
    image_base64: string;
    width: number;
    height: number;
    crs: string;
    transform: {
      a: number;
      b: number;
      c: number;
      d: number;
      e: number;
      f: number;
    };
    resolution: GeoSpatialResolution;
    bounds_world: GeoSpatialBounds;
    available_bands: string[];
    spectral_source: string;
    rgb_rendered_for_model: boolean;
    provenance: Record<string, unknown>;
  };
  investigation?: NexSpaceQueryResponse | null;
  trace?: TraceStage[];
  reason?: string;
  actions?: string[];
}

export interface ResolvedLocation {
  query: string;
  displayName?: string;
  display_name?: string;
  latitude?: number;
  longitude?: number;
  bbox?: {
    west: number;
    south: number;
    east: number;
    north: number;
  } | null;
  bbox_list?: number[] | null; // [min_lon, min_lat, max_lon, max_lat]
  source:
  | "EXPLICIT_COORDINATES"
  | "GEOCODED_QUERY"
  | "IMAGE_METADATA"
  | "SCENE_METADATA"
  | "MODEL_ESTIMATE"
  | "UNRESOLVED";
  confidence?: number;
  geocoding_available?: boolean;
  error?: string | null;
}

export interface MapState {
  center: {
    lat: number;
    lng: number;
  };
  zoom: number;
  footprint?: GeoJSONFeature;
  detections?: GeoJSONFeature[];
  regions?: GeoJSONFeature[];
  linearFeatures?: GeoJSONFeature[];
  aoi?: GeoJSONFeature;
  location?: ResolvedLocation;
}

export type MapStatus =
  | "LOADING"
  | "READY"
  | "NO_LOCATION"
  | "NO_API_KEY"
  | "API_ERROR"
  | "GEOCODING_ERROR"
  | "NO_GEOMETRY"
  | "SCENE_ONLY"
  | "DETECTIONS_VISIBLE";


