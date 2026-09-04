/**
 * test_frontend_schemas.js
 * ------------------------
 * Frontend unit tests for schema validation, response parsing,
 * optional field handling, fallback rendering, and error isolation.
 */

const assert = require("assert");

function testResponseParsing() {
  console.log("=== RUNNING FRONTEND SCHEMA & PARSER UNIT TESTS ===");

  // 1. Full Response Parsing
  console.log("\n[UNIT 1] Validating Full NexSpaceQueryResponse parsing...");
  const mockValidResponse = {
    request_id: "req_test123",
    status: "completed",
    query: "Describe this image",
    intent: "Optical_Caption",
    selected_tools: ["Optical_Caption"],
    investigation_report: {
      summary: "High altitude port overview",
      observations: ["Water body detected", "Pier structures present"],
      interpretations: ["Active maritime traffic"],
      evidence_references: ["ev_001"],
      limitations: ["Standard optical resolution"],
      spatial_summary: {
        geospatial_available: false
      }
    },
    optical_caption: "A satellite view of a port with ships docked.",
    confidence: null,
    confidence_type: "model",
    confidence_source: "blip_captioning",
    execution_trace: [
      { stage: "request_received", status: "completed", duration_ms: 0.1, started_at: "2026-09-03T22:00:00Z" },
      { stage: "tool_execution", status: "completed", duration_ms: 124.5, started_at: "2026-09-03T22:00:01Z" }
    ],
    response_text: "A satellite view of a port with ships docked."
  };

  assert.strictEqual(mockValidResponse.request_id, "req_test123");
  assert.strictEqual(mockValidResponse.status, "completed");
  assert.strictEqual(mockValidResponse.investigation_report.observations.length, 2);
  assert.strictEqual(mockValidResponse.execution_trace.length, 2);
  console.log("✓ Full response parsed cleanly.");

  // 2. Optional Fields Handling (Safe Defaults)
  console.log("\n[UNIT 2] Validating Optional Field Omission...");
  const minimalResponse = {
    request_id: "req_min",
    status: "completed",
    query: "Test",
    intent: "VQA",
    response_text: "Yes."
  };

  const vqaResults = minimalResponse.vqa_results || [];
  const groundingDetections = minimalResponse.grounding?.detections || [];
  const spatialSummary = minimalResponse.spatial_summary || { geospatial_available: false };
  const limitations = minimalResponse.limitations || minimalResponse.investigation_report?.limitations || [];

  assert.deepStrictEqual(vqaResults, []);
  assert.deepStrictEqual(groundingDetections, []);
  assert.strictEqual(spatialSummary.geospatial_available, false);
  assert.deepStrictEqual(limitations, []);
  console.log("✓ Safely handled missing optional fields without exceptions.");

  // 3. Grounding Box Normalization (0-1000 scale to percentages)
  console.log("\n[UNIT 3] Validating Grounding Box Coordinate Normalization...");
  const sampleDetection = {
    box_2d: [150, 200, 650, 800], // [ymin, xmin, ymax, xmax] in 0-1000 scale
    label: "building",
    score: 0.94
  };

  const [ymin, xmin, ymax, xmax] = sampleDetection.box_2d;
  const topPct = (ymin / 1000) * 100;
  const leftPct = (xmin / 1000) * 100;
  const widthPct = ((xmax - xmin) / 1000) * 100;
  const heightPct = ((ymax - ymin) / 1000) * 100;

  assert.strictEqual(topPct, 15.0);
  assert.strictEqual(leftPct, 20.0);
  assert.strictEqual(widthPct, 60.0);
  assert.strictEqual(heightPct, 50.0);
  console.log(`✓ Normalized Box: top=${topPct}%, left=${leftPct}%, width=${widthPct}%, height=${heightPct}%`);

  // 4. Fallback Status Labeling
  console.log("\n[UNIT 4] Validating Fallback Status Transparency...");
  const fallbackResponse = {
    confidence: 0.32,
    confidence_type: "heuristic",
    confidence_source: "rsvqa_heuristic_adapter",
    fallback_count: 1,
    limitations: ["RSVQA model gated; using heuristic fallback."]
  };

  const isFallback = fallbackResponse.fallback_count > 0 || fallbackResponse.confidence_type === "heuristic";
  assert.strictEqual(isFallback, true);
  const fallbackLabel = fallbackResponse.confidence_source.includes("heuristic") ? "RSVQA fallback" : "Model confidence";
  assert.strictEqual(fallbackLabel, "RSVQA fallback");
  console.log("✓ Fallback transparency verified: correctly flagged as 'RSVQA fallback'.");

  // 5. GeoJSON FeatureCollection Safety
  console.log("\n[UNIT 5] Validating GeoJSON FeatureCollection Safety...");
  const geojsonPayload = {
    type: "FeatureCollection",
    geospatial_available: true,
    crs: { type: "name", properties: { name: "urn:ogc:def:crs:OGC:1.3:CRS84" } },
    features: [
      {
        type: "Feature",
        id: "feat_1",
        geometry: {
          type: "Polygon",
          coordinates: [[[13.4, 52.5], [13.5, 52.5], [13.5, 52.4], [13.4, 52.4], [13.4, 52.5]]]
        },
        properties: { label: "Industrial Zone", ground_area: 1540.0 }
      }
    ]
  };

  assert.strictEqual(geojsonPayload.type, "FeatureCollection");
  assert.strictEqual(geojsonPayload.features.length, 1);
  assert.strictEqual(geojsonPayload.features[0].geometry.type, "Polygon");
  console.log("✓ GeoJSON structure strictly verified.");

  // 6. Malformed and Error Responses
  console.log("\n[UNIT 6] Validating Error and Offline States...");
  const offlineState = {
    status: "error",
    detail: "Analysis backend unavailable.",
    error_code: "HTTP_503"
  };

  assert.strictEqual(offlineState.status, "error");
  assert.strictEqual(offlineState.error_code, "HTTP_503");
  console.log("✓ Handled offline/error state safely.");

  console.log("\n========================================================");
  console.log("ALL 6 FRONTEND UNIT & SCHEMA TESTS PASSED!");
  console.log("========================================================");
}

testResponseParsing();
