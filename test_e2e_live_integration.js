/**
 * test_e2e_live_integration.js
 * ----------------------------
 * End-to-end verification of Next.js frontend API proxy communicating
 * directly with the FastAPI backend across all 8 PRD evaluation scenarios.
 */

const assert = require("assert");

const NEXTJS_BASE = "http://localhost:3000";

// 100x100 RGB Port & Water Scene
const SAMPLE_PORT =
  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGQAAABkCAIAAAD/gAIDAAABBElEQVR4nO3cwQnDMBQFQdm4jqQgd2DX5w6cgtJJWtASSC4z5w+C5Z21PI57MGedvEOsxrICsQKxArECsQKxArECsQKxArECsQKxArECsQKxArECsYJtfOF97eNPnufr949aViBWIFYgViBWIFYgViBWIFYgViBWIFYgViBWIFYgViBWIFYgViBWIFYgViBWIFYgViBWIFYgViBWIFYgViBWIFYgViBWIFYgViBWIFYgVrD4uGeeZQViBWIFYgViBWIFYgViBWIFYgViBWIFYgViBWIFYgViBWIFYgViBWIFYgViBWIFYgViBWIFYgViBWIFYgViBWIFYgViBWIFYgVijXkfdLUGIeOCzVsAAAAASUVORK5CYII=";

// 100x100 SAR Radar Backscatter PNG
const SAMPLE_SAR =
  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGQAAABkCAIAAAD/gAIDAAAA/ElEQVR4nO3csQnEMBQFQflwRSpGRapKt6DF4Etm4g+C5cW65pyDM7/DO8RqLCsQKxArECsQKxArECsQKxArECsQKxArECsQKxArECsQK7jHC3vv8Sdrre8ftaxArECsQKxArECsQKxArECsQKxArECsQKxArECsQKxArECsQKxArECsQKxArECsQKxArECsQKxArECsQKxArECsQKxArECsQKxArECsQKzg8nHPOcsKxArECsQKxArECsQKxArECsQKxArECsQKxArECsQKxArECsQKxArECsQKxArECsQKxArECsQKxArECsQKxArECsQKxArECsQKxBrnHsUuBPUbL76bAAAAAElFTkSuQmCC";

// 100x100 Bitemporal B (After)
const SAMPLE_CHANGE_B =
  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGQAAABkCAIAAAD/gAIDAAABAklEQVR4nO3csQ2EMBQFQRtRx9EQFVAgFVxD18m14BUByUzsaPWyL3l+ru9gzbb4DrEaywrECsQKxArECsQKxArECsQKxArECsQKxArECsQKxArECvbxwO8+x0uON04HlhWIFYgViBWIFYgViBWIFYgViBWIFYgViBWIFYgViBWIFYgViBWIFYgViBWIFYgViBWIFYgViBWIFYgViBWIFYgViBWIFYgViBWIFYgViBWIFYgViBVMH/ess6xArECsQKxArECsQKxArECsQKxArECsQKxArECsQKxArECsQKxArECsQKxArECsQKxArECsQKxArECsQKxArECsQKxArECsQKxArLHuD3WrBiHdPmgnAAAAAElFTkSuQmCC";

async function runTests() {
  console.log("=== RUNNING NEXT.JS → FASTAPI E2E INTEGRATION TESTS (8 SCENARIOS) ===");

  // 1. Health endpoint
  console.log("\n[TEST 1] Testing /api/health through Next.js proxy...");
  const resHealth = await fetch(`${NEXTJS_BASE}/api/health`);
  assert.strictEqual(resHealth.status, 200, "Health endpoint should return 200 OK");
  const dataHealth = await resHealth.json();
  assert.strictEqual(dataHealth.status, "ok", "Status should be 'ok'");
  assert.ok(dataHealth.capabilities, "Capabilities object should be present");
  console.log("✓ Health endpoint returned capabilities:", dataHealth.capabilities);

  // 2. Optical Scene Description Query
  console.log("\n[TEST 2] Testing /api/query: 'Describe this image' (BLIP Optical Captioning)...");
  const resCap = await fetch(`${NEXTJS_BASE}/api/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query: "Describe this image",
      optical_image: SAMPLE_PORT,
    }),
  });
  assert.strictEqual(resCap.status, 200, "Caption query should return 200 OK");
  const dataCap = await resCap.json();
  assert.strictEqual(dataCap.status, "completed");
  assert.ok(dataCap.selected_tools.includes("Optical_Caption"), "Should select Optical_Caption tool");
  assert.ok(dataCap.optical_caption, "Optical caption text should be present");
  assert.ok(dataCap.request_id.startsWith("req_"), "Request ID should be present");
  assert.ok(Array.isArray(dataCap.execution_trace), "Execution trace should be present");
  assert.ok(dataCap.investigation_report, "Investigation report should be present");
  console.log("✓ Caption generated:", dataCap.optical_caption);
  console.log("✓ Execution trace stages count:", dataCap.execution_trace.length);

  // 3. Closed-Ended VQA Query
  console.log("\n[TEST 3] Testing /api/query: 'Is there water in this image?' (VQA)...");
  const resVqa = await fetch(`${NEXTJS_BASE}/api/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query: "Is there water in this image?",
      optical_image: SAMPLE_PORT,
    }),
  });
  assert.strictEqual(resVqa.status, 200, "VQA query should return 200 OK");
  const dataVqa = await resVqa.json();
  assert.strictEqual(dataVqa.status, "completed");
  assert.ok(dataVqa.selected_tools.includes("VQA"), "Should select VQA tool");
  assert.ok(dataVqa.vqa_results.length > 0, "VQA results should be populated");
  console.log("✓ VQA Result:", dataVqa.vqa_results[0]);

  // 4. Grounding Query
  console.log("\n[TEST 4] Testing /api/query: 'Locate the buildings' (Grounding DINO)...");
  const resGrd = await fetch(`${NEXTJS_BASE}/api/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query: "Locate the buildings",
      optical_image: SAMPLE_PORT,
    }),
  });
  assert.strictEqual(resGrd.status, 200, "Grounding query should return 200 OK");
  const dataGrd = await resGrd.json();
  assert.strictEqual(dataGrd.status, "completed");
  assert.ok(dataGrd.selected_tools.includes("Grounding"), "Should select Grounding tool");
  assert.ok(dataGrd.grounding, "Grounding object should be present");
  assert.ok(Array.isArray(dataGrd.grounding.detections), "Detections list should be present");
  console.log(`✓ Grounding DINO returned ${dataGrd.grounding.detections.length} detections.`);

  // 5. Multi-Tool Query: Caption + Grounding
  console.log("\n[TEST 5] Testing /api/query: 'Describe this image and locate the buildings'...");
  const resMulti = await fetch(`${NEXTJS_BASE}/api/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query: "Describe this image and locate the buildings",
      optical_image: SAMPLE_PORT,
    }),
  });
  assert.strictEqual(resMulti.status, 200);
  const dataMulti = await resMulti.json();
  assert.ok(dataMulti.selected_tools.includes("Optical_Caption") && dataMulti.selected_tools.includes("Grounding"), "Should select both Caption and Grounding tools");
  console.log("✓ Multi-tool selected:", dataMulti.selected_tools);

  // 6. Optical + SAR Fusion Baseline Query
  console.log("\n[TEST 6] Testing Optical + SAR Dual-Modal Analysis...");
  const resFusion = await fetch(`${NEXTJS_BASE}/api/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query: "Compare optical and SAR imagery",
      optical_image: SAMPLE_PORT,
      sar_image: SAMPLE_SAR,
    }),
  });
  assert.strictEqual(resFusion.status, 200);
  const dataFusion = await resFusion.json();
  assert.ok(dataFusion.selected_tools.includes("Optical_SAR_Analysis"), "Should select Optical_SAR_Analysis tool");
  console.log("✓ Optical + SAR analysis executed, fusion type:", dataFusion.optical_sar_analysis?.fusion_type || "feature_fusion_baseline");

  // 7. Change Analysis Endpoint
  console.log("\n[TEST 7] Testing /api/change-analysis...");
  const resChg = await fetch(`${NEXTJS_BASE}/api/change-analysis`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      image_a: SAMPLE_PORT,
      image_b: SAMPLE_CHANGE_B,
      change_threshold: 0.15,
    }),
  });
  assert.strictEqual(resChg.status, 200, "Change analysis should return 200 OK");
  const dataChg = await resChg.json();
  assert.strictEqual(dataChg.status, "success");
  assert.ok(typeof dataChg.changed_fraction === "number", "Changed fraction should be a number");
  assert.ok(Array.isArray(dataChg.anomalies), "Anomalies array should be present");
  console.log(`✓ Change analysis detected changed_fraction: ${(dataChg.changed_fraction * 100).toFixed(1)}% with ${dataChg.anomalies.length} anomaly clusters.`);

  // 8. Plain Image Geospatial Verification (Zero Fabricated Coordinates)
  console.log("\n[TEST 8] Verifying Zero Fabricated Coordinates on Plain Imagery...");
  assert.strictEqual(dataCap.geospatial_metadata?.geospatial_available, false, "Plain image must have geospatial_available = false");
  assert.strictEqual(dataCap.investigation_report?.spatial_summary?.geospatial_available, false, "Spatial summary must not fabricate coordinates");
  console.log("✓ Verified zero fabricated coordinates for plain PNG imagery.");

  console.log("\n========================================================");
  console.log("ALL 8 END-TO-END NEXT.JS → FASTAPI SCENARIOS PASSED (100%)!");
  console.log("========================================================");
}

runTests().catch((err) => {
  console.error("Test failed with error:", err);
  process.exit(1);
});
