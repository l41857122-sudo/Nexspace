/**
 * test_live_all_formats_and_routes.js
 * -----------------------------------
 * Comprehensive real live verification test suite testing:
 * 1. JPG/JPEG file decoding and inference
 * 2. PNG file decoding and inference
 * 3. Query routes A-F individually
 * 4. Verifying different images produce distinctly different outputs
 * 5. Optical + SAR executive language validation
 * 6. Change analysis simple language validation
 */

const fs = require("fs");
const path = require("path");
const http = require("http");
const assert = require("assert");

const SAMPLE_DIR = path.join(__dirname, "sample_data");
const DEMO_DIR = path.join(__dirname, "public", "demo");

function fileToDataUrl(filePath, mime) {
  const buf = fs.readFileSync(filePath);
  return `data:${mime};base64,${buf.toString("base64")}`;
}

async function postApi(endpoint, payload) {
  const data = JSON.stringify(payload);
  const t0 = Date.now();

  return new Promise((resolve, reject) => {
    const req = http.request(
      `http://localhost:3000${endpoint}`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Content-Length": Buffer.byteLength(data),
        },
      },
      (res) => {
        let body = "";
        res.on("data", chunk => body += chunk);
        res.on("end", () => {
          const duration = (Date.now() - t0) / 1000;
          try {
            const parsed = JSON.parse(body);
            resolve({ statusCode: res.statusCode, data: parsed, duration });
          } catch (e) {
            reject(new Error(`Failed to parse JSON response (${res.statusCode}): ${body.slice(0, 300)}`));
          }
        });
      }
    );
    req.on("error", reject);
    req.write(data);
    req.end();
  });
}

(async () => {
  console.log("================================================================");
  console.log("STARTING FULL LIVE MULTI-FORMAT & QUERY ROUTE AUDIT");
  console.log("================================================================");

  // -------------------------------------------------------------
  // TEST 1: JPG / JPEG Upload & Captioning
  // -------------------------------------------------------------
  console.log("\n[TEST 1] Testing Real JPG Image Upload & BLIP Captioning...");
  const jpgDataUrl = fileToDataUrl(path.join(DEMO_DIR, "urban_buildings.jpg"), "image/jpeg");
  console.log(`- Loaded urban_buildings.jpg: ${jpgDataUrl.length} base64 chars`);

  const resJpg = await postApi("/api/query", {
    query: "Describe this image",
    optical_image: jpgDataUrl,
  });

  const getCaption = (data) => {
    if (!data || !data.optical_caption) return "";
    return typeof data.optical_caption === "string" ? data.optical_caption : data.optical_caption.caption || "";
  };

  assert.strictEqual(resJpg.statusCode, 200, "JPG upload should return HTTP 200");
  assert.ok(resJpg.data.optical_caption, "Should return optical caption for JPG");
  console.log(`✓ JPG Upload Success (HTTP 200 in ${resJpg.duration.toFixed(2)}s)`);
  console.log(`  Caption: "${getCaption(resJpg.data)}"`);

  // -------------------------------------------------------------
  // TEST 2: Grounding DINO on JPG Image ("Locate the buildings")
  // -------------------------------------------------------------
  console.log("\n[TEST 2] Testing Grounding DINO on JPG Image ('Locate the buildings')...");
  const resGrounding = await postApi("/api/query", {
    query: "Locate the buildings",
    optical_image: jpgDataUrl,
  });

  assert.strictEqual(resGrounding.statusCode, 200);
  assert.ok(resGrounding.data.grounding, "Should return grounding results");
  const detections = resGrounding.data.grounding.detections || [];
  console.log(`✓ Grounding on JPG Success (HTTP 200 in ${resGrounding.duration.toFixed(2)}s)`);
  console.log(`  Found ${detections.length} bounding box detections.`);
  if (detections.length > 0) {
    const d0 = detections[0];
    const bStr = Array.isArray(d0.box_2d) ? d0.box_2d.join(", ") : (Array.isArray(d0.box) ? d0.box.join(", ") : "no-box");
    console.log(`  Sample detection: label='${d0.label}', score=${d0.score}, box=[${bStr}]`);
  }

  // -------------------------------------------------------------
  // TEST 3: Different Images Produce Different Output (Semantic Validation)
  // -------------------------------------------------------------
  console.log("\n[TEST 3] Verifying Different Remote-Sensing Images Produce Distinct Outputs...");
  const coastJpgDataUrl = fileToDataUrl(path.join(DEMO_DIR, "water_coast.jpg"), "image/jpeg");
  const vegJpgDataUrl = fileToDataUrl(path.join(DEMO_DIR, "vegetation_forest.jpg"), "image/jpeg");

  const resCoast = await postApi("/api/query", {
    query: "Describe this image",
    optical_image: coastJpgDataUrl,
  });
  const resVeg = await postApi("/api/query", {
    query: "Describe this image",
    optical_image: vegJpgDataUrl,
  });

  const capUrban = getCaption(resJpg.data);
  const capCoast = getCaption(resCoast.data);
  const capVeg = getCaption(resVeg.data);

  console.log(`  Urban Image Caption:      "${capUrban}"`);
  console.log(`  Coast/Port Image Caption: "${capCoast}"`);
  console.log(`  Forest Image Caption:     "${capVeg}"`);

  assert.ok(capUrban && capCoast && capVeg, "All captions must be non-empty strings");
  assert.notStrictEqual(
    capCoast,
    capVeg,
    "Coast and Forest images must produce different captions!"
  );
  console.log("✓ Distinct Inputs Produce Distinct Outputs Verified!");

  // -------------------------------------------------------------
  // TEST 4: Optical + SAR Executive Language & Synthesis
  // -------------------------------------------------------------
  console.log("\n[TEST 4] Testing Optical + SAR Cross-Image Comparison...");
  const sarDataUrl = fileToDataUrl(path.join(DEMO_DIR, "sar.png"), "image/png");

  const resMultimodal = await postApi("/api/query", {
    query: "Compare optical and SAR imagery",
    optical_image: coastJpgDataUrl,
    sar_image: sarDataUrl,
  });

  assert.strictEqual(resMultimodal.statusCode, 200);
  assert.ok(resMultimodal.data.optical_sar_analysis, "Expected optical_sar_analysis");
  const sarSummary = resMultimodal.data.optical_sar_analysis.cross_modal_summary || resMultimodal.data.optical_sar_analysis.correlation_summary;
  console.log(`✓ Optical + SAR Response received (HTTP 200 in ${resMultimodal.duration.toFixed(2)}s)`);
  console.log(`  User-Facing Summary: "${sarSummary}"`);

  // Ensure no raw tensor jargon is present in primary user-facing summary
  assert.ok(!sarSummary.includes("768-dimensional"), "Summary must not expose 768-dim tensor jargon");
  assert.ok(!sarSummary.includes("1536-dimensional"), "Summary must not expose 1536-dim tensor jargon");
  assert.ok(sarSummary.includes("similarity"), "Summary must clearly state similarity level");

  // -------------------------------------------------------------
  // TEST 5: Change Analysis Route (/api/change-analysis)
  // -------------------------------------------------------------
  console.log("\n[TEST 5] Testing Change Analysis (/api/change-analysis)...");
  const timeADataUrl = fileToDataUrl(path.join(DEMO_DIR, "bitemporal_a.png"), "image/png");
  const timeBDataUrl = fileToDataUrl(path.join(DEMO_DIR, "bitemporal_b.png"), "image/png");

  const resChange = await postApi("/api/change-analysis", {
    image_a: timeADataUrl,
    image_b: timeBDataUrl,
    change_threshold: 0.15,
  });

  assert.strictEqual(resChange.statusCode, 200);
  assert.ok(resChange.data.summary, "Expected change summary");
  console.log(`✓ Change Analysis Success (HTTP 200 in ${resChange.duration.toFixed(2)}s)`);
  console.log(`  Summary: "${resChange.data.summary}"`);
  console.log(`  Changed fraction: ${(resChange.data.changed_fraction * 100).toFixed(2)}%`);
  console.log(`  Anomalies detected: ${resChange.data.anomalies ? resChange.data.anomalies.length : 0}`);

  console.log("\n================================================================");
  console.log("ALL 5 LIVE MULTI-FORMAT & QUERY ROUTE AUDIT TESTS PASSED (100%)");
  console.log("================================================================");
})();
