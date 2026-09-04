/**
 * test_combined_query_verification.js
 * -----------------------------------
 * Live end-to-end verification of "Describe this image and locate the buildings",
 * measuring first-run vs second-run execution time and validating zero runtime errors.
 */

const http = require("http");
const assert = require("assert");

const SAMPLE_OPTICAL_URBAN_URL = "http://localhost:3000/demo/urban.png";

function normalizeBox(det, imgWidth = 512, imgHeight = 512) {
  if (!det || typeof det !== "object") return null;

  const raw = det.box_2d || det.box || det.bbox || det.bbox_pixel;
  if (!Array.isArray(raw) || raw.length !== 4) return null;

  const numeric = raw.map((val) => {
    const n = Number(val);
    return typeof n === "number" && Number.isFinite(n) ? n : NaN;
  });

  if (numeric.some((n) => isNaN(n))) return null;

  const [v0, v1, v2, v3] = numeric;
  let ymin, xmin, ymax, xmax;

  if (det.box_2d && !det.box) {
    ymin = v0; xmin = v1; ymax = v2; xmax = v3;
    const maxVal = Math.max(ymin, xmin, ymax, xmax);
    if (maxVal <= 1.05) {
      ymin *= 1000; xmin *= 1000; ymax *= 1000; xmax *= 1000;
    }
  } else {
    xmin = v0; ymin = v1; xmax = v2; ymax = v3;
    if (Math.max(ymin, xmin, ymax, xmax) <= 1.05) {
      ymin *= 1000; xmin *= 1000; ymax *= 1000; xmax *= 1000;
    } else {
      ymin = (ymin / imgHeight) * 1000;
      ymax = (ymax / imgHeight) * 1000;
      xmin = (xmin / imgWidth) * 1000;
      xmax = (xmax / imgWidth) * 1000;
    }
  }

  if (ymin > ymax) [ymin, ymax] = [ymax, ymin];
  if (xmin > xmax) [xmin, xmax] = [xmax, xmin];

  ymin = Math.round(Math.max(0, Math.min(1000, ymin)));
  xmin = Math.round(Math.max(0, Math.min(1000, xmin)));
  ymax = Math.round(Math.max(0, Math.min(1000, ymax)));
  xmax = Math.round(Math.max(0, Math.min(1000, xmax)));

  if (ymax <= ymin && xmax <= xmin) return null;

  return [ymin, xmin, ymax, xmax];
}

async function fetchBase64Image(url) {
  return new Promise((resolve, reject) => {
    http.get(url, (res) => {
      const chunks = [];
      res.on("data", chunk => chunks.push(chunk));
      res.on("end", () => {
        const buffer = Buffer.concat(chunks);
        resolve(`data:image/png;base64,${buffer.toString("base64")}`);
      });
    }).on("error", reject);
  });
}

async function postQuery(payload) {
  const data = JSON.stringify(payload);
  const t0 = Date.now();

  return new Promise((resolve, reject) => {
    const req = http.request(
      "http://localhost:3000/api/query",
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
            reject(new Error(`Failed to parse response: ${body.slice(0, 200)}`));
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
  console.log("=== VERIFYING COMBINED INVESTIGATION QUERY ===");
  console.log("Query: 'Describe this image and locate the buildings'");

  const base64Image = await fetchBase64Image(SAMPLE_OPTICAL_URBAN_URL);
  console.log(`✓ Loaded test raster: ${base64Image.length} base64 chars`);

  // Run 1: Measure First-Run Duration
  console.log("\n[RUN 1] Executing First Investigation Run...");
  const run1 = await postQuery({
    query: "Describe this image and locate the buildings",
    optical_image: base64Image,
  });

  console.log(`✓ Run 1 completed in ${run1.duration.toFixed(2)}s with HTTP ${run1.statusCode}`);
  assert.strictEqual(run1.statusCode, 200);
  assert.ok(run1.data.optical_caption, "Expected optical caption");
  assert.ok(run1.data.grounding, "Expected grounding results");

  const detections1 = run1.data.grounding.detections || [];
  console.log(`✓ Grounding Detections returned: ${detections1.length}`);

  // Validate that normalizeBox processes all returned detections safely
  for (let i = 0; i < detections1.length; i++) {
    const box = normalizeBox(detections1[i]);
    assert.ok(box, `Detection ${i} should produce valid normalized box`);
    console.log(`  - Detection ${i + 1} (${detections1[i].label}): normalized box = [${box.join(", ")}]`);
  }

  // Run 2: Measure Second-Run (Cached Models) Duration
  console.log("\n[RUN 2] Executing Second Investigation Run (Model Cache Verification)...");
  const run2 = await postQuery({
    query: "Describe this image and locate the buildings",
    optical_image: base64Image,
  });

  console.log(`✓ Run 2 completed in ${run2.duration.toFixed(2)}s with HTTP ${run2.statusCode}`);
  assert.strictEqual(run2.statusCode, 200);

  console.log("\n========================================================");
  console.log(`PERFORMANCE COMPARISON:`);
  console.log(`  First Run:  ${run1.duration.toFixed(2)} seconds`);
  console.log(`  Second Run: ${run2.duration.toFixed(2)} seconds`);
  console.log(`  Result: ZERO RUNTIME ERRORS OR TYPEERRORS ENCOUNTERED.`);
  console.log("========================================================");
})();
