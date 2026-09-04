/**
 * test_grounding_robustness.js
 * ----------------------------
 * Unit tests verifying that grounding bounding box normalization never crashes,
 * handles malformed / null / NaN inputs safely, and extracts valid 0-1000 coordinates.
 */

const assert = require("assert");

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
    // box / bbox_pixel is [xmin, ymin, xmax, ymax] in pixel coordinates
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

console.log("=== RUNNING GROUNDING ROBUSTNESS UNIT TESTS ===");

// 1. Valid box_2d in 0-1000 scale
const t1 = normalizeBox({ label: "building", box_2d: [100, 150, 400, 600], score: 0.85 });
assert.deepStrictEqual(t1, [100, 150, 400, 600], "Test 1: Valid 0-1000 box_2d failed");
console.log("✓ Test 1 Passed: Valid box_2d [100, 150, 400, 600] normalized correctly.");

// 2. Valid box in [xmin, ymin, xmax, ymax] pixel format (512x512)
const t2 = normalizeBox({ label: "ship", box: [51.2, 102.4, 256.0, 307.2], score: 0.9 }, 512, 512);
assert.deepStrictEqual(t2, [200, 100, 600, 500], "Test 2: Pixel [xmin, ymin, xmax, ymax] failed");
console.log("✓ Test 2 Passed: Pixel box converted to [ymin, xmin, ymax, xmax] 0-1000 scale.");

// 3. Normalized 0.0 - 1.0 format
const t3 = normalizeBox({ label: "roof", box_2d: [0.1, 0.2, 0.5, 0.8], score: 0.7 });
assert.deepStrictEqual(t3, [100, 200, 500, 800], "Test 3: 0-1 float box failed");
console.log("✓ Test 3 Passed: 0.0-1.0 float box scaled to 0-1000 scale.");

// 4. Missing box_2d entirely
const t4 = normalizeBox({ label: "broken" });
assert.strictEqual(t4, null, "Test 4: Missing box should return null");
console.log("✓ Test 4 Passed: Missing box returned null safely.");

// 5. Null / undefined box_2d
const t5 = normalizeBox({ label: "null_box", box_2d: null });
assert.strictEqual(t5, null, "Test 5: Null box should return null");
console.log("✓ Test 5 Passed: Null box returned null safely.");

// 6. Malformed box_2d length
const t6 = normalizeBox({ label: "short_box", box_2d: [100, 200] });
assert.strictEqual(t6, null, "Test 6: Short box should return null");
console.log("✓ Test 6 Passed: Array with length != 4 returned null safely.");

// 7. NaN / Infinity in box_2d
const t7 = normalizeBox({ label: "nan_box", box_2d: [100, NaN, 300, 400] });
assert.strictEqual(t7, null, "Test 7: NaN should return null");
const t7b = normalizeBox({ label: "inf_box", box_2d: [100, Infinity, 300, 400] });
assert.strictEqual(t7b, null, "Test 7b: Infinity should return null");
console.log("✓ Test 7 Passed: NaN and Infinity coordinates rejected safely.");

// 8. Inverted coordinates [ymax, xmax, ymin, xmin]
const t8 = normalizeBox({ label: "inverted", box_2d: [800, 900, 200, 300] });
assert.deepStrictEqual(t8, [200, 300, 800, 900], "Test 8: Inverted coordinates failed");
console.log("✓ Test 8 Passed: Inverted coordinates [ymax > ymin] automatically reordered.");

// 9. Mixed detections array filtering
const rawList = [
  { label: "good_1", box_2d: [100, 100, 300, 300], score: 0.8 },
  { label: "bad_null", box_2d: null },
  { label: "bad_str", box_2d: "not_an_array" },
  { label: "bad_nan", box_2d: [0, NaN, 100, 100] },
  { label: "good_2", box: [200, 200, 400, 400], score: 0.75 },
];

const cleaned = rawList.map(d => ({ ...d, normalized_box: normalizeBox(d) })).filter(d => d.normalized_box !== null);
assert.strictEqual(cleaned.length, 2, "Test 9: Mixed array filtering failed");
console.log("✓ Test 9 Passed: Mixed detections array filtered to 2 valid items without errors.");

console.log("\n========================================================");
console.log("ALL 9 GROUNDING ROBUSTNESS UNIT TESTS PASSED (100%)!");
console.log("========================================================");
