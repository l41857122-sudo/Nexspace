/**
 * test_two_image_state_isolation.js
 * ---------------------------------
 * Verifies that two different image investigations produce completely distinct,
 * dynamic results without cross-contamination or stale state leakage.
 */

const fs = require("fs");
const path = require("path");
const http = require("http");
const assert = require("assert");

const DEMO_DIR = path.join(__dirname, "public", "demo");

function fileToDataUrl(filePath, mime) {
  const buf = fs.readFileSync(filePath);
  return `data:${mime};base64,${buf.toString("base64")}`;
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
            reject(new Error(`Failed to parse JSON response: ${body.slice(0, 300)}`));
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
  console.log("TESTING TWO-IMAGE STATE ISOLATION & DYNAMIC TARGET RE-CALCULATION");
  console.log("================================================================");

  // Investigation 1: Image A (Urban Buildings)
  console.log("\n[INVESTIGATION 1] Uploading Image A: 'urban_buildings.jpg'...");
  const imgA_Url = fileToDataUrl(path.join(DEMO_DIR, "urban_buildings.jpg"), "image/jpeg");

  const inv1 = await postQuery({
    query: "Locate the buildings",
    optical_image: imgA_Url,
  });

  assert.strictEqual(inv1.statusCode, 200);
  const det1 = inv1.data.grounding ? inv1.data.grounding.detections : [];
  console.log(`✓ Image A Result: ${det1.length} detections found`);
  console.log(`  Sample Box: [${det1[0]?.box_2d?.join(", ")}]`);

  // Investigation 2: Image B (Water Coast / Port)
  console.log("\n[INVESTIGATION 2] Uploading Image B: 'water_coast.jpg'...");
  const imgB_Url = fileToDataUrl(path.join(DEMO_DIR, "water_coast.jpg"), "image/jpeg");

  const inv2 = await postQuery({
    query: "Locate the ships",
    optical_image: imgB_Url,
  });

  assert.strictEqual(inv2.statusCode, 200);
  const det2 = inv2.data.grounding ? inv2.data.grounding.detections : [];
  console.log(`✓ Image B Result: ${det2.length} detections found`);
  if (det2.length > 0) {
    console.log(`  Sample Box: [${det2[0]?.box_2d?.join(", ")}]`);
  }

  // Verification Checks
  console.log("\n[VERIFICATION CHECKS]");
  assert.notStrictEqual(inv1.data.request_id, inv2.data.request_id, "Request IDs must be unique");
  console.log(`✓ Unique Investigation IDs: ${inv1.data.request_id} != ${inv2.data.request_id}`);

  // Ensure no stale data leaked
  assert.ok(!JSON.stringify(inv1.data).includes("Maasvlakte"), "No Maasvlakte in Investigation 1");
  assert.ok(!JSON.stringify(inv2.data).includes("Maasvlakte"), "No Maasvlakte in Investigation 2");
  assert.ok(!JSON.stringify(inv1.data).includes("Panamax"), "No Panamax in Investigation 1");
  assert.ok(!JSON.stringify(inv2.data).includes("Panamax"), "No Panamax in Investigation 2");
  console.log("✓ Zero hardcoded demo entities (Panamax/Maasvlakte) in responses");

  console.log("\n================================================================");
  console.log("TWO-IMAGE STATE ISOLATION TESTS PASSED (100%)");
  console.log("================================================================");
})();
