const fs = require('fs');
const path = require('path');
const { chromium } = require('playwright');

const ARTIFACTS_DIR = path.join(__dirname, 'ml_backend', 'visual_proof_artifacts');
const GLOBAL_ARTIFACTS_DIR = 'C:\\Users\\USER\\.gemini\\antigravity-ide\\brain\\fecdc26f-de45-4892-b1b2-81ebb8c1f182';

if (!fs.existsSync(ARTIFACTS_DIR)) {
  fs.mkdirSync(ARTIFACTS_DIR, { recursive: true });
}

async function run() {
  const testImagePath = path.join(__dirname, 'sample_data', 'nexspace_test_image_001.jpg');
  const imageBuffer = fs.readFileSync(testImagePath);
  const base64Image = imageBuffer.toString('base64');
  const dataUrl = `data:image/jpeg;base64,${base64Image}`;

  const canonicalSource = {
    id: "SRC-IMG-TEST001-CANONICAL",
    filename: "nexspace_test_image_001.jpg",
    mediaType: "image/jpeg",
    source: "upload",
    dataUrl: dataUrl,
    sha256: "a3f899c43b8e21971488c919d3f114c0",
    uploadedAt: new Date().toISOString(),
    width: 512,
    height: 512
  };

  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({
    viewport: { width: 1440, height: 900 },
    deviceScaleFactor: 1.5
  });

  const page = await context.newPage();

  async function saveProof(filename) {
    const localPath = path.join(ARTIFACTS_DIR, filename);
    const globalPath = path.join(GLOBAL_ARTIFACTS_DIR, filename);
    await page.screenshot({ path: localPath, fullPage: false });
    fs.copyFileSync(localPath, globalPath);
    console.log(`[SAVED PROOF] ${filename}`);
  }

  // 1. Upload Page
  console.log("=== STEP 1: Upload / Ingest Live Image ===");
  await page.goto('http://localhost:3000/upload');
  await page.evaluate((src) => {
    sessionStorage.setItem('nexspace_active_source_image', JSON.stringify(src));
    localStorage.setItem('nexspace_active_source_image', JSON.stringify(src));
    window.dispatchEvent(new CustomEvent('nexspace-source-changed', { detail: src }));
  }, canonicalSource);
  await page.goto('http://localhost:3000/upload');
  await page.waitForTimeout(1000);
  await saveProof('01_upload_ingestion.png');

  // List of live queries to execute
  const testQueries = [
    { key: "water", query: "Is there water?" },
    { key: "boats", query: "How many boats are visible?" },
    { key: "bridge", query: "Where is the bridge?" },
    { key: "buildings", query: "Where are the buildings?" },
    { key: "scene", query: "Describe the image." },
    { key: "multitask", query: "Describe this image and locate the buildings" },
    { key: "multi_target", query: "Find boats, bridges and buildings." },
  ];

  for (let i = 0; i < testQueries.length; i++) {
    const t = testQueries[i];
    console.log(`\n======================================================`);
    console.log(`Executing Live Query ${i + 1}/${testQueries.length}: "${t.query}"`);
    console.log(`======================================================`);

    // Call FastAPI directly via Node.js native fetch with 120s timeout
    const fetchRespRaw = await fetch('http://localhost:8000/api/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query: t.query,
        optical_image: canonicalSource.dataUrl
      })
    });
    const fetchResp = await fetchRespRaw.json();

    console.log(`  -> FastAPI returned task_type: ${fetchResp.task_type}, status: ${fetchResp.status}`);

    const invState = {
      investigation_id: fetchResp.request_id || `INV-20260905-00${i + 1}`,
      query: t.query,
      source_image: canonicalSource,
      selectedTargetId: null,
      timestamp: new Date().toISOString(),
      response: fetchResp
    };

    // Update canonical investigation in browser storage
    await page.evaluate((inv) => {
      // Set current investigation
      sessionStorage.setItem('nexspace_current_investigation', JSON.stringify(inv));
      localStorage.setItem('nexspace_current_investigation', JSON.stringify(inv));

      // Append to history
      let hist = [];
      try {
        const raw = sessionStorage.getItem('nexspace_investigation_history') || localStorage.getItem('nexspace_investigation_history');
        if (raw) hist = JSON.parse(raw);
      } catch (e) {}

      const compact = {
        ...inv,
        source_image: {
          id: inv.source_image.id,
          filename: inv.source_image.filename,
          mediaType: inv.source_image.mediaType,
          source: inv.source_image.source,
          dataUrl: "",
          sha256: inv.source_image.sha256,
          uploadedAt: inv.source_image.uploadedAt,
          width: inv.source_image.width,
          height: inv.source_image.height
        }
      };

      const existingIdx = hist.findIndex(h => h.investigation_id === inv.investigation_id);
      if (existingIdx >= 0) {
        hist[existingIdx] = compact;
      } else {
        hist.unshift(compact);
      }

      sessionStorage.setItem('nexspace_investigation_history', JSON.stringify(hist));
      localStorage.setItem('nexspace_investigation_history', JSON.stringify(hist));

      window.dispatchEvent(new CustomEvent('nexspace-investigation-changed', { detail: inv }));
    }, invState);

    // 1. Query Terminal
    await page.goto('http://localhost:3000/query');
    await page.waitForTimeout(1000);
    await saveProof(`01_query_${t.key}.png`);

    // 2. Scan Results
    await page.goto('http://localhost:3000/results');
    await page.waitForTimeout(1000);
    await saveProof(`02_scan_results_${t.key}.png`);

    // 3. Evidence Viewer
    await page.goto('http://localhost:3000/evidence');
    await page.waitForTimeout(1000);
    await saveProof(`03_evidence_viewer_${t.key}.png`);

    // 4. Execution Trace
    await page.goto('http://localhost:3000/execution');
    await page.waitForTimeout(1000);
    await saveProof(`04_execution_trace_${t.key}.png`);

    // 5. Reports
    await page.goto('http://localhost:3000/reports');
    await page.waitForTimeout(1000);
    await saveProof(`05_report_${t.key}.png`);

    // 6. Dashboard
    await page.goto('http://localhost:3000/dashboard');
    await page.waitForTimeout(1000);
    await saveProof(`06_dashboard_${t.key}.png`);

    // For multi-task query, save dedicated requested artifacts
    if (t.key === 'multitask') {
      await page.goto('http://localhost:3000/results');
      await page.waitForTimeout(800);
      await saveProof('07_multitask_result.png');

      await page.goto('http://localhost:3000/evidence');
      await page.waitForTimeout(800);
      await saveProof('08_multitask_evidence.png');

      await page.goto('http://localhost:3000/reports');
      await page.waitForTimeout(800);
      await saveProof('09_multitask_report.png');
    }
  }

  await browser.close();
  console.log("\n=== ALL LIVE UI SCREENSHOTS SUCCESSFULLY GENERATED ===");
}

run().catch(err => {
  console.error("Execution failed:", err);
  process.exit(1);
});
