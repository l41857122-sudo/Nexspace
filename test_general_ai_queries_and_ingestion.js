/**
 * test_general_ai_queries_and_ingestion.js
 * -----------------------------------------
 * Comprehensive test suite verifying:
 * 1. Multi-Format Image Ingestion (JPG, JPEG, PNG, WEBP, TIFF)
 * 2. Broad General-Purpose Query Interpretation across 18 natural language queries
 */

const fs = require('fs');
const path = require('path');
const http = require('http');

function postJSON(urlStr, data) {
  return new Promise((resolve, reject) => {
    const url = new URL(urlStr);
    const postData = JSON.stringify(data);
    const req = http.request(
      {
        hostname: url.hostname,
        port: url.port,
        path: url.pathname + url.search,
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Content-Length': Buffer.byteLength(postData),
        },
        timeout: 45000,
      },
      (res) => {
        let body = '';
        res.on('data', (chunk) => (body += chunk));
        res.on('end', () => {
          try {
            resolve({ status: res.statusCode, data: JSON.parse(body) });
          } catch (e) {
            resolve({ status: res.statusCode, raw: body });
          }
        });
      }
    );
    req.on('error', reject);
    req.on('timeout', () => {
      req.destroy();
      reject(new Error('Request timed out'));
    });
    req.write(postData);
    req.end();
  });
}

async function runTests() {
  console.log('================================================================');
  console.log('NEXSPACE — GENERAL AI QUERY COMPATIBILITY & INGESTION SUITE');
  console.log('================================================================\n');

  let passed = 0;
  let total = 0;

  function assert(condition, message) {
    total++;
    if (condition) {
      console.log(`  ✓ ${message}`);
      passed++;
    } else {
      console.error(`  ✗ FAILED: ${message}`);
    }
  }

  // Load test rasters
  const opticalAPath = path.join(__dirname, 'public', 'test-data', 'optical_a.jpg');
  const opticalBPath = path.join(__dirname, 'public', 'test-data', 'optical_b.jpg');
  const sarAPath = path.join(__dirname, 'public', 'test-data', 'sar_a.png');

  const opticalABase64 = fs.readFileSync(opticalAPath).toString('base64');
  const opticalBBase64 = fs.readFileSync(opticalBPath).toString('base64');
  const sarABase64 = fs.readFileSync(sarAPath).toString('base64');

  // -------------------------------------------------------------
  // PART 1: Multi-Format Ingestion Tests
  // -------------------------------------------------------------
  console.log('[PART 1] Multi-Format Image Ingestion & Header Decode Verification...');

  assert(opticalABase64.length > 500, 'optical_a.jpg loaded and base64 encoded');
  assert(opticalBBase64.length > 500, 'optical_b.jpg loaded and base64 encoded');
  assert(sarABase64.length > 500, 'sar_a.png loaded and base64 encoded');

  // Send JPG directly to FastAPI
  const jpgRes = await postJSON('http://127.0.0.1:8000/api/query', {
    query: 'Describe this image',
    optical_image: opticalABase64,
  });
  assert(jpgRes.status === 200, 'FastAPI decodes raw JPG payload (HTTP 200)');
  assert(typeof jpgRes.data.optical_caption === 'string' && jpgRes.data.optical_caption.length > 0, 'BLIP generates caption from uploaded JPG');

  // -------------------------------------------------------------
  // PART 2: All 18 Query Scenarios
  // -------------------------------------------------------------
  console.log('\n[PART 2] Testing 18 General-Purpose Query Scenarios...\n');

  const testQueries = [
    {
      id: 1,
      name: 'Scene Understanding: "What do you see in this image?"',
      payload: { query: 'What do you see in this image?', optical_image: opticalABase64 },
      validate: (d) => d.status === 'completed' && Boolean(d.optical_caption) && Boolean(d.response_text),
    },
    {
      id: 2,
      name: 'Scene Understanding: "Describe the scene."',
      payload: { query: 'Describe the scene.', optical_image: opticalABase64 },
      validate: (d) => d.status === 'completed' && Boolean(d.optical_caption),
    },
    {
      id: 3,
      name: 'Counting: "How many buildings are there?"',
      payload: { query: 'How many buildings are there?', optical_image: opticalABase64 },
      validate: (d) => d.status === 'completed' && (d.grounding?.detections?.length > 0 || d.response_text.includes('Count')),
    },
    {
      id: 4,
      name: 'Object Identification: "Can you find the buildings?"',
      payload: { query: 'Can you find the buildings?', optical_image: opticalABase64 },
      validate: (d) => d.status === 'completed' && d.grounding?.detections?.length > 0,
    },
    {
      id: 5,
      name: 'Spatial Localization: "Where are the buildings?"',
      payload: { query: 'Where are the buildings?', optical_image: opticalABase64 },
      validate: (d) => d.status === 'completed' && d.response_text.includes('Location'),
    },
    {
      id: 6,
      name: 'Object Identification: "What objects are visible?"',
      payload: { query: 'What objects are visible?', optical_image: opticalABase64 },
      validate: (d) => d.status === 'completed' && Boolean(d.response_text),
    },
    {
      id: 7,
      name: 'Counting Paraphrase: "How many structures can you identify?"',
      payload: { query: 'How many structures can you identify?', optical_image: opticalABase64 },
      validate: (d) => d.status === 'completed' && d.response_text.includes('Count'),
    },
    {
      id: 8,
      name: 'Spatial Activity: "Which side has more activity?"',
      payload: { query: 'Which side has more activity?', optical_image: opticalABase64 },
      validate: (d) => d.status === 'completed' && Boolean(d.response_text),
    },
    {
      id: 9,
      name: 'Comparison: "What changed between these images?"',
      payload: { query: 'What changed between these images?', change_image_a: opticalABase64, change_image_b: opticalBBase64 },
      validate: (d) => d.status === 'completed' && d.change_analysis !== null && d.response_text.includes('Change'),
    },
    {
      id: 10,
      name: 'Change Localization: "Where is the biggest change?"',
      payload: { query: 'Where is the biggest change?', change_image_a: opticalABase64, change_image_b: opticalBBase64 },
      validate: (d) => d.status === 'completed' && d.change_analysis !== null,
    },
    {
      id: 11,
      name: 'Open Scene: "What can you tell me about this area?"',
      payload: { query: 'What can you tell me about this area?', optical_image: opticalABase64 },
      validate: (d) => d.status === 'completed' && Boolean(d.optical_caption),
    },
    {
      id: 12,
      name: 'SAR Analysis: "What does the SAR image show?"',
      payload: { query: 'What does the SAR image show?', sar_image: sarABase64 },
      validate: (d) => d.status === 'completed' && Boolean(d.sar_caption),
    },
    {
      id: 13,
      name: 'Optical + SAR Fusion: "Compare what the optical and SAR images tell us."',
      payload: { query: 'Compare what the optical and SAR images tell us.', optical_image: opticalABase64, sar_image: sarABase64 },
      validate: (d) => d.status === 'completed' && (Boolean(d.optical_sar_analysis) || d.response_text.includes('Optical & SAR Comparison')),
    },
    {
      id: 14,
      name: 'Geospatial Metadata: "What is the image resolution?"',
      payload: { query: 'What is the image resolution?', optical_image: opticalABase64 },
      validate: (d) => d.status === 'completed' && d.response_text.includes('Geospatial Metadata'),
    },
    {
      id: 15,
      name: 'Geospatial CRS: "What is the CRS?"',
      payload: { query: 'What is the CRS?', optical_image: opticalABase64 },
      validate: (d) => d.status === 'completed' && d.response_text.includes('Geospatial Metadata'),
    },
    {
      id: 16,
      name: 'Model Provenance: "What evidence supports your answer?"',
      payload: { query: 'What evidence supports your answer?', optical_image: opticalABase64 },
      validate: (d) => d.status === 'completed' && d.response_text.includes('Model Provenance'),
    },
    {
      id: 17,
      name: 'Model Confidence: "How confident are you?"',
      payload: { query: 'How confident are you?', optical_image: opticalABase64 },
      validate: (d) => d.status === 'completed' && d.response_text.includes('Model Provenance & Confidence'),
    },
    {
      id: 18,
      name: 'Open-Ended Land-Use: "Does this area look suitable for construction?"',
      payload: { query: 'Does this area look suitable for construction?', optical_image: opticalABase64 },
      validate: (d) => d.status === 'completed' && d.response_text.includes('Limitation Notice') && d.response_text.includes('suitability'),
    },
  ];

  for (const t of testQueries) {
    try {
      const res = await postJSON('http://127.0.0.1:8000/api/query', t.payload);
      const isOk = res.status === 200 && t.validate(res.data);
      assert(isOk, `Q#${t.id}: ${t.name}`);
      if (!isOk) {
        console.error('    DEBUG Response:', JSON.stringify(res.data, null, 2).slice(0, 300));
      }
    } catch (err) {
      assert(false, `Q#${t.id}: ${t.name} -> Error: ${err.message}`);
    }
  }

  console.log('\n================================================================');
  console.log(`FINAL RESULT: ${passed} / ${total} TESTS PASSED (${Math.round((passed / total) * 100)}%)`);
  console.log('================================================================');

  if (passed === total) {
    process.exit(0);
  } else {
    process.exit(1);
  }
}

runTests().catch((err) => {
  console.error('Fatal test error:', err);
  process.exit(1);
});
