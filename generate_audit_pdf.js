const PDFDocument = require("pdfkit");
const fs = require("fs");
const path = require("path");

const publicPath = path.join(__dirname, "public", "NexSpace_Backend_And_Agent_Controller_PRD_Audit.pdf");
const doc = new PDFDocument({
  size: "A4",
  margins: { top: 40, bottom: 40, left: 45, right: 45 },
  bufferPages: true
});

const writeStream = fs.createWriteStream(publicPath);
doc.pipe(writeStream);

// Palette
const C_DARK_BG = "#06111D";
const C_CYAN = "#06B6D4";
const C_SKY = "#38BDF8";
const C_TEXT = "#E2E8F0";
const C_MUTED = "#94A3B8";
const C_EMERALD = "#10B981";
const C_AMBER = "#F59E0B";
const C_RED = "#EF4444";
const C_BORDER = "#1E293B";

function addHeader(title, subtitle) {
  doc.fontSize(16).fillColor(C_CYAN).font("Helvetica-Bold").text(title);
  if (subtitle) {
    doc.fontSize(8.5).fillColor(C_MUTED).font("Helvetica").text(subtitle);
  }
  doc.moveDown(0.5);
  doc.strokeColor(C_BORDER).lineWidth(0.8).moveTo(45, doc.y).lineTo(550, doc.y).stroke();
  doc.moveDown(0.5);
}

function addSectionTitle(title) {
  doc.moveDown(0.6);
  doc.fontSize(12).fillColor(C_SKY).font("Helvetica-Bold").text(title);
  doc.moveDown(0.3);
}

function addParagraph(text) {
  doc.fontSize(8).fillColor(C_TEXT).font("Helvetica").text(text, { align: "justify", lineGap: 2 });
  doc.moveDown(0.3);
}

function addAuditItem({ requirement, file, status, statusColor, currentDoes, prdRequires, exactChange, blockers }) {
  doc.fontSize(9).font("Helvetica-Bold").fillColor(C_CYAN).text(`Requirement: `, { continued: true });
  doc.fontSize(9).font("Helvetica-Bold").fillColor("#FFFFFF").text(requirement);
  
  doc.fontSize(7.5).font("Helvetica-Bold").fillColor(C_MUTED).text(`Current Component: `, { continued: true });
  doc.font("Helvetica").fillColor(C_TEXT).text(file);

  doc.fontSize(7.5).font("Helvetica-Bold").fillColor(C_MUTED).text(`Current Status: `, { continued: true });
  doc.font("Helvetica-Bold").fillColor(statusColor || C_AMBER).text(status);

  doc.fontSize(7.5).font("Helvetica-Bold").fillColor(C_MUTED).text(`What it currently does: `, { continued: true });
  doc.font("Helvetica").fillColor(C_TEXT).text(currentDoes, { lineGap: 1.5 });

  doc.fontSize(7.5).font("Helvetica-Bold").fillColor(C_MUTED).text(`What the PRD requires: `, { continued: true });
  doc.font("Helvetica").fillColor(C_TEXT).text(prdRequires, { lineGap: 1.5 });

  doc.fontSize(7.5).font("Helvetica-Bold").fillColor(C_MUTED).text(`Exact change needed: `, { continued: true });
  doc.font("Helvetica").fillColor(C_TEXT).text(exactChange, { lineGap: 1.5 });

  doc.fontSize(7.5).font("Helvetica-Bold").fillColor(C_MUTED).text(`Dependencies / Blockers: `, { continued: true });
  doc.font("Helvetica").fillColor(C_TEXT).text(blockers, { lineGap: 1.5 });

  doc.moveDown(0.5);
  doc.strokeColor(C_BORDER).lineWidth(0.5).moveTo(45, doc.y).lineTo(550, doc.y).stroke();
  doc.moveDown(0.5);
}

// -------------------------------------------------------------
// COVER / TITLE PAGE
// -------------------------------------------------------------
doc.rect(0, 0, 595.28, 841.89).fill("#07111C");

doc.fontSize(22).fillColor("#00E5FF").font("Helvetica-Bold").text("NexSpace / SatQuery AI", 45, 110);
doc.fontSize(14).fillColor("#38BDF8").font("Helvetica-Bold").text("Backend & Agent Controller PRD Technical Audit");
doc.fontSize(9).fillColor("#94A3B8").font("Helvetica").text("Comprehensive Codebase Inspection, Ground-Truth Verification & Gap Analysis");

doc.moveDown(1.5);
doc.strokeColor("#1E293B").lineWidth(1).moveTo(45, doc.y).lineTo(550, doc.y).stroke();
doc.moveDown(1.2);

doc.fontSize(9.5).fillColor("#E2E8F0").font("Helvetica-Bold").text("EXECUTIVE AUDIT SUMMARY:");
doc.fontSize(8).fillColor("#94A3B8").font("Helvetica").text(
  "This technical audit evaluates the actual implemented backend code in ml_backend/, processing/, server/services, " +
  "and Next.js API routes against the NexSpace PRD. Each capability is classified across five ground-truth states: " +
  "REAL & WORKING, PARTIALLY IMPLEMENTED, MOCK/SIMULATED, MISSING, or BROKEN."
, { lineGap: 2.5 });

doc.moveDown(1.5);
doc.fontSize(9).fillColor("#38BDF8").font("Helvetica-Bold").text("EXECUTIVE AUDIT ANSWERS TO KEY QUESTIONS");

const keyAnswers = [
  { q: "Which specialist models are actually loaded & used?", a: "None loaded by default. tools.py wraps google/paligemma-3b-ft-rsvqa-lr-224 and Salesforce/blip-image-captioning-base in deferred try...except blocks; if weights are absent, it returns deterministic string fallbacks." },
  { q: "Are PaliGemma / BLIP / YOLO real inference or mock logic?", a: "PaliGemma and BLIP have real Hugging Face pipeline scaffolding but default to mock strings. YOLOv8 is 100% simulated via detect_objects_placeholder() returning hardcoded vessel/infrastructure coordinates." },
  { q: "Is Agent Controller agentic or rule-based?", a: "Rule-based regex & heuristic keyword matching in router.py (e.g. re.search(r'\\bhow many\\b')). No autonomous LLM function-calling graph is active." },
  { q: "Does Optical + SAR fusion actually exist?", a: "No joint tensor embedding fusion exists. In orchestrator.py, fusion is purely string concatenation of optical_caption_result and sar_caption_result." },
  { q: "Are execution traces real or simulated?", a: "Simulated. Stage graphs and timing in queryService.ts and /api/query/[id]/trace return static mock timestamps and hardcoded GPU allocations." },
  { q: "Are confidence scores real or hardcoded?", a: "Hardcoded. Sets confidence to 0.32 if 'how many' is present, and 0.85 for all other queries." },
  { q: "Are evidence bounding boxes from models or hardcoded?", a: "Hardcoded static coordinate arrays in ComparisonPage, ScanResultsPage, and pipeline.py." },
  { q: "Does Next.js /api/query communicate with FastAPI backend?", a: "Yes, attempts fetch('http://localhost:8000/api/query') with a 3s timeout and falls back to an inlined TypeScript rule router if FastAPI is offline." }
];

keyAnswers.forEach(({ q, a }) => {
  doc.fontSize(7.5).font("Helvetica-Bold").fillColor(C_CYAN).text(`• ${q}: `, { continued: true });
  doc.font("Helvetica").fillColor(C_TEXT).text(a, { lineGap: 1.5 });
  doc.moveDown(0.2);
});

doc.moveDown(2);
doc.fontSize(7.5).fillColor("#64748B").font("Helvetica").text("Audit Timestamp: August 2026 · Local Workspace Distribution · NexSpace Platform v2.4.1");

// -------------------------------------------------------------
// SECTION 2: AGENT CONTROLLER REQUIREMENTS
// -------------------------------------------------------------
doc.addPage();
addHeader("1. Agent Controller Implementation Audit", "Detailed capability evaluation against PRD specifications");

addAuditItem({
  requirement: "Natural-Language Query Classification",
  file: "ml_backend/router.py (_is_counting, _is_open_ended), app/api/query/route.ts",
  status: "PARTIALLY IMPLEMENTED (Regex & Keyword Heuristics)",
  statusColor: C_AMBER,
  currentDoes: "Uses regex patterns and starter phrase arrays (e.g., 'what is visible', 'is there') to classify queries into counting, open-ended, or closed-ended categories.",
  prdRequires: "Autonomous semantic intent classification capable of parsing complex geospatial constraints (bounding boxes, sensor modalities, temporal windows, and confidence thresholds).",
  exactChange: "Replace or augment keyword heuristics with an LLM function-calling schema or small classification model.",
  blockers: "LLM API Key (Anthropic/Gemini/OpenAI) or local quantized intent classifier model."
});

addAuditItem({
  requirement: "VQA Intent & RSVQA Query Normalization",
  file: "ml_backend/router.py (_normalize_closed_ended, _normalize_counting, decompose_open_ended)",
  status: "REAL AND WORKING (Heuristic Engine)",
  statusColor: C_EMERALD,
  currentDoes: "Decomposes open-ended queries into atomic RSVQA binary sub-questions using static probe dictionaries (FEATURE_PROBES) and formats questions with trailing '?' and leading capitals.",
  prdRequires: "Decompose complex geospatial queries into atomic, verifiable sub-queries for visual verification against multi-spectral satellite tiles.",
  exactChange: "Expand static probe dictionary to dynamic zero-shot spatial entity extraction.",
  blockers: "None."
});

addAuditItem({
  requirement: "Captioning / Grounding Routing",
  file: "ml_backend/router.py, ml_backend/orchestrator.py",
  status: "PARTIALLY IMPLEMENTED (Captioning only, Grounding missing)",
  statusColor: C_AMBER,
  currentDoes: "Routes open-ended requests to Optical_Caption or SAR_Caption and invokes CaptioningTool.caption().",
  prdRequires: "Free-form captioning coupled with visual grounding (returning localized bounding boxes and segmentation masks for referenced scene objects).",
  exactChange: "Integrate visual grounding model (Grounding-DINO or PaliGemma grounding tokens <loc0000>...<loc1023>).",
  blockers: "Grounding model weights and PyTorch GPU runtime."
});

addAuditItem({
  requirement: "Change Analysis / Change-VQA Routing",
  file: "ml_backend/router.py (line 172), ml_backend/change_analysis.py",
  status: "REAL (Classical Diff) / MISSING (Change-VQA)",
  statusColor: C_AMBER,
  currentDoes: "Triggers pixel differencing pipeline when both change_image_a and change_image_b are supplied.",
  prdRequires: "Bi-temporal pixel diff heatmap generation combined with Change-VQA (answering natural language questions about what changed between T0 and T1).",
  exactChange: "Feed bi-temporal image pair and difference mask into a multimodal VLM to answer natural language questions about detected changes.",
  blockers: "Change-VLM model or dual-image prompt pipeline."
});

addAuditItem({
  requirement: "Optical + SAR Analysis Routing",
  file: "ml_backend/router.py (lines 190, 213), ml_backend/orchestrator.py",
  status: "PARTIALLY IMPLEMENTED (Dispatches separate text captions)",
  statusColor: C_AMBER,
  currentDoes: "Detects boolean flags has_optical and has_sar and dispatches separate calls to Optical_Caption and SAR_Caption.",
  prdRequires: "Co-registered dual-modal fusion that analyzes complementary strengths (optical multi-spectral reflectance + SAR backscatter/dielectric roughness).",
  exactChange: "Implement dual-channel ingestion and cross-modality reasoning pipeline.",
  blockers: "Paired co-registered optical/SAR sample datasets."
});

// -------------------------------------------------------------
// SECTION 2 CONT: CONTROLLER CONTINUED
// -------------------------------------------------------------
doc.addPage();
addHeader("1. Agent Controller Audit (Continued)", "Validation, Tool Execution, Output Fusion & Traces");

addAuditItem({
  requirement: "Input Validation",
  file: "ml_backend/server.py (decode_b64_image), app/api/query/route.ts",
  status: "PARTIALLY IMPLEMENTED (Format decode only)",
  statusColor: C_AMBER,
  currentDoes: "Validates base64 string decoding and handles empty query strings.",
  prdRequires: "Validation of raster format (GeoTIFF/JP2), spatial resolution/GSD, Coordinate Reference System (CRS), channel count, and coregistration alignment.",
  exactChange: "Add metadata validation schema (CRS check via rasterio.crs, spatial bounds matching, band validation).",
  blockers: "rasterio / GDAL in ml_backend."
});

addAuditItem({
  requirement: "Tool Selection & Task-Specific Parameter Selection",
  file: "ml_backend/router.py (route)",
  status: "REAL (Deterministic Rules)",
  statusColor: C_EMERALD,
  currentDoes: "Populates target_tools list (VQA, Optical_Caption, SAR_Caption, Change_Analysis) based on intent heuristics.",
  prdRequires: "Dynamic tool invocation with specialized parameter configurations (e.g. confidence thresholds, change sensitivity, target categories).",
  exactChange: "Pass runtime parameters (change_threshold, iou_threshold, spectral_indices) from the query down to tool invocations.",
  blockers: "Query frontend passing filter state in payload."
});

addAuditItem({
  requirement: "Specialist Tool Execution",
  file: "ml_backend/orchestrator.py (lines 64-78)",
  status: "PARTIALLY IMPLEMENTED (Synchronous execution)",
  statusColor: C_AMBER,
  currentDoes: "Iterates through target_tools synchronously and invokes tool instances.",
  prdRequires: "Async concurrent tool execution with timeout handling, GPU batch scheduling, and error isolation.",
  exactChange: "Wrap tool calls in asyncio.gather() with individual exception handling so one tool failure does not abort the entire pipeline.",
  blockers: "None."
});

addAuditItem({
  requirement: "Output Fusion & Synthesis",
  file: "ml_backend/orchestrator.py (_synthesize)",
  status: "REAL (Markdown Template Synthesis)",
  statusColor: C_EMERALD,
  currentDoes: "Combines optical description, SAR description, structured VQA bullet points, and change summary into a Markdown document.",
  prdRequires: "Unified synthesis resolving conflicts between tools and presenting structured intelligence findings.",
  exactChange: "Add conflict resolution logic (e.g., if optical caption says clear water but SAR shows high metallic backscatter, surface an explicit vessel anomaly note).",
  blockers: "None."
});

addAuditItem({
  requirement: "Confidence Handling & Uncertainty Calibration",
  file: "ml_backend/router.py, ml_backend/tools.py (VQAResult.low_confidence)",
  status: "PARTIALLY IMPLEMENTED (Hardcoded threshold logic)",
  statusColor: C_AMBER,
  currentDoes: "Flags queries with 'how many' with requires_count_warning = True and hardcodes confidence to 0.32 vs 0.85.",
  prdRequires: "Extraction of true model softmax probabilities, temperature scaling, and calibrated uncertainty intervals.",
  exactChange: "Extract model output token logits/probabilities from Hugging Face pipeline outputs and compute true confidence scores.",
  blockers: "Model execution with logits output enabled."
});

addAuditItem({
  requirement: "Execution Trace Generation",
  file: "server/services/queryService.ts (generateExecutionStages), app/api/query/[id]/trace/route.ts",
  status: "MOCK / SIMULATED",
  statusColor: C_RED,
  currentDoes: "Generates static simulated stages (Data Ingestion, Radiometric Correction, Neural Feature Extraction, etc.) with mock progress and timestamps.",
  prdRequires: "Real execution telemetry capturing actual execution stage start times, duration in ms, GPU memory allocations, and tensor shape metadata.",
  exactChange: "Instrument orchestrator.py and Next.js pipeline stages with real execution timers and telemetry log capture.",
  blockers: "None."
});

// -------------------------------------------------------------
// SECTION 3: SPECIALIST TOOLS AUDIT
// -------------------------------------------------------------
doc.addPage();
addHeader("2. Specialist Tools Audit", "VQA, Captioning, Change Analysis & Fusion Engines");

addAuditItem({
  requirement: "VQA Tool (RSVQA / PaliGemma)",
  file: "ml_backend/tools.py (VQATool)",
  status: "PARTIALLY IMPLEMENTED / FALLBACK MOCK",
  statusColor: C_AMBER,
  currentDoes: "Configured to load google/paligemma-3b-ft-rsvqa-lr-224 via Hugging Face pipeline('image-text-to-text'). If model load fails or weights are absent, returns deterministic mock answers ('yes', '12 (estimated)').",
  prdRequires: "Real tensor execution on optical/multispectral imagery answering closed-ended spatial presence, attribute, and count queries.",
  exactChange: "Download and cache model weights (or configure an API inference endpoint / ONNX runtime) and execute real inference.",
  blockers: "Hugging Face token (model is gated) and GPU/CPU RAM (~6GB VRAM for 3B FP16)."
});

addAuditItem({
  requirement: "Captioning / Scene Description Tool (BLIP)",
  file: "ml_backend/tools.py (CaptioningTool)",
  status: "PARTIALLY IMPLEMENTED / FALLBACK MOCK",
  statusColor: C_AMBER,
  currentDoes: "Configured for Salesforce/blip-image-captioning-base. Returns mock scene descriptions when weights are not downloaded locally.",
  prdRequires: "Automated natural language description of satellite tile land cover, infrastructure, and maritime activity.",
  exactChange: "Load BLIP / BLIP-2 checkpoint and run pipeline(image).",
  blockers: "transformers, torch, pillow installed in environment."
});

addAuditItem({
  requirement: "Change Analysis Tool",
  file: "ml_backend/change_analysis.py",
  status: "REAL AND WORKING (Classical Diff Engine)",
  statusColor: C_EMERALD,
  currentDoes: "Converts image pair to grayscale numpy arrays, calculates pixel intensity diff |B - A|, normalizes dynamic range, builds RGBA heatmap, alpha-blends overlay, calculates changed pixel fraction, and produces human-readable text summary.",
  prdRequires: "Co-registered bi-temporal change detection with pixel heatmap, delta percentage, and anomaly categorization.",
  exactChange: "Working as designed for classical diff. Needs connected anomaly clustering (contour extraction) to output labeled bounding boxes dynamically.",
  blockers: "None."
});

addAuditItem({
  requirement: "Optical + SAR Fusion Specialist Tool",
  file: "ml_backend/orchestrator.py",
  status: "MISSING (Only text concatenation exists)",
  statusColor: C_RED,
  currentDoes: "Calls optical tool and SAR tool independently and lists both text captions in markdown.",
  prdRequires: "Joint feature extraction or multi-sensor fusion model leveraging SAR backscatter penetration and optical spectral reflection.",
  exactChange: "Implement dual-channel feature merging or a multimodal prompt pipeline passing both sensor representations.",
  blockers: "Paired Optical/SAR training or inference model."
});

// -------------------------------------------------------------
// SECTION 4: INPUT & INGESTION AUDIT
// -------------------------------------------------------------
addSectionTitle("3. Input / Ingestion & Pipeline Audit");

addAuditItem({
  requirement: "Image Type & Modality Validation",
  file: "ml_backend/server.py, server/services/ingestionService.ts",
  status: "PARTIALLY IMPLEMENTED (Format check only)",
  statusColor: C_AMBER,
  currentDoes: "Verifies file extension (.tif, .jp2, .zip) in Next.js and base64 decode validity in FastAPI.",
  prdRequires: "Automated band detection (RGB, NIR, SWIR, SAR VV/VH), bit-depth normalization (uint8, uint16, float32), and modality tagging.",
  exactChange: "Inspect image channel headers via PIL.Image or rasterio and tag modality automatically.",
  blockers: "None."
});

addAuditItem({
  requirement: "Resolution / GSD & CRS Validation",
  file: "processing/pipeline.py, app/upload/page.tsx",
  status: "PARTIALLY IMPLEMENTED / MOCK",
  statusColor: C_AMBER,
  currentDoes: "Next.js upload checklist displays simulated CRS (EPSG:32651) and GSD (10m). processing/pipeline.py computes real NDVI math ((NIR - Red) / (NIR + Red)) but does not parse geospatial geotags.",
  prdRequires: "Extraction of real GeoTIFF affine transform matrices, ground sample distance (GSD), and re-projection to UTM / EPSG:4326.",
  exactChange: "Utilize rasterio.open() in processing/main.py to extract real CRS and resolution metadata.",
  blockers: "rasterio package in processing service."
});

addAuditItem({
  requirement: "Paired-Image & Co-Registration Validation",
  file: "ml_backend/change_analysis.py (lines 62-67), processing/pipeline.py",
  status: "PARTIALLY IMPLEMENTED",
  statusColor: C_AMBER,
  currentDoes: "Checks image_a.size == image_b.size and resizes B to match A if dimensions differ.",
  prdRequires: "Feature matching (e.g. ORB / SIFT + RANSAC homography) to compute real co-registration RMS error and warp images prior to differencing.",
  exactChange: "Implement OpenCV ORB feature alignment in change_analysis.py to calculate true coregistration RMS pixel error.",
  blockers: "opencv-python-headless."
});

// -------------------------------------------------------------
// SECTION 5: CONCLUSION & RECOMMENDATIONS
// -------------------------------------------------------------
doc.addPage();
addHeader("4. Executive Classification & Recommended Roadmap", "Categorized status and implementation order");

addSectionTitle("1. Already Working");
addParagraph("• Classical Pixel-Level Change Analysis (ml_backend/change_analysis.py): Fully functional numpy differencing, dynamic range normalization, RGBA heatmap creation, and alpha overlay generation.");
addParagraph("• Deterministic Query Structuring & Decomposition (ml_backend/router.py): Rule-based normalization of closed-ended and counting queries into RSVQA format with structured sub-questions.");
addParagraph("• Low-Confidence Warning Flagging (ml_backend/router.py): Automatically flags queries with 'how many' and injects estimation warnings into synthesized output.");
addParagraph("• FastAPI Query & Change Analysis API (ml_backend/server.py): Functional REST API endpoints (/api/query, /api/change-analysis, /api/health) with base64 image decoding/encoding.");
addParagraph("• Next.js API Fallback Layer (app/api/query/route.ts): Reliably calls FastAPI backend with a 3-second timeout and fails gracefully to embedded router logic when backend is offline.");
addParagraph("• Spectral NDVI Computation (processing/pipeline.py): Real mathematical implementation of normalized difference vegetation index on multi-band numpy arrays.");

addSectionTitle("2. Needs Modification");
addParagraph("• ml_backend/tools.py: Download/cache real model checkpoints or connect an active inference runtime so real VQA and Captioning inference execute instead of falling back to default strings.");
addParagraph("• ml_backend/change_analysis.py: Add OpenCV ORB feature alignment for true co-registration validation and extract bounding boxes from changed connected components.");
addParagraph("• app/api/query/route.ts & app/components/QueryPage.tsx: Connect frontend advanced filter controls (date range, sensor modality, resolution, cloud cover) so they are passed in the request body to the backend.");
addParagraph("• server/services/queryService.ts: Replace hardcoded execution stages with real timing and telemetry instrumentation.");

addSectionTitle("3. Missing");
addParagraph("• Change-VQA: Multi-modal reasoning over bi-temporal pairs to answer textual questions about what changed.");
addParagraph("• Optical + SAR Feature Fusion: Joint multi-sensor representation model (currently only text output concatenation exists).");
addParagraph("• True Model Confidence Calculation: Extracting real token logits/probabilities from model outputs rather than using hardcoded values (0.32 / 0.85).");
addParagraph("• Real Object Detection / Grounding: YOLOv8 or Grounding-DINO inference (currently simulated via detect_objects_placeholder).");
addParagraph("• GeoTIFF / CRS Metadata Extraction: Parsing affine transforms and EPSG projection headers from GeoTIFF files via rasterio.");

addSectionTitle("4. Blocked by Another Team / Component");
addParagraph("• Real DB Persistence: Prisma ORM schema is complete in prisma/schema.prisma, but blocked by missing DATABASE_URL / .env configuration and inactive PostgreSQL instance.");
addParagraph("• Live Model Inference: Blocked by missing Hugging Face authentication token for the gated PaliGemma model (google/paligemma-3b-ft-rsvqa-lr-224) and local GPU/VRAM setup.");
addParagraph("• Unified Full-Stack Startup: Next.js (npm run dev) and FastAPI (uvicorn server:app) must be run concurrently.");

addSectionTitle("5. Recommended Implementation Order");
addParagraph("1. Step 1: Environment & Full-Stack Startup: Configure .env.local and add a unified dev script (npm run dev starting both Next.js and FastAPI) so communication is live.");
addParagraph("2. Step 2: Frontend Filter Connection: Update QueryPage.tsx to pass selected filter state to /api/query and router.py.");
addParagraph("3. Step 3: Specialist Tool Weights & Model Runtime: Configure model downloading/caching in tools.py with real logit confidence scoring.");
addParagraph("4. Step 4: Dynamic Anomaly Extraction & Alignment: Add OpenCV co-registration and contour detection in change_analysis.py to output real anomaly bounding boxes.");
addParagraph("5. Step 5: Real Execution Telemetry: Instrument orchestrator.py to record real stage durations and pipe them into the execution trace views.");
addParagraph("6. Step 6: Database Integration: Initialize PostgreSQL container via docker-compose.yml and run npx prisma db push to enable persistence.");

// Finalize PDF
doc.end();
console.log("Audit PDF generated successfully at:", publicPath);
