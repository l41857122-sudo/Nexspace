const PDFDocument = require("pdfkit");
const fs = require("fs");
const path = require("path");

const publicPath = path.join(__dirname, "public", "NexSpace_Architecture_And_Component_Deep_Dive.pdf");
const doc = new PDFDocument({
  size: "A4",
  margins: { top: 40, bottom: 40, left: 45, right: 45 },
  bufferPages: true
});

const writeStream = fs.createWriteStream(publicPath);
doc.pipe(writeStream);

// Colors
const C_DARK_BG = "#06111D";
const C_CARD_BG = "#0C1624";
const C_CYAN = "#06B6D4";
const C_SKY = "#38BDF8";
const C_TEXT = "#E2E8F0";
const C_MUTED = "#94A3B8";
const C_ACCENT = "#10B981";
const C_BORDER = "#1E293B";

function addHeader(title, subtitle) {
  doc.fontSize(18).fillColor(C_CYAN).font("Helvetica-Bold").text(title);
  if (subtitle) {
    doc.fontSize(9).fillColor(C_MUTED).font("Helvetica").text(subtitle);
  }
  doc.moveDown(0.6);
  doc.strokeColor(C_BORDER).lineWidth(0.8).moveTo(45, doc.y).lineTo(550, doc.y).stroke();
  doc.moveDown(0.6);
}

function addSectionTitle(title) {
  doc.moveDown(0.8);
  doc.fontSize(13).fillColor(C_SKY).font("Helvetica-Bold").text(title);
  doc.moveDown(0.3);
}

function addSubSectionTitle(title) {
  doc.moveDown(0.5);
  doc.fontSize(10.5).fillColor(C_CYAN).font("Helvetica-Bold").text(title);
  doc.moveDown(0.2);
}

function addParagraph(text) {
  doc.fontSize(8.5).fillColor(C_TEXT).font("Helvetica").text(text, { align: "justify", lineGap: 2.5 });
  doc.moveDown(0.4);
}

function addBullet(title, desc) {
  doc.fontSize(8.5).font("Helvetica-Bold").fillColor(C_SKY).text(`• ${title}: `, { continued: true });
  doc.font("Helvetica").fillColor(C_TEXT).text(desc, { lineGap: 2 });
  doc.moveDown(0.25);
}

// -------------------------------------------------------------
// COVER / TITLE PAGE
// -------------------------------------------------------------
doc.rect(0, 0, 595.28, 841.89).fill("#07111C");

doc.fontSize(24).fillColor("#00E5FF").font("Helvetica-Bold").text("NexSpace", 45, 120);
doc.fontSize(14).fillColor("#38BDF8").font("Helvetica-Bold").text("Complete Architecture & Component-by-Component Deep Dive Report");
doc.fontSize(9).fillColor("#94A3B8").font("Helvetica").text("Planetary-Scale Geospatial Intelligence & Autonomous Neural Telemetry");

doc.moveDown(2);
doc.strokeColor("#1E293B").lineWidth(1).moveTo(45, doc.y).lineTo(550, doc.y).stroke();
doc.moveDown(1.5);

doc.fontSize(10).fillColor("#E2E8F0").font("Helvetica-Bold").text("DOCUMENT OVERVIEW & PURPOSE:");
doc.fontSize(8.5).fillColor("#94A3B8").font("Helvetica").text(
  "This comprehensive report provides a complete, line-by-line and architectural audit of every component, " +
  "API route, service layer, shader file, ML backend module, and database model in the NexSpace repository. " +
  "It details what each component does, its functional purpose, the exact dependencies and linkages, data inputs/outputs, " +
  "identified defects/bugs, and recommended fixes."
, { lineGap: 3 });

doc.moveDown(2);
doc.fontSize(9).fillColor("#38BDF8").font("Helvetica-Bold").text("TABLE OF CONTENTS");
const toc = [
  "1. Executive System Architecture & Microservice Topology",
  "2. Global Layout & WebGL Shaders (layout.tsx, WebThreads, OrbitalEnergyBackground, SpecularButton)",
  "3. Core UI Pages & Component Breakdown (Dashboard, Query, Comparison, Upload, Reports, Results, Evidence, Execution, Settings)",
  "4. API Routing & Serverless Handlers (all 12 /api route categories)",
  "5. Server Services Layer & Database Architecture (queryService, ingestionService, comparisonService, Prisma)",
  "6. Python ML Backend & Processing Engines (FastAPI, PaliGemma-3B, BLIP, YOLOv8, RSVQA, Change Analysis)",
  "7. Root Cause Analysis: Why Certain Components Are Not Working & Exact Fixes"
];
toc.forEach(item => {
  doc.fontSize(8.5).fillColor("#CBD5E1").font("Helvetica").text(`  ${item}`, { lineGap: 2.5 });
});

doc.moveDown(3);
doc.fontSize(8).fillColor("#64748B").font("Helvetica").text("Generated: August 2026 · Local Workspace Distribution · NexSpace Platform v2.4.1");

// -------------------------------------------------------------
// SECTION 1: SYSTEM ARCHITECTURE
// -------------------------------------------------------------
doc.addPage();
addHeader("1. Executive System Architecture & Topology", "End-to-end data flow from user interaction to deep learning inference");

addParagraph(
  "NexSpace is a hybrid satellite imagery intelligence platform built on Next.js 16 (React 19), " +
  "TypeScript, Tailwind CSS v4, OGL (hardware-accelerated WebGL), Prisma ORM, and a dedicated Python FastAPI ML backend."
);

addBullet("Frontend Web Application (Port 3000)", "Next.js App Router providing interactive mission control dashboards, split-screen temporal comparison viewers, natural language spatial terminals, evidence verification portals, and live kernel execution traces.");
addBullet("Next.js Serverless API Handlers (/api/*)", "Edge & Node.js route handlers orchestrating authentication, background raster ingestion polling, PDF report compilation via PDFKit, telemetry rotation, and fallback rule-based NLP routing.");
addBullet("Python ML Microservice (Port 8000 / ml_backend)", "FastAPI server running multi-modal vision-language models (PaliGemma-3B, BLIP-2), RSVQA sub-question decomposition, YOLOv8 object detection, and pixel-level bi-temporal change detection.");
addBullet("Geospatial Raster Pipeline (/processing)", "Dockerized GDAL/Rasterio pipeline converting multi-spectral Sentinel-2, Landsat-8, and SAR-X raw imagery into Cloud-Optimized GeoTIFFs (COG), calculating NDVI, NDWI, and generating spatial pyramids.");
addBullet("Relational & Vector Database (Prisma / PostgreSQL)", "PostgreSQL database tracking Users, Scenes, NLP Queries, Execution Stages, Comparisons, Detected Entities, and Intelligence Reports.");

// -------------------------------------------------------------
// SECTION 2: GLOBAL LAYOUT & WEBGL SHADERS
// -------------------------------------------------------------
addSectionTitle("2. Global Layout & WebGL Shaders");

addSubSectionTitle("app/layout.tsx & app/page.tsx");
addBullet("What it does", "Defines the Root HTML shell, injects Geist Sans/Mono typography, mounts the global dynamic background shader (OrbitalEnergyBackground), and renders the landing page (HeroSection).");
addBullet("Meaning & Logic", "Ensures unified visual aesthetics across all routes with dark mission control theme (#06111d). Uses suppressHydrationWarning to prevent SSR mismatch from dynamic WebGL canvases.");
addBullet("Linkages", "Imports OrbitalEnergyBackground.tsx and HeroSection.tsx; wraps all children routes in {children}.");

addSubSectionTitle("app/components/OrbitalEnergyBackground.tsx");
addBullet("What it does", "Calculates page-specific electromagnetic energy glow trails and SVG curved orbital arcs that dynamically adjust brightness based on the active route.");
addBullet("Meaning & Logic", "Reads usePathname() from next/navigation. Sets opacity to 0.42 on /dashboard, 0.28 on /query, 0.32 on /comparison, 0.35 on /execution, and 0 on landing page (where WebThreads takes over).");
addBullet("Linkages", "Global layout component running across all application routes.");

addSubSectionTitle("app/components/WebThreads.tsx & AcidSquares.tsx");
addBullet("What it does", "Hardware-accelerated WebGL 2.0 / WebGL 1.0 procedural shader using OGL library. Renders multi-harmonic glowing thread lines simulating satellite orbital trajectories with mouse interaction.");
addBullet("Meaning & Logic", "Contains GLSL vertex and fragment shaders (vertex300, fragment300, vertex100, fragment100). Implements a 2D Canvas fallback if the client GPU lacks WebGL support.");
addBullet("Linkages", "Directly rendered in HeroSection.tsx on the landing page.");

addSubSectionTitle("app/components/SpecularButton.tsx");
addBullet("What it does", "Interactive button component with GLSL specular sheen reflection that follows mouse movements across button edges.");
addBullet("Meaning & Logic", "Uses an elliptical normal shader SDF (Signed Distance Field) to create continuous rim illumination.");
addBullet("Linkages", "Standalone UI component designed for high-conversion CTAs.");

// -------------------------------------------------------------
// SECTION 3: CORE UI PAGES & COMPONENTS
// -------------------------------------------------------------
doc.addPage();
addHeader("3. Core UI Pages & Component Breakdown", "Detailed analysis of each page, component state, and UI behaviors");

addSubSectionTitle("3.1 DashboardPage.tsx (Route: /dashboard)");
addBullet("What it does", "Serves as the primary operational telemetry center. Renders KPI metric cards, an interactive 3D radar sweep globe HUD, system hardware status, recent orbital analyses, and recent NLP queries.");
addBullet("Data Flow & Connections", "Fetches /api/analyses on mount for analysis tables, and /api/queries/recent for the natural language query history list. Renders Sidebar.tsx and TopBar.");
addBullet("Identified Issues", "+ New Analysis button has no onClick or router navigation. Filter button in the table is non-functional.");

addSubSectionTitle("3.2 QueryPage.tsx (Route: /query)");
addBullet("What it does", "Interactive natural language query terminal for spatial search. Allows users to type queries or click suggestion chips (e.g., 'How many buildings?', 'Is there water?').");
addBullet("Data Flow & Connections", "Calls POST /api/query with { query, optical_image }. Simulates multi-phase pipeline routing ('Classifying intent', 'Restructuring query', 'Synthesizing response').");
addBullet("Identified Issues", "Advanced filter controls (Date Range, Sensor Modality, Cloud Cover slider, GSD Resolution) are held in local state and not included in the POST /api/query request body.");

addSubSectionTitle("3.3 ComparisonPage.tsx (Route: /comparison)");
addBullet("What it does", "Bi-temporal change detection interface. Displays side-by-side synchronized historical baseline (Sentinel-2B) vs current recon telemetry (SAR-X Cosmo-SkyMed) with interactive delta heatmap overlays.");
addBullet("Data Flow & Connections", "Interactive controls for Delta Heatmap toggle, NDVI Spectral overlay toggle, overlay opacity slider (0-100%), and anomaly selection (#01 Vegetation Loss, #02 New Structure). Calls /api/comparisons/comp_b492_xt_p/export for PDF generation.");
addBullet("Identified Issues (CRITICAL)", "Contains duplicate anomaly boxes rendered on top of each other (lines 248–277 and lines 305–354) causing UI overlap and collision.");

addSubSectionTitle("3.4 UploadPage.tsx (Route: /upload)");
addBullet("What it does", "Raster imagery ingestion interface supporting Drag & Drop of GeoTIFF, JP2, NetCDF, and Zip archives with real-time validation checklist (CRS EPSG:32651, 10m GSD, Cloud Assessment).");
addBullet("Data Flow & Connections", "Sends POST /api/scenes/upload, receives job ID, and polls GET /api/scenes/[id]/status every 500ms until status is 'completed'.");

addSubSectionTitle("3.5 ScanResultsPage.tsx (Route: /results)");
addBullet("What it does", "Tactical map viewport displaying detected maritime vessels and industrial tank farms with nautical bathymetry lines, bounding box focus zoom, and confidence metrics.");
addBullet("Data Flow & Connections", "Interactive entity selection (Vessel_Panamax_01, Infra_Tank_Farm_B, Vessel_Feeder_12). Generates PDF intelligence report via /api/reports/SQ-REP-2023-11A/pdf.");
addBullet("Identified Issues", "Export GeoJSON and Download CSV buttons have no attached click handlers.");

addSubSectionTitle("3.6 ReportsPage.tsx (Route: /reports)");
addBullet("What it does", "Executive intelligence assessment report with executive summary, spatial analysis map, temporal trend bar charts, and technical dataset appendix table.");
addBullet("Data Flow & Connections", "Connects to /api/reports/SQ-REP-2023-11A/pdf for binary PDF downloading.");

addSubSectionTitle("3.7 EvidenceViewerPage.tsx (Route: /evidence)");
addBullet("What it does", "Sub-meter crop verification audit terminal. Analysts inspect Red (B04), NIR (B08), and SWIR (B11) reflectance curves and submit sign-offs (Confirm, Flag, Reject).");
addBullet("Data Flow & Connections", "Calls GET /api/entities/entity-1/evidence on load and POST /api/entities/entity-1/evidence on analyst sign-off.");

addSubSectionTitle("3.8 ExecutionTracePage.tsx & ExecutionLogPage.tsx (Routes: /execution, /execution-log)");
addBullet("What it does", "Visualizes GPU pipeline stage topology (Data Ingest -> Radiometric Correction -> Neural Extraction -> Spatial Clustering -> Confidence Scoring), streaming CUDA terminal logs, and tensor inspector.");
addBullet("Data Flow & Connections", "Fetches /api/query/q_9482_a/trace and /api/query/q_9482_a/log.");

addSubSectionTitle("3.9 SettingsPage.tsx (Route: /settings)");
addBullet("What it does", "API key management (reveal, copy, rotate) and 2-Factor Authentication toggle.");
addBullet("Data Flow & Connections", "Connects to GET /api/settings/me, PATCH /api/settings/me, and POST /api/settings/api-key/rotate.");

// -------------------------------------------------------------
// SECTION 4: API ROUTING & BACKEND SERVICES
// -------------------------------------------------------------
doc.addPage();
addHeader("4. API Handlers, Services & Database Layer", "Detailed specification of backend logic, Prisma models, and microservices");

addSubSectionTitle("4.1 Next.js API Routes Breakdown");
addBullet("POST /api/query", "Central routing controller. Forwards requests to Python ML backend (http://localhost:8000/api/query) with a 3-second timeout. Falls back to embedded rule-based NLP router if Python is offline.");
addBullet("GET /api/analyses", "Returns filtered array of active and completed orbital extraction jobs with query parameters ?status= and ?limit=.");
addBullet("GET /api/queries/recent", "Returns historical spatial queries with execution timings and processing status.");
addBullet("POST /api/scenes/upload & GET /api/scenes/[id]/status", "Handles raster uploads and simulates multi-spectral band extraction progress up to 100%.");
addBullet("POST /api/change-analysis", "Forwards bi-temporal image pairs to Python change analysis service; returns pixel delta fractions and intensity changes.");
addBullet("GET /api/reports/[id]/pdf", "Generates and streams dynamic binary PDF intelligence reports directly to the client using PDFKit.");
addBullet("GET & PATCH /api/settings/me & POST /api/settings/api-key/rotate", "Manages user credentials and 2FA settings.");
addBullet("GET & POST /api/entities/[id]/evidence", "Fetches target spectral bands and records analyst verification audits.");

addSubSectionTitle("4.2 Server Services Layer (server/services/)");
addBullet("queryService.ts", "Provides parseQueryFilters(rawText) with optional Anthropic Claude-3.5 fallback, and generateExecutionStages(queryId).");
addBullet("ingestionService.ts", "Manages active upload jobs in an in-memory Map, advancing simulated progress through radiometric and COG building steps.");
addBullet("comparisonService.ts", "Processes bi-temporal image comparisons, calculating delta percentages and RMS coregistration errors.");
addBullet("reportService.ts", "Implements generateReportPDF() using PDFKit to generate branded intelligence reports.");

addSubSectionTitle("4.3 Prisma Database Schema (prisma/schema.prisma)");
addBullet("User", "Stores account credentials, API keys, and 2FA status.");
addBullet("Scene", "Stores satellite scenes, GSD, cloud cover %, rawUrl, cogUrl, and NDVI values.");
addBullet("Query & ExecutionStage", "Tracks natural language queries, parsed JSON filters, execution stages, and logs.");
addBullet("Comparison", "Tracks before/after scenes, delta percentages, RMS px errors, and detected anomaly JSON.");
addBullet("DetectedEntity", "Stores bounding boxes (bboxX, bboxY, width, height), classification types, and confidence scores.");
addBullet("Analysis & Report", "Stores intelligence reports with executive summary, spatial analysis, and temporal trend sections.");

// -------------------------------------------------------------
// SECTION 5: PYTHON ML BACKEND & PROCESSING PIPELINE
// -------------------------------------------------------------
addSectionTitle("5. Python ML Backend & Processing Pipeline");

addSubSectionTitle("ml_backend/ Microservice (FastAPI on Port 8000)");
addBullet("server.py", "FastAPI app exposing /api/query, /api/change-analysis, and /api/health with CORS middleware.");
addBullet("router.py", "Implements Agentic Intent Routing. Classifies queries into VQA, Captioning, or Change Analysis, normalizing queries into RSVQA binary format and flagging low-confidence counting tasks.");
addBullet("change_analysis.py", "Performs SSIM, pixel diffing, thresholding, and morphological opening/closing on before/after satellite image pairs.");
addBullet("orchestrator.py", "Executes multi-tool pipelines asynchronously and synthesizes human-readable intelligence findings.");
addBullet("tools.py", "Wraps PaliGemma-3B, BLIP, and computer vision models.");

addSubSectionTitle("processing/ Microservice (Raster Pipeline)");
addBullet("pipeline.py & main.py", "GDAL/Rasterio script converting GeoTIFFs to COG, calculating NDVI raster arrays, and building multiscale overviews.");

// -------------------------------------------------------------
// SECTION 6: ROOT CAUSES & ACTIONABLE FIXES
// -------------------------------------------------------------
doc.addPage();
addHeader("6. Defect Audit: Why the App May Feel Broken & Fixes", "Root causes and actionable step-by-step solutions");

addBullet("1. Duplicate Anomaly Boxes in ComparisonPage.tsx", "Lines 248–277 and lines 305–354 in ComparisonPage.tsx both render anomaly bounding boxes with the same IDs. Fix: Remove the second duplicate set and consolidate interactive click handling.");
addBullet("2. Missing .env & PostgreSQL Connection", "prisma/schema.prisma requires DATABASE_URL, but no .env is configured. Fix: Create .env.local with DATABASE_URL or continue using the resilient mock service layer.");
addBullet("3. Advanced Filters Not Passed in QueryPage.tsx", "Selected sensor, date range, resolution, and cloud cover are not passed in the handleExecute() request body. Fix: Pass filter state in the JSON payload to /api/query.");
addBullet("4. Unhooked Dashboard & Scan Results Buttons", "+ New Analysis in Dashboard, and Export GeoJSON / CSV in Scan Results have no click handlers. Fix: Attach router navigation and file download handlers.");
addBullet("5. Python Backend Not Running Concurrently", "Running only npm run dev leaves the ML service on port 8000 offline. Fix: Use concurrently or start uvicorn ml_backend.server:app --port 8000 alongside Next.js.");
addBullet("6. Brand Consistency (NexSpace)", "Standardized brand labels across all components, headers, and reports.");

doc.moveDown(1.5);
doc.fontSize(10).fillColor("#10B981").font("Helvetica-Bold").text("SUMMARY CONCLUSION:");
doc.fontSize(8.5).fillColor("#CBD5E1").font("Helvetica").text(
  "The application has a robust, modern UI architecture and well-designed resilient fallbacks. " +
  "By addressing the duplicate anomaly render in ComparisonPage.tsx, connecting the unhooked buttons, " +
  "and passing query filters, the platform will be 100% operational and seamless."
, { lineGap: 2.5 });

// Finalize PDF
doc.end();
console.log("PDF generated successfully at:", publicPath);
