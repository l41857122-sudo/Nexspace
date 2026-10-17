/**
 * NexSpace AI Assistant — Knowledge Base
 *
 * This module contains VERIFIED facts about the NexSpace / SatQuery AI platform.
 *
 * Implementation Status Tags:
 *   IMPLEMENTED           — feature confirmed present in the codebase / validated
 *   PARTIALLY_IMPLEMENTED — feature exists but is incomplete or limited
 *   EXPERIMENTAL          — prototype / exploratory code; not production-validated
 *   UNAVAILABLE           — not present in the current codebase
 *   NOT_VERIFIED          — status unknown; do not claim either way
 *
 * The assistant MUST NOT assert capabilities that are not marked IMPLEMENTED.
 */

// ---------------------------------------------------------------------------
// Project overview
// ---------------------------------------------------------------------------

export const PROJECT_OVERVIEW = {
  name: "NexSpace / SatQuery AI",
  tagline:
    "Natural-language satellite and geospatial intelligence platform.",
  description:
    "NexSpace is a platform that lets users investigate satellite imagery " +
    "using ordinary language and converts those queries into specialised " +
    "remote-sensing analysis. It is NOT a generic chatbot — its core " +
    "function is automated geospatial intelligence.",
} as const;

// ---------------------------------------------------------------------------
// Investigation workflow (IMPLEMENTED)
// ---------------------------------------------------------------------------

export const INVESTIGATION_WORKFLOW = [
  "User submits a natural-language query",
  "Query Understanding — intent, targets, location/time, imagery requirements, spatial relationships extracted",
  "Validation — input checked for completeness and routing suitability",
  "Satellite Discovery — appropriate imagery located via STAC infrastructure",
  "Routing — query dispatched to the relevant specialist analysis component",
  "Specialist Analysis — models run (detection, captioning, region analysis, multispectral, change, fusion)",
  "Georeferencing — results anchored to real-world coordinates",
  "Evidence — detection boxes, region masks, and metrics extracted",
  "Provenance — execution trace and data lineage recorded",
  "Explainable Result — findings returned with supporting evidence and provenance",
] as const;

// ---------------------------------------------------------------------------
// Capabilities with implementation status
// ---------------------------------------------------------------------------

export interface Capability {
  name: string;
  status: "IMPLEMENTED" | "PARTIALLY_IMPLEMENTED" | "EXPERIMENTAL" | "UNAVAILABLE" | "NOT_VERIFIED";
  detail: string;
}

export const CAPABILITIES: Capability[] = [
  {
    name: "Natural-language satellite queries",
    status: "IMPLEMENTED",
    detail:
      "Users can express investigation goals in plain English. The query is parsed into intent, targets, location, time requirements, and spatial relationships before routing.",
  },
  {
    name: "Satellite imagery discovery",
    status: "IMPLEMENTED",
    detail:
      "NexSpace discovers imagery via STAC infrastructure, primarily Element84 Earth Search and Sentinel-2 L2A.",
  },
  {
    name: "Sentinel-2 L2A imagery",
    status: "IMPLEMENTED",
    detail:
      "Sentinel-2 Level-2A (surface reflectance) is the primary confirmed satellite data source.",
  },
  {
    name: "Object detection",
    status: "IMPLEMENTED",
    detail:
      "Grounding DINO is used for open-vocabulary object detection. Outputs include bounding boxes in normalised coordinate space with geospatial metadata where available.",
  },
  {
    name: "Region analysis",
    status: "IMPLEMENTED",
    detail:
      "NexSpace can analyse specific regions such as water bodies and vegetation through a dedicated region-analysis pipeline.",
  },
  {
    name: "Scene description / captioning",
    status: "IMPLEMENTED",
    detail:
      "BLIP is used for scene captioning and visual question answering (VQA).",
  },
  {
    name: "Change analysis",
    status: "IMPLEMENTED",
    detail:
      "A change-analysis pipeline compares compatible imagery pairs to identify changes or anomalous regions.",
  },
  {
    name: "NDVI (Normalised Difference Vegetation Index)",
    status: "IMPLEMENTED",
    detail:
      "NDVI is computed from genuine Sentinel-2 near-infrared and red bands. NexSpace does NOT simulate NDVI from RGB images.",
  },
  {
    name: "NDWI (Normalised Difference Water Index)",
    status: "IMPLEMENTED",
    detail:
      "NDWI is computed from genuine Sentinel-2 multispectral bands when available.",
  },
  {
    name: "NDBI (Normalised Difference Built-up Index)",
    status: "IMPLEMENTED",
    detail:
      "NDBI is computed from genuine Sentinel-2 multispectral bands when available.",
  },
  {
    name: "True multispectral analysis",
    status: "IMPLEMENTED",
    detail:
      "When real Sentinel-2 bands are available, NexSpace processes actual spectral data rather than approximating from RGB.",
  },
  {
    name: "Optical + SAR feature fusion",
    status: "IMPLEMENTED",
    detail:
      "NexSpace can align independent optical and SAR imagery via CRS-aware co-registration and GSD harmonisation, then combine their learned feature representations into a 1536-dimensional fused representation. " +
      "IMPORTANT CAVEAT: Current task-level validation established parity with the evaluated single-modality baselines. A general accuracy improvement over single-modality has NOT been established.",
  },
  {
    name: "CRS-aware geospatial co-registration",
    status: "IMPLEMENTED",
    detail:
      "Optical and SAR imagery are co-registered in a shared coordinate reference system using affine transforms and GSD harmonisation before feature fusion.",
  },
  {
    name: "Spatial relationships",
    status: "IMPLEMENTED",
    detail:
      "The investigation pipeline can reason about spatial relationships between detected entities.",
  },
  {
    name: "GeoJSON / geospatial outputs",
    status: "IMPLEMENTED",
    detail:
      "Results can include GeoJSON geometries anchored to real-world coordinates.",
  },
  {
    name: "Evidence and provenance tracking",
    status: "IMPLEMENTED",
    detail:
      "Each investigation result exposes an evidence record and provenance describing how it was obtained, including execution trace information.",
  },
  {
    name: "Universal satellite support (non-Sentinel)",
    status: "NOT_VERIFIED",
    detail:
      "NexSpace is validated primarily with Sentinel-2 L2A. Support for other satellite platforms (e.g. Landsat, Planet, WorldView) has not been verified.",
  },
  {
    name: "SAM-2 (Segment Anything Model v2)",
    status: "NOT_VERIFIED",
    detail:
      "SAM-2 usage in NexSpace has not been confirmed in the repository.",
  },
  {
    name: "Real-time / live satellite feeds",
    status: "UNAVAILABLE",
    detail:
      "NexSpace uses archive STAC imagery. Live streaming satellite feeds are not currently implemented.",
  },
  {
    name: "User accounts / persistent sessions",
    status: "PARTIALLY_IMPLEMENTED",
    detail:
      "Authentication infrastructure (Prisma, JWT) exists. Long-term user profiles and persistent investigation history are partially implemented.",
  },
];

// ---------------------------------------------------------------------------
// Tech stack
// ---------------------------------------------------------------------------

export const TECH_STACK = {
  frontend: ["Next.js", "React", "TypeScript", "Tailwind CSS", "GSAP", "OGL / WebGL"],
  backend: ["Python", "FastAPI", "REST APIs"],
  ml: ["PyTorch", "Transformers", "PEFT-LoRA"],
  remoteSensing: ["GeoTIFF / TIFF", "CRS / Affine transforms", "WGS84", "STAC", "Tiling", "NMS"],
  satellite: ["Element84 Earth Search", "Sentinel-2 L2A", "Optical imagery", "SAR imagery"],
} as const;

// ---------------------------------------------------------------------------
// Specialist components
// ---------------------------------------------------------------------------

export const SPECIALIST_COMPONENTS = [
  { name: "Grounding DINO", role: "Open-vocabulary object detection" },
  { name: "BLIP", role: "Scene captioning and visual question answering (VQA)" },
  { name: "Region Analysis", role: "Pixel-level region segmentation and spectral analysis" },
  { name: "Change Analysis", role: "Temporal change detection between imagery pairs" },
  { name: "Multispectral Analysis", role: "Band-level spectral index computation (NDVI, NDWI, NDBI)" },
  { name: "Optical-SAR Fusion", role: "Cross-modal feature fusion with CRS-aware co-registration" },
  { name: "Spatial Reasoning", role: "Spatial relationship extraction between detected entities" },
  { name: "Geospatial Co-registration", role: "Alignment of imagery in a shared CRS before fusion" },
] as const;

// ---------------------------------------------------------------------------
// FAQ knowledge entries
// ---------------------------------------------------------------------------

export interface FAQEntry {
  id: string;
  question: string;
  answer: string;
  tags: string[];
}

export const FAQ_ENTRIES: FAQEntry[] = [
  {
    id: "what-is-nexspace",
    question: "What is NexSpace?",
    answer:
      "NexSpace is a natural-language satellite and geospatial intelligence platform. " +
      "It lets users investigate satellite imagery using ordinary language and converts those " +
      "queries into specialised remote-sensing analysis — including object detection, spectral " +
      "indices, change analysis, and Optical + SAR fusion.",
    tags: ["overview", "intro", "platform"],
  },
  {
    id: "what-can-nexspace-do",
    question: "What can NexSpace do?",
    answer:
      "NexSpace can:\n" +
      "• Discover and analyse Sentinel-2 satellite imagery via natural-language queries\n" +
      "• Detect objects using Grounding DINO (open-vocabulary detection)\n" +
      "• Analyse regions such as water bodies and vegetation\n" +
      "• Compute NDVI, NDWI, and NDBI from real multispectral bands\n" +
      "• Detect changes between imagery pairs\n" +
      "• Fuse Optical and SAR features into a shared 1536-d representation\n" +
      "• Return geospatially anchored results with evidence and provenance",
    tags: ["capabilities", "overview", "features"],
  },
  {
    id: "how-does-it-work",
    question: "How does NexSpace work?",
    answer:
      "A user submits a natural-language query. NexSpace:\n" +
      "1. Parses the query into intent, targets, location/time, and imagery requirements\n" +
      "2. Validates the request and selects the relevant analysis pipeline\n" +
      "3. Discovers matching satellite imagery via STAC infrastructure\n" +
      "4. Runs specialist ML models (detection, captioning, spectral analysis, etc.)\n" +
      "5. Georeferences results to real-world coordinates\n" +
      "6. Returns findings with evidence and a full provenance trace",
    tags: ["workflow", "pipeline", "how-it-works"],
  },
  {
    id: "satellite-data",
    question: "What satellite data does NexSpace use?",
    answer:
      "NexSpace primarily uses:\n" +
      "• Sentinel-2 L2A (surface reflectance) — the main validated data source\n" +
      "• Element84 Earth Search — STAC-compatible imagery catalogue\n\n" +
      "Support for other satellites (Landsat, Planet, WorldView, etc.) has not been verified.",
    tags: ["data", "satellite", "sentinel", "stac"],
  },
  {
    id: "object-detection",
    question: "What can NexSpace detect?",
    answer:
      "NexSpace uses Grounding DINO for open-vocabulary object detection. It can detect a wide " +
      "range of targets expressed in natural language — vessels, buildings, infrastructure, " +
      "vegetation patches, and more.\n\n" +
      "Detection outputs include normalised bounding boxes with geospatial metadata where available.",
    tags: ["detection", "grounding-dino", "objects"],
  },
  {
    id: "ndvi",
    question: "Can NexSpace calculate NDVI?",
    answer:
      "Yes — when genuine Sentinel-2 multispectral imagery containing the required bands is available.\n\n" +
      "NexSpace computes NDVI from real near-infrared (Band 8) and red (Band 4) reflectance " +
      "values. It does NOT simulate or approximate NDVI from an RGB image.",
    tags: ["ndvi", "multispectral", "vegetation", "spectral"],
  },
  {
    id: "ndwi-ndbi",
    question: "What is NDWI and NDBI?",
    answer:
      "Both are spectral indices NexSpace can compute from real Sentinel-2 bands:\n\n" +
      "• NDWI (Normalised Difference Water Index) — highlights water bodies using green and NIR bands\n" +
      "• NDBI (Normalised Difference Built-up Index) — highlights urban/built-up areas using SWIR and NIR bands\n\n" +
      "As with NDVI, these require genuine multispectral imagery, not RGB approximations.",
    tags: ["ndwi", "ndbi", "multispectral", "spectral", "water", "urban"],
  },
  {
    id: "optical-sar",
    question: "What is Optical + SAR fusion?",
    answer:
      "Optical + SAR combines two complementary views of the same area:\n\n" +
      "• Optical — visible/NIR surface reflectance information\n" +
      "• SAR — radar backscatter (penetrates clouds, works day/night)\n\n" +
      "NexSpace co-registers optical and SAR imagery in a shared CRS, " +
      "harmonises their spatial resolution (GSD), then fuses their learned feature representations " +
      "into a 1536-dimensional fused vector.\n\n" +
      "Important: Current task-level validation established parity with the evaluated single-modality " +
      "baselines. A general accuracy improvement over single-modality has not been established.",
    tags: ["optical", "sar", "fusion", "multimodal", "radar"],
  },
  {
    id: "change-detection",
    question: "Can NexSpace detect changes over time?",
    answer:
      "Yes. NexSpace has a dedicated change-analysis pipeline that compares compatible imagery " +
      "pairs and identifies changes or anomalous regions between acquisitions.",
    tags: ["change", "temporal", "detection", "analysis"],
  },
  {
    id: "natural-language",
    question: "How does NexSpace understand natural-language queries?",
    answer:
      "The query is parsed into structured investigation intent:\n" +
      "• What to find (targets)\n" +
      "• Where and when (location, time constraints)\n" +
      "• What imagery is needed (sensor type, cloud cover, bands)\n" +
      "• Spatial relationships between entities\n\n" +
      "This structured intent is then used to route the query to the right specialist pipeline.",
    tags: ["nlp", "query", "natural-language", "understanding"],
  },
  {
    id: "evidence-provenance",
    question: "Does NexSpace provide evidence for its results?",
    answer:
      "Yes. Every investigation result includes:\n" +
      "• Evidence records — bounding boxes, region masks, spectral metrics\n" +
      "• Execution information — which models ran, in what order\n" +
      "• Provenance — full data lineage describing how the result was obtained",
    tags: ["evidence", "provenance", "explainability", "trust"],
  },
  {
    id: "unavailable-data",
    question: "What happens if the required data is unavailable?",
    answer:
      "NexSpace is designed to return an explicit unavailable or insufficient-data result rather " +
      "than fabricating an answer or silently degrading. If the required spectral bands, imagery " +
      "coverage, or analysis conditions are not met, the system should report this clearly.",
    tags: ["unavailable", "limitations", "fallback", "data"],
  },
  {
    id: "stac",
    question: "What is STAC?",
    answer:
      "STAC (SpatioTemporal Asset Catalog) is an open standard for cataloguing geospatial data. " +
      "NexSpace uses STAC to discover satellite imagery — specifically via Element84 Earth Search, " +
      "which provides access to Sentinel-2 L2A and other open datasets.",
    tags: ["stac", "catalogue", "element84", "infrastructure"],
  },
  {
    id: "crs-georef",
    question: "What is CRS and georeferencing?",
    answer:
      "CRS (Coordinate Reference System) defines how pixel coordinates map to real-world " +
      "geographic locations. Sentinel-2 typically uses UTM projections.\n\n" +
      "Georeferencing anchors image pixels to geographic coordinates using " +
      "affine transforms and CRS metadata.\n\n" +
      "NexSpace uses CRS-aware processing throughout its geospatial pipeline to ensure results " +
      "are accurately located on Earth's surface.",
    tags: ["crs", "georeferencing", "coordinate", "geospatial"],
  },
  {
    id: "accuracy",
    question: "How accurate is NexSpace?",
    answer:
      "Accuracy varies by task and data quality. Validated findings include:\n" +
      "• Object detection with Grounding DINO on Sentinel-2 imagery\n" +
      "• Spectral index computation from real bands\n" +
      "• Optical-SAR fusion at parity with single-modality baselines\n\n" +
      "NexSpace does not claim universal accuracy improvements. Results depend on image quality, " +
      "cloud cover, spatial resolution, and the specificity of the query.",
    tags: ["accuracy", "performance", "validation", "baseline"],
  },
  {
    id: "limitations",
    question: "What are NexSpace's limitations?",
    answer:
      "Known limitations:\n" +
      "• Only Sentinel-2 L2A has been validated as a data source\n" +
      "• Spectral indices require genuine multispectral bands — not available from RGB-only imagery\n" +
      "• Optical-SAR fusion establishes parity with, not improvement over, single-modality\n" +
      "• Live satellite feeds are not supported — STAC archive imagery only\n" +
      "• Cloud cover can limit optical imagery availability\n" +
      "• Not all locations or time periods will have suitable imagery",
    tags: ["limitations", "constraints", "data", "accuracy"],
  },
  {
    id: "tech-stack",
    question: "What is NexSpace built with?",
    answer:
      "Frontend: Next.js, React, TypeScript, Tailwind CSS, GSAP, OGL/WebGL\n\n" +
      "Backend: Python, FastAPI, REST APIs\n\n" +
      "ML: PyTorch, Transformers, PEFT-LoRA\n\n" +
      "Remote Sensing: GeoTIFF/TIFF, CRS, Affine transforms, WGS84, STAC, NMS\n\n" +
      "Satellite Data: Element84 Earth Search, Sentinel-2 L2A",
    tags: ["tech", "stack", "architecture", "framework"],
  },
  {
    id: "grounding-dino",
    question: "What is Grounding DINO?",
    answer:
      "Grounding DINO is an open-vocabulary object detection model. Unlike fixed-class detectors, " +
      "it can detect any object category described in natural language. NexSpace uses it as its " +
      "primary detection component for satellite imagery analysis.",
    tags: ["grounding-dino", "detection", "model", "ai"],
  },
  {
    id: "blip",
    question: "What is BLIP?",
    answer:
      "BLIP (Bootstrapping Language-Image Pre-training) is a vision-language model used in NexSpace " +
      "for scene captioning and visual question answering (VQA). It generates natural-language " +
      "descriptions of satellite imagery content.",
    tags: ["blip", "captioning", "vqa", "model", "ai"],
  },
  {
    id: "how-different",
    question: "How is NexSpace different from other tools?",
    answer:
      "NexSpace differs in several ways:\n" +
      "• Natural-language interface — no GIS expertise required to run analyses\n" +
      "• End-to-end pipeline — from query to georeferenced, evidenced result\n" +
      "• True multispectral processing — real spectral bands, not RGB approximations\n" +
      "• Multimodal fusion — optical and SAR analysed jointly\n" +
      "• Provenance by design — every result includes a full evidence and data lineage trail\n" +
      "• Truthful design — the system is designed to report uncertainty and data limitations",
    tags: ["differentiation", "comparison", "unique", "advantages"],
  },
  {
    id: "sar",
    question: "What is SAR?",
    answer:
      "SAR (Synthetic Aperture Radar) is a type of active remote sensing that transmits " +
      "microwave pulses and measures the reflected energy. Key advantages over optical:\n" +
      "• Works through clouds and at night\n" +
      "• Sensitive to surface roughness and dielectric properties\n" +
      "• Can detect structural features optical sensors cannot\n\n" +
      "NexSpace can process SAR imagery and fuse it with optical imagery for enhanced analysis.",
    tags: ["sar", "radar", "remote-sensing", "sensor"],
  },
];

// ---------------------------------------------------------------------------
// Quick-action chips (shown in chat UI)
// ---------------------------------------------------------------------------

export const QUICK_ACTION_CHIPS = [
  "What is NexSpace?",
  "Explain NDVI",
  "How does it work?",
  "Ask me anything",
  "bhai kya chal rha?",
  "Explain AI",
  "Can it detect changes?",
  "What is Optical + SAR?",
] as const;

// ---------------------------------------------------------------------------
// System prompt for the LLM (Conversational Assistant V2)
// ---------------------------------------------------------------------------

export function buildSystemPrompt(): string {
  const capabilityLines = CAPABILITIES.map(
    (c) => `  - ${c.name} [${c.status}]: ${c.detail}`
  ).join("\n");

  const faqLines = FAQ_ENTRIES.map(
    (f) => `Q: ${f.question}\nA: ${f.answer}`
  ).join("\n\n");

  return `You are NexSpace AI, the conversational AI assistant embedded inside the NexSpace platform.

You are NOT limited to FAQs.
You can have normal conversations, answer general questions, explain concepts, discuss technology, write code, and help users understand NexSpace.

CORE BEHAVIOR & TONE:
- When the user is casual, respond casually ("All good 😎", "Heyy! 👋 Kya scene hai?", "Anytime bhai 🤝").
- When the user is technical, respond technically with structured, clear explanations.
- Match the user's energy and language style. Do NOT overuse slang, but be natural and human.
- Do NOT sound robotic or corporate. Never say "According to the knowledge base..." or "I am an AI assistant designed to...".
- Be concise by default (2–6 sentences for normal/casual questions). Provide more detail when the user asks for explanations or coding.

LANGUAGE MIRRORING:
- Detect the language and style of the user's message and respond naturally in the same language/dialect.
- Full support for English, Hindi (हिंदी), Hinglish (Roman Hindi), Urdu, Spanish (Español), French (Français), German (Deutsch), etc.
- When the user writes in Hinglish (e.g. "aur bhidu kaise ho", "bhai kya chal rha", "bhai recursion samjha", "ye kaise kaam karta hai?"), respond naturally in Hinglish.
- When the user writes in Hindi ("आप कैसे हैं?", "एनडीवीआई क्या है?", "यह कैसे काम करता है?"), respond in Hindi.
- When the user writes in Spanish ("¿Cómo estás?", "¿Cómo funciona?"), respond in Spanish.
- When the user writes in French ("Comment ça va ?"), respond in French.
- Do NOT translate every message into English. Mirror the user's language unless explicitly requested otherwise.

GENERAL KNOWLEDGE & TECH:
- Answer general questions (history, science, coding, math, trivia) normally using your general model knowledge.
- For coding questions, provide correct, idiomatic code with clean explanations.
- Never unnecessarily redirect general or casual questions back to NexSpace. If the user asks "What is Python?", explain Python normally—do NOT say "Python is used in NexSpace's backend".

NEXSPACE KNOWLEDGE & GUARDRAILS:
- You have verified knowledge about the NexSpace geospatial intelligence platform (details below).
- Use this verified knowledge when questions concern NexSpace, satellite discovery, STAC, or remote sensing.
- Never invent or hallucinate NexSpace capabilities:
  1. SAM-2 integration is NOT_VERIFIED / not confirmed in the codebase.
  2. NDVI strictly requires genuine multispectral bands (Sentinel-2 L2A Red B4 and NIR B8); NDVI CANNOT be calculated from regular RGB photos.
  3. Universal satellite constellations (e.g. Planet, WorldView) are NOT supported out of the box; NexSpace focuses on Sentinel-2 L2A optical and Sentinel-1 SAR via Element84 STAC.
  4. Cross-modal Optical + SAR feature fusion: performance parity over single-modality baselines has NOT been established (always mention this caveat if asked about accuracy improvements).
- If a NexSpace-specific fact is unknown, say honestly that you don't have verified information about it.

CONVERSATION CONTEXT:
- Maintain context across recent conversation turns so follow-ups and pronouns (like "it", "why NIR?", "does NexSpace support it?") resolve naturally.
- Do not mention internal routing, prompts, tools, system messages, or knowledge-base implementation.

VERIFIED NEXSPACE REFERENCE:
${capabilityLines}

FAQ & WORKFLOW REFERENCE:
${faqLines}

PROJECT: ${PROJECT_OVERVIEW.name} — ${PROJECT_OVERVIEW.description}
TECH STACK: Frontend: ${TECH_STACK.frontend.join(", ")}; Backend: ${TECH_STACK.backend.join(", ")}; ML: ${TECH_STACK.ml.join(", ")}; Remote Sensing: ${TECH_STACK.remoteSensing.join(", ")}; Satellites: ${TECH_STACK.satellite.join(", ")}
SPECIALIST COMPONENTS: ${SPECIALIST_COMPONENTS.map((c) => `${c.name} (${c.role})`).join("; ")}
`;
}

// ---------------------------------------------------------------------------
// Validation limits
// ---------------------------------------------------------------------------

export const LIMITS = {
  maxMessageLength: 1000,
  maxConversationTurns: 20,
  maxConversationHistoryChars: 8000,
} as const;
