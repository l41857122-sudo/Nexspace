import Anthropic from "@anthropic-ai/sdk";

export interface QueryRequest {
  query?: string;
  text?: string;
  filters?: any;
  optical_image?: string;
  sar_image?: string;
  change_image_a?: string;
  change_image_b?: string;
}

export interface ParsedFilters {
  location: string;
  dateRange: string;
  sensor: string;
  entityType: string;
  cloudCoverMax: number;
}

const anthropic = process.env.ANTHROPIC_API_KEY ? new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY }) : null;

export async function parseQueryFilters(rawText: string): Promise<ParsedFilters> {
  const text = rawText.toLowerCase();
  
  let location = "Sector 9B / Quadrant 7A";
  if (text.includes("suez")) location = "Suez Canal";
  else if (text.includes("malacca")) location = "Malacca Strait";
  else if (text.includes("amazon")) location = "Amazon Basin";
  else if (text.includes("lagos")) location = "Lagos";

  let sensor = "Multispectral";
  if (text.includes("sar") || text.includes("radar")) sensor = "SAR";
  else if (text.includes("thermal") || text.includes("infrared")) sensor = "Thermal";
  else if (text.includes("optical")) sensor = "Optical";

  let entityType = "Vessel/Infrastructure";
  if (text.includes("vessel") || text.includes("ship") || text.includes("maritime")) entityType = "Cargo/Container";
  else if (text.includes("building") || text.includes("construction") || text.includes("urban")) entityType = "Infrastructure";
  else if (text.includes("ndvi") || text.includes("vegetation") || text.includes("forest")) entityType = "Vegetation";

  let dateRange = "Last 48 Hours";
  if (text.includes("30 days") || text.includes("30d")) dateRange = "Last 30 Days";
  else if (text.includes("7 days") || text.includes("7d")) dateRange = "Last 7 Days";

  if (anthropic && rawText) {
    try {
      const prompt = `You are a geospatial NLP assistant. Extract structured filters from this query: "${rawText}".
Return JSON object with keys: location (string), dateRange (string), sensor (string), entityType (string), cloudCoverMax (number).`;
      
      const response = await anthropic.messages.create({
        model: "claude-3-5-sonnet-20241022",
        max_tokens: 200,
        messages: [{ role: "user", content: prompt }]
      });

      const contentText = response.content[0].type === "text" ? response.content[0].text : "";
      const parsed = JSON.parse(contentText);
      return {
        location: parsed.location || location,
        dateRange: parsed.dateRange || dateRange,
        sensor: parsed.sensor || sensor,
        entityType: parsed.entityType || entityType,
        cloudCoverMax: parsed.cloudCoverMax ?? 20
      };
    } catch (e) {
      // Fallback to rule-based parser on API error or missing key
    }
  }

  return {
    location,
    dateRange,
    sensor,
    entityType,
    cloudCoverMax: 20
  };
}

export function generateExecutionStages(queryId: string) {
  return [
    {
      id: `stage-1-${queryId}`,
      queryId,
      name: "Data Ingestion & Verification",
      state: "done",
      order: 1,
      logs: [
        "14:02:11 [INFO] Initializing pipeline...",
        "14:02:12 Loading source array TRQ_64A0_RAW (34.2 GB)",
        "14:02:18 [SUCCESS] Data ingestion complete. Checksum matched."
      ],
      progressPct: 100
    },
    {
      id: `stage-2-${queryId}`,
      queryId,
      name: "Radiometric & Atmospheric Correction",
      state: "done",
      order: 2,
      logs: [
        "14:02:19 Applying radiometric calibration profile...",
        "14:02:35 [WARN] Cloud cover detected in sector 9 (coverage ~12%)",
        "14:02:40 [SUCCESS] Radiometric correction applied. Tensor shape: [1024, 1024, 6]"
      ],
      progressPct: 100
    },
    {
      id: `stage-3-${queryId}`,
      queryId,
      name: "Neural Feature Extraction (YOLOv8 + RSVQA)",
      state: "active",
      order: 3,
      logs: [
        "14:02:41 [INFO] Booting Neural Extraction Engine (GPU:0, GPU:1)...",
        "14:02:43 Allocating VRAM... 16000MB reserved",
        "14:02:45 Commencing deep feature extraction using model RESNET_SAT_v4",
        "14:03:02 Processing batch 42/64 ..."
      ],
      progressPct: 75
    },
    {
      id: `stage-4-${queryId}`,
      queryId,
      name: "Spatial Clustering & Vector Indexing",
      state: "pending",
      order: 4,
      logs: ["Awaiting stage 3 completion..."],
      progressPct: 0
    },
    {
      id: `stage-5-${queryId}`,
      queryId,
      name: "Confidence Scoring & Synthesis",
      state: "pending",
      order: 5,
      logs: ["Awaiting stage 4 completion..."],
      progressPct: 0
    }
  ];
}
