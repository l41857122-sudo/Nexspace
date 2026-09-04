import { NextResponse } from "next/server";
import type { NexSpaceQueryResponse } from "../../types/nexspace";

const ML_BACKEND_URL = process.env.ML_BACKEND_URL || "http://localhost:8000";

interface QueryBody {
  query?: string;
  optical_image?: string;
  sar_image?: string;
  change_image_a?: string;
  change_image_b?: string;
  probe_features?: string[];
}

function runOfflineFallback(queryStr: string, hasOptical: boolean, hasSar: boolean, hasChange: boolean): NexSpaceQueryResponse {
  const query = (queryStr || "").trim();
  const targetTools: string[] = [];
  const restructuredVqaQueries: string[] = [];
  let requiresCountWarning = false;
  const reasoningParts: string[] = [];

  if (hasChange) {
    targetTools.push("Change_Analysis");
    reasoningParts.push(
      "Before/after image pair detected -> triggering Change Analysis pipeline."
    );
  }

  const isCounting = /\bhow many\b/i.test(query);
  const openEndedMarkers = [
    "what is visible", "what can you see", "describe", "tell me about",
    "what's in this image", "what is in this image", "summarize",
    "give me an overview", "what does this scene show"
  ];
  const closedStarters = [
    "is there", "are there", "is it", "are they", "does", "do",
    "was there", "were there", "can you see a", "which", "what type of",
    "what color", "what is the color of"
  ];

  const qLower = query.toLowerCase();
  const isOpenEnded = openEndedMarkers.some(m => qLower.includes(m)) ||
    (!isCounting && !closedStarters.some(s => qLower.startsWith(s)) && qLower.startsWith("what"));

  if (query) {
    if (isCounting) {
      targetTools.push("VQA");
      const match = query.match(/how many ([a-zA-Z\s]+?)(\?|$)/i);
      const obj = match ? match[1].trim() : "objects";
      restructuredVqaQueries.push(`how many ${obj}?`);
      requiresCountWarning = true;
      reasoningParts.push(
        "Detected counting query -> routed to VQA; flagged requires_count_warning=True."
      );
    } else if (isOpenEnded) {
      if (hasOptical) targetTools.push("Optical_Caption");
      targetTools.push("VQA");
      restructuredVqaQueries.push("Is there a body of water present?");
      restructuredVqaQueries.push("Are there buildings present?");
      restructuredVqaQueries.push("Is there a road present?");
      restructuredVqaQueries.push("Is there vegetation present?");
      reasoningParts.push(
        "Open-ended query -> routed to Optical Captioning and decomposed into RSVQA binary sub-questions."
      );
    } else {
      targetTools.push("VQA");
      let norm = query.endsWith("?") ? query : query + "?";
      norm = norm[0].toUpperCase() + norm.slice(1);
      restructuredVqaQueries.push(norm);
      reasoningParts.push("Closed-ended query -> routed directly to VQA.");
    }
  } else if (hasOptical && !hasChange) {
    targetTools.push("Optical_Caption");
    reasoningParts.push("No query text provided -> defaulting to Optical Captioning.");
  }

  if (hasSar) {
    targetTools.push("SAR_Caption");
    reasoningParts.push("SAR imagery provided -> routing to SAR Captioning.");
  }

  const uniqueTools = Array.from(new Set(targetTools));
  const vqaResults = restructuredVqaQueries.map(q => ({
    question: q,
    answer: q.toLowerCase().includes("how many") ? "12 (estimated)" : "yes",
    confidence: q.toLowerCase().includes("how many") ? 0.32 : 0.85,
    low_confidence: q.toLowerCase().includes("how many")
  }));

  const parts: string[] = [
    "⚠️ **Notice: Python ML backend is currently offline.** Displaying client-side structural routing fallback.",
    `Selected tools: ${uniqueTools.join(", ")}`
  ];

  return {
    request_id: "req_offline_fallback",
    status: "completed",
    query: queryStr,
    intent: isCounting ? "VQA" : (isOpenEnded ? "Optical_Caption" : "VQA"),
    plan: {
      task_type: isCounting ? "VQA" : (isOpenEnded ? "Optical_Caption" : "VQA"),
      target_tools: uniqueTools,
      parameters: { query: queryStr },
      execution_strategy: "offline_fallback"
    },
    selected_tools: uniqueTools,
    routing_decision: {
      target_tools: uniqueTools,
      restructured_vqa_queries: restructuredVqaQueries,
      requires_count_warning: requiresCountWarning,
      execution_reasoning: reasoningParts.join(" ") || "Client-side fallback routing."
    },
    vqa_results: vqaResults,
    optical_caption: uniqueTools.includes("Optical_Caption") ? "[Offline Fallback: Start Python ML Server for live BLIP inference]" : null,
    sar_caption: uniqueTools.includes("SAR_Caption") ? "[Offline Fallback: Start Python ML Server for live SAR inference]" : null,
    change_analysis: null,
    evidence: [],
    evidence_graph: { query_id: "req_offline_fallback", nodes: [], edges: [] },
    investigation_report: {
      summary: "Python ML Backend is offline. Start FastAPI server (`python ml_backend/server.py`) on port 8000 for live neural inference.",
      observations: ["FastAPI server at http://localhost:8000 did not respond."],
      interpretations: ["Operating in emergency client-side fallback mode."],
      evidence_references: [],
      limitations: ["No live neural inference (BLIP, Grounding DINO, PaliGemma) available in offline mode."]
    },
    execution_trace: [
      {
        stage: "client_fallback",
        status: "completed",
        started_at: new Date().toISOString(),
        duration_ms: 1.0,
        metadata: { source: "nextjs_offline_router" }
      }
    ],
    confidence: 0.5,
    confidence_type: "heuristic",
    confidence_source: "nextjs_fallback",
    fallback_count: 1,
    limitations: ["Python ML backend offline; using client-side structural stub."],
    response_text: parts.join("\n\n"),
    backend_status: "offline_fallback"
  };
}

export async function POST(req: Request) {
  let body: QueryBody = {};
  try {
    body = await req.json();
  } catch (e) {
    // Empty body
  }

  // Forward to FastAPI backend with 60s timeout for model inference
  try {
    const controllerSignal = AbortSignal.timeout(60000);
    const res = await fetch(`${ML_BACKEND_URL}/api/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: controllerSignal
    });

    if (res.ok) {
      const data = await res.json();
      data.backend_status = "online";
      return NextResponse.json(data);
    } else {
      // Forward client/server error with proper status code
      const errorData = await res.json().catch(() => ({ detail: "ML Backend Error" }));
      return NextResponse.json(errorData, { status: res.status });
    }
  } catch (err: any) {
    // FastAPI server unreachable
    console.warn("[Next.js Proxy] FastAPI ML backend offline at:", ML_BACKEND_URL, err?.message);
  }

  const fallback = runOfflineFallback(
    body.query || "",
    !!body.optical_image,
    !!body.sar_image,
    !!(body.change_image_a && body.change_image_b)
  );

  return NextResponse.json(fallback);
}
