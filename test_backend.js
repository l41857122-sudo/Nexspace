// Node.js test script verifying the NexSpace ML Controller Backend Router logic

function runBackendRouter(queryStr, hasOptical = true, hasSar = false, hasChange = false) {
  const query = (queryStr || "").trim();
  const targetTools = [];
  const restructuredVqaQueries = [];
  let requiresCountWarning = false;
  const reasoningParts = [];

  if (hasChange) {
    targetTools.push("Change_Analysis");
    reasoningParts.push(
      "Before/after image pair detected -> triggering Change Analysis pipeline for pixel-level heatmap and spatial diff."
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
        "Detected counting query -> routed to VQA with 'how many [object]?' syntax; flagged requires_count_warning=True due to known low accuracy (confidence 0.25-0.40) on counting tasks."
      );
    } else if (isOpenEnded) {
      if (hasOptical) targetTools.push("Optical_Caption");
      targetTools.push("VQA");
      restructuredVqaQueries.push("Is there a body of water present?");
      restructuredVqaQueries.push("Are there buildings present?");
      restructuredVqaQueries.push("Is there a road present?");
      restructuredVqaQueries.push("Is there vegetation present?");
      reasoningParts.push(
        "Open-ended query detected -> routed to Optical Captioning for free-form scene description, and decomposed into structured RSVQA-style binary sub-questions for VQA."
      );
    } else {
      targetTools.push("VQA");
      let norm = query.endsWith("?") ? query : query + "?";
      norm = norm[0].toUpperCase() + norm.slice(1);
      restructuredVqaQueries.push(norm);
      reasoningParts.push("Closed-ended query detected -> routed directly to VQA after RSVQA-style normalization.");
    }
  } else if (hasOptical && !hasChange) {
    targetTools.push("Optical_Caption");
    reasoningParts.push("No query text provided -> defaulting to Optical Captioning for general scene description.");
  }

  if (hasSar) {
    targetTools.push("SAR_Caption");
    reasoningParts.push("SAR imagery provided -> also routing to SAR Captioning for radar-domain description.");
  }

  const uniqueTools = Array.from(new Set(targetTools));
  const vqaResults = restructuredVqaQueries.map(q => ({
    question: q,
    answer: q.toLowerCase().includes("how many") ? "12 (estimated)" : "yes",
    confidence: q.toLowerCase().includes("how many") ? 0.32 : 0.85,
    low_confidence: q.toLowerCase().includes("how many")
  }));

  let opticalCaption = null;
  if (uniqueTools.includes("Optical_Caption")) {
    opticalCaption = "An aerial satellite overview showing mixed urban infrastructure, vegetation, and water bodies.";
  }

  let sarCaption = null;
  if (uniqueTools.includes("SAR_Caption")) {
    sarCaption = "[SAR radar scene] High-backscatter structural reflection showing urban grid and coastal line.";
  }

  let changeAnalysis = null;
  if (uniqueTools.includes("Change_Analysis")) {
    changeAnalysis = {
      summary: "Change analysis detected moderate change across a notable portion of the scene between Image A (before) and Image B (after): 18.4% of pixels exceeded threshold.",
      changed_fraction: 0.184,
      mean_intensity_delta: 34.2
    };
  }

  const parts = [];
  if (opticalCaption) parts.push(`**Optical scene description:** ${opticalCaption}`);
  if (sarCaption) parts.push(`**SAR scene description:** ${sarCaption}`);
  if (vqaResults.length > 0) {
    const lines = vqaResults.map(r => `- ${r.question} → ${r.answer}${r.low_confidence ? "  ⚠️ low confidence" : ""}`);
    parts.push(`**Structured VQA findings:**\n${lines.join("\n")}`);
  }
  if (changeAnalysis) parts.push(`**Change analysis:** ${changeAnalysis.summary}`);
  if (requiresCountWarning) {
    parts.push("⚠️ Note: Exact numeric counts are derived with low model confidence (~0.25-0.40). Treat this count as an estimate.");
  }

  return {
    routing_decision: {
      target_tools: uniqueTools,
      restructured_vqa_queries: restructuredVqaQueries,
      requires_count_warning: requiresCountWarning,
      execution_reasoning: reasoningParts.join(" ") || "No actionable query or imagery provided."
    },
    vqa_results: vqaResults,
    optical_caption: opticalCaption,
    sar_caption: sarCaption,
    change_analysis: changeAnalysis,
    response_text: parts.join("\n\n")
  };
}

console.log("==================================================================");
console.log("SCENARIO 1: Open-ended Query");
console.log("==================================================================");
console.log(JSON.stringify(runBackendRouter("What is visible in this image? Is there a river?"), null, 2));

console.log("\n==================================================================");
console.log("SCENARIO 2: Counting Query (Low Confidence Warning)");
console.log("==================================================================");
console.log(JSON.stringify(runBackendRouter("How many residential buildings are in this area?"), null, 2));

console.log("\n==================================================================");
console.log("SCENARIO 3: Multi-modal Optical + SAR Fusion");
console.log("==================================================================");
console.log(JSON.stringify(runBackendRouter("Describe this scene using both sensors", true, true, false), null, 2));

console.log("\n==================================================================");
console.log("SCENARIO 4: Bi-temporal Change Analysis");
console.log("==================================================================");
console.log(JSON.stringify(runBackendRouter("What changed between these dates?", false, false, true), null, 2));
