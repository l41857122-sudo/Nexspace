/**
 * NexSpace Conversational AI Assistant V2 — Test Suite
 *
 * Covers Section 23 Test Matrix:
 *  1. GENERAL: "what is AI?", "what is Python?", "capital of France"
 *  2. CASUAL: "hello", "how are you?", "aur bhidu kaise ho?", "thanks"
 *  3. HINGLISH: "bhai recursion samjha", "NDVI kya hota hai?"
 *  4. HINDI: "आप कैसे हैं?", "एनडीवीआई क्या है?"
 *  5. MULTILINGUAL: "¿Cómo estás?", "Comment ça va?"
 *  6. TECHNICAL: "what is DNS?", "HTTP vs HTTPS", "write binary search in C++"
 *  7. NEXSPACE: "What is NexSpace?", "Can NexSpace calculate NDVI?", "How does Optical + SAR work?"
 *  8. FOLLOW-UP: "What is NDVI?" -> "Why NIR?" -> "Does NexSpace support it?"
 *  9. HALLUCINATION PREVENTION: SAM-2 not verified, NDVI from RGB not allowed, universal satellites not claimed, Optical-SAR parity caveat
 *  10. INPUT VALIDATION: Empty message (400), Oversized message (400)
 *  11. CONVERSATION HISTORY LIMIT: Graceful trimming
 *  12. SECURITY & REGRESSION: Server-side API key protection, UI element IDs audit, /api/query regression check
 *  13. KNOWLEDGE BASE INTEGRITY: Capabilities, FAQ entries, system prompt
 */

import { QUICK_ACTION_CHIPS, LIMITS, buildSystemPrompt, FAQ_ENTRIES, CAPABILITIES } from "./lib/nexspace-ai/knowledge";

// ---------------------------------------------------------------------------
// Test harness
// ---------------------------------------------------------------------------

interface TestResult {
  name: string;
  passed: boolean;
  message: string;
}

const results: TestResult[] = [];
let total = 0;
let passed = 0;
let failed = 0;

const testQueue: Array<() => Promise<void>> = [];

function test(name: string, fn: () => void | Promise<void>): void {
  total++;
  const run = async () => {
    try {
      await fn();
      results.push({ name, passed: true, message: "PASS" });
      passed++;
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      results.push({ name, passed: false, message: `FAIL: ${msg}` });
      failed++;
    }
  };
  testQueue.push(run);
}

function assert(condition: boolean, message: string): void {
  if (!condition) throw new Error(message);
}

function assertIncludes(haystack: string, needle: string): void {
  if (!haystack.toLowerCase().includes(needle.toLowerCase())) {
    throw new Error(`Expected response to include "${needle}", got: "${haystack.slice(0, 120)}..."`);
  }
}

function assertNotIncludes(haystack: string, needle: string): void {
  if (haystack.toLowerCase().includes(needle.toLowerCase())) {
    throw new Error(`Response should NOT include "${needle}"`);
  }
}

// ---------------------------------------------------------------------------
// Helper: POST to assistant API
// ---------------------------------------------------------------------------

async function callAssistant(
  message: string,
  conversation: Array<{ role: "user" | "assistant"; content: string }> = []
): Promise<{ status: number; body: Record<string, unknown> }> {
  const BASE_URL = process.env.NEXT_PUBLIC_BASE_URL || "http://localhost:3000";
  const res = await fetch(`${BASE_URL}/api/assistant/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, conversation }),
  });
  const body = await res.json();
  return { status: res.status, body };
}

// ---------------------------------------------------------------------------
// 1. GENERAL KNOWLEDGE TESTS
// ---------------------------------------------------------------------------

test("1a. GENERAL — What is AI?", async () => {
  const { status, body } = await callAssistant("what is AI?");
  assert(status === 200, `Expected 200, got ${status}`);
  const msg = body.message as string;
  const mentionsAI =
    msg.toLowerCase().includes("intelligence") ||
    msg.toLowerCase().includes("machine") ||
    msg.toLowerCase().includes("system") ||
    msg.toLowerCase().includes("learning");
  assert(mentionsAI, "Should explain Artificial Intelligence clearly");
});

test("1b. GENERAL — What is Python?", async () => {
  const { status, body } = await callAssistant("what is Python?");
  assert(status === 200, `Expected 200, got ${status}`);
  const msg = body.message as string;
  assertIncludes(msg, "python");
  const mentionsLang =
    msg.toLowerCase().includes("programming") ||
    msg.toLowerCase().includes("language") ||
    msg.toLowerCase().includes("readable");
  assert(mentionsLang, "Should explain Python as a general programming language");
  // Must NOT force a NexSpace advertisement on a generic question
  assertNotIncludes(msg, "python is used by nexspace's backend");
});

test("1c. GENERAL — Capital of France", async () => {
  const { status, body } = await callAssistant("capital of France");
  assert(status === 200, `Expected 200, got ${status}`);
  const msg = body.message as string;
  assertIncludes(msg, "Paris");
});

// ---------------------------------------------------------------------------
// 2. CASUAL CHAT TESTS
// ---------------------------------------------------------------------------

test("2a. CASUAL — Hello", async () => {
  const { status, body } = await callAssistant("hello");
  assert(status === 200, `Expected 200, got ${status}`);
  const msg = body.message as string;
  const isFriendly =
    msg.toLowerCase().includes("hey") ||
    msg.toLowerCase().includes("hello") ||
    msg.toLowerCase().includes("kya scene");
  assert(isFriendly, "Should respond with friendly greeting");
});

test("2b. CASUAL — How are you?", async () => {
  const { status, body } = await callAssistant("how are you?");
  assert(status === 200, `Expected 200, got ${status}`);
  const msg = body.message as string;
  const isCasual =
    msg.toLowerCase().includes("good") ||
    msg.toLowerCase().includes("great") ||
    msg.toLowerCase().includes("badiya") ||
    msg.toLowerCase().includes("working on");
  assert(isCasual, "Should respond casually to 'how are you?'");
});

test("2c. CASUAL — aur bhidu kaise ho?", async () => {
  const { status, body } = await callAssistant("aur bhidu kaise ho?");
  assert(status === 200, `Expected 200, got ${status}`);
  const msg = body.message as string;
  const matchesBambaiya =
    msg.toLowerCase().includes("badiya") ||
    msg.toLowerCase().includes("bhidu") ||
    msg.toLowerCase().includes("kya scene");
  assert(matchesBambaiya, "Should mirror playful Mumbai slang style");
});

test("2d. CASUAL — thanks", async () => {
  const { status, body } = await callAssistant("thanks");
  assert(status === 200, `Expected 200, got ${status}`);
  const msg = body.message as string;
  const isPolite =
    msg.toLowerCase().includes("anytime") ||
    msg.toLowerCase().includes("welcome") ||
    msg.toLowerCase().includes("bhai");
  assert(isPolite, "Should respond warmly to thanks");
});

// ---------------------------------------------------------------------------
// 3. HINGLISH TESTS
// ---------------------------------------------------------------------------

test("3a. HINGLISH — bhai recursion samjha", async () => {
  const { status, body } = await callAssistant("bhai recursion samjha");
  assert(status === 200, `Expected 200, got ${status}`);
  const msg = body.message as string;
  assertIncludes(msg, "recursion");
  const isHinglish =
    msg.toLowerCase().includes("bhai") ||
    msg.toLowerCase().includes("function") ||
    msg.toLowerCase().includes("base condition") ||
    msg.toLowerCase().includes("call");
  assert(isHinglish, "Should explain recursion naturally in Hinglish");
});

test("3b. HINGLISH — NDVI kya hota hai?", async () => {
  const { status, body } = await callAssistant("NDVI kya hota hai?");
  assert(status === 200, `Expected 200, got ${status}`);
  const msg = body.message as string;
  assertIncludes(msg, "ndvi");
  const explainsVegetation =
    msg.toLowerCase().includes("vegetation") ||
    msg.toLowerCase().includes("khet") ||
    msg.toLowerCase().includes("plants") ||
    msg.toLowerCase().includes("greenery") ||
    msg.toLowerCase().includes("ped-paudhe");
  assert(explainsVegetation, "Should explain NDVI in accessible Hinglish");
});

// ---------------------------------------------------------------------------
// 4. HINDI (DEVANAGARI) TESTS
// ---------------------------------------------------------------------------

test("4a. HINDI — आप कैसे हैं?", async () => {
  const { status, body } = await callAssistant("आप कैसे हैं?");
  assert(status === 200, `Expected 200, got ${status}`);
  const msg = body.message as string;
  const isHindi =
    msg.includes("नमस्ते") ||
    msg.includes("बढ़िया") ||
    msg.includes("हाँ") ||
    msg.includes("ठीक");
  assert(isHindi, "Should respond in natural Hindi");
});

test("4b. HINDI — एनडीवीआई क्या है?", async () => {
  const { status, body } = await callAssistant("एनडीवीआई क्या है?");
  assert(status === 200, `Expected 200, got ${status}`);
  const msg = body.message as string;
  const explainsInHindi =
    msg.includes("वनस्पति") ||
    msg.includes("उपग्रह") ||
    msg.includes("पौधों") ||
    msg.includes("NDVI");
  assert(explainsInHindi, "Should explain NDVI in Hindi");
});

// ---------------------------------------------------------------------------
// 5. MULTILINGUAL TESTS
// ---------------------------------------------------------------------------

test("5a. MULTILINGUAL — ¿Cómo estás?", async () => {
  const { status, body } = await callAssistant("¿Cómo estás?");
  assert(status === 200, `Expected 200, got ${status}`);
  const msg = body.message as string;
  const isSpanish =
    msg.toLowerCase().includes("hola") ||
    msg.toLowerCase().includes("bien") ||
    msg.toLowerCase().includes("ayudar");
  assert(isSpanish, "Should respond in Spanish");
});

test("5b. MULTILINGUAL — Comment ça va?", async () => {
  const { status, body } = await callAssistant("Comment ça va?");
  assert(status === 200, `Expected 200, got ${status}`);
  const msg = body.message as string;
  const isFrench =
    msg.toLowerCase().includes("bonjour") ||
    msg.toLowerCase().includes("bien") ||
    msg.toLowerCase().includes("merci");
  assert(isFrench, "Should respond in French");
});

// ---------------------------------------------------------------------------
// 6. TECHNICAL & CODING TESTS
// ---------------------------------------------------------------------------

test("6a. TECHNICAL — what is DNS?", async () => {
  const { status, body } = await callAssistant("what is DNS?");
  assert(status === 200, `Expected 200, got ${status}`);
  const msg = body.message as string;
  assertIncludes(msg, "dns");
  const mentionsMapping =
    msg.toLowerCase().includes("domain") ||
    msg.toLowerCase().includes("ip") ||
    msg.toLowerCase().includes("phonebook");
  assert(mentionsMapping, "Should explain DNS correctly");
});

test("6b. TECHNICAL — HTTP vs HTTPS", async () => {
  const { status, body } = await callAssistant("HTTP vs HTTPS");
  assert(status === 200, `Expected 200, got ${status}`);
  const msg = body.message as string;
  const mentionsSecurity =
    msg.toLowerCase().includes("encrypt") ||
    msg.toLowerCase().includes("tls") ||
    msg.toLowerCase().includes("ssl") ||
    msg.toLowerCase().includes("secure");
  assert(mentionsSecurity, "Should explain HTTP vs HTTPS security differences");
});

test("6c. TECHNICAL — write binary search in C++", async () => {
  const { status, body } = await callAssistant("write binary search in C++");
  assert(status === 200, `Expected 200, got ${status}`);
  const msg = body.message as string;
  const containsCode =
    msg.includes("binarySearch") ||
    msg.includes("mid") ||
    msg.includes("vector") ||
    msg.includes("int left");
  assert(containsCode, "Should provide valid C++ binary search code");
});

// ---------------------------------------------------------------------------
// 7. NEXSPACE SPECIFIC TESTS
// ---------------------------------------------------------------------------

test("7a. NEXSPACE — What is NexSpace?", async () => {
  const { status, body } = await callAssistant("What is NexSpace?");
  assert(status === 200, `Expected 200, got ${status}`);
  assert(body.status === "success", `Expected success status, got ${body.status}`);
  const msg = body.message as string;
  assertIncludes(msg, "nexspace");
  assertIncludes(msg, "satellite");
});

test("7b. NEXSPACE — Can NexSpace calculate NDVI?", async () => {
  const { status, body } = await callAssistant("Can NexSpace calculate NDVI?");
  assert(status === 200, `Expected 200, got ${status}`);
  const msg = body.message as string;
  assertIncludes(msg, "ndvi");
  const mentionsBands =
    msg.toLowerCase().includes("band") ||
    msg.toLowerCase().includes("sentinel") ||
    msg.toLowerCase().includes("multispectral");
  assert(mentionsBands, "Should mention Sentinel bands for NDVI calculation");
});

test("7c. NEXSPACE — How does Optical + SAR work?", async () => {
  const { status, body } = await callAssistant("How does Optical + SAR work?");
  assert(status === 200, `Expected 200, got ${status}`);
  const msg = body.message as string;
  assertIncludes(msg, "optical");
  assertIncludes(msg, "sar");
});

// ---------------------------------------------------------------------------
// 8. FOLLOW-UP & CONTEXT MEMORY
// ---------------------------------------------------------------------------

test("8. FOLLOW-UP — NDVI -> Why NIR? -> Does NexSpace support it?", async () => {
  // Step 1: Initial question
  const r1 = await callAssistant("What is NDVI?");
  assert(r1.status === 200, "Step 1 failed");
  const msg1 = r1.body.message as string;
  assertIncludes(msg1, "ndvi");

  // Step 2: Contextual follow-up "Why NIR?"
  const history1 = [
    { role: "user" as const, content: "What is NDVI?" },
    { role: "assistant" as const, content: msg1 },
  ];
  const r2 = await callAssistant("Why NIR?", history1);
  assert(r2.status === 200, "Step 2 failed");
  const msg2 = r2.body.message as string;
  const resolvesNIR =
    msg2.toLowerCase().includes("nir") ||
    msg2.toLowerCase().includes("near-infrared") ||
    msg2.toLowerCase().includes("chlorophyll") ||
    msg2.toLowerCase().includes("reflect");
  assert(resolvesNIR, "Step 2 should explain why Near-Infrared is necessary for NDVI");

  // Step 3: Pronoun resolution "Does NexSpace support it?"
  const history2 = [
    ...history1,
    { role: "user" as const, content: "Why NIR?" },
    { role: "assistant" as const, content: msg2 },
  ];
  const r3 = await callAssistant("Does NexSpace support it?", history2);
  assert(r3.status === 200, "Step 3 failed");
  const msg3 = r3.body.message as string;
  const confirmsSupport =
    msg3.toLowerCase().includes("yes") ||
    msg3.toLowerCase().includes("ndvi") ||
    msg3.toLowerCase().includes("sentinel");
  assert(confirmsSupport, "Step 3 should resolve 'it' to NDVI and confirm support");
});

// ---------------------------------------------------------------------------
// 9. HALLUCINATION PREVENTION (UNKNOWN NEXSPACE FACTS)
// ---------------------------------------------------------------------------

test("9a. HALLUCINATION — SAM-2 not verified", async () => {
  const { status, body } = await callAssistant("Does NexSpace use SAM-2?");
  assert(status === 200, `Expected 200, got ${status}`);
  const msg = body.message as string;
  const claimsYes =
    msg.toLowerCase().includes("yes, nexspace uses sam-2") ||
    msg.toLowerCase().includes("nexspace uses sam 2");
  assert(!claimsYes, "Should NOT claim SAM-2 is verified in NexSpace");
  const expressesUncertainty =
    msg.toLowerCase().includes("not confirmed") ||
    msg.toLowerCase().includes("not verified") ||
    msg.toLowerCase().includes("uncertain");
  assert(expressesUncertainty, "Should clarify that SAM-2 is not confirmed/verified");
});

test("9b. HALLUCINATION — NDVI from RGB photo not allowed", async () => {
  const { status, body } = await callAssistant("Can NexSpace compute NDVI from a regular RGB photo?");
  assert(status === 200, `Expected 200, got ${status}`);
  const msg = body.message as string;
  const claimsRgbWorks =
    msg.toLowerCase().includes("yes, nexspace can compute ndvi from rgb") ||
    msg.toLowerCase().includes("rgb photos work for ndvi");
  assert(!claimsRgbWorks, "Should NOT claim regular RGB photos can produce real NDVI");
  const clarifiesBands =
    msg.toLowerCase().includes("genuine") ||
    msg.toLowerCase().includes("real") ||
    msg.toLowerCase().includes("band") ||
    msg.toLowerCase().includes("cannot");
  assert(clarifiesBands, "Should clarify real multispectral bands are required");
});

test("9c. HALLUCINATION — Universal satellite constellations not claimed", async () => {
  const { status, body } = await callAssistant("Can NexSpace analyze any satellite imagery, like from Planet or WorldView?");
  assert(status === 200, `Expected 200, got ${status}`);
  const msg = body.message as string;
  const claimsUniversal =
    msg.toLowerCase().includes("yes, any satellite") ||
    msg.toLowerCase().includes("supports all satellites") ||
    msg.toLowerCase().includes("supports planet");
  assert(!claimsUniversal, "Should NOT claim universal satellite support");
});

test("9d. HALLUCINATION — Optical-SAR fusion parity caveat", async () => {
  const { status, body } = await callAssistant("Does Optical-SAR fusion improve accuracy?");
  assert(status === 200, `Expected 200, got ${status}`);
  const msg = body.message as string;
  const mentionsCaveat =
    msg.toLowerCase().includes("parity") ||
    msg.toLowerCase().includes("not been established") ||
    msg.toLowerCase().includes("caveat") ||
    msg.toLowerCase().includes("experimental");
  assert(mentionsCaveat, "Should include accuracy parity caveat for Optical-SAR fusion");
});

// ---------------------------------------------------------------------------
// 10. INPUT VALIDATION TESTS
// ---------------------------------------------------------------------------

test("10a. VALIDATION — Empty message returns 400", async () => {
  const { status, body } = await callAssistant("   ");
  assert(status === 400, `Expected 400 for empty message, got ${status}`);
  assert(body.status === "error", "Expected error status for empty message");
});

test("10b. VALIDATION — Oversized message returns 400", async () => {
  const oversized = "a".repeat(LIMITS.maxMessageLength + 1);
  const { status, body } = await callAssistant(oversized);
  assert(status === 400, `Expected 400 for oversized message, got ${status}`);
  assert(body.status === "error", "Expected error status for oversized message");
});

// ---------------------------------------------------------------------------
// 11. CONVERSATION HISTORY LIMIT TEST
// ---------------------------------------------------------------------------

test("11. HISTORY — 50-turn conversation handled gracefully without crashing", async () => {
  const largeHistory: Array<{ role: "user" | "assistant"; content: string }> = [];
  for (let i = 0; i < 50; i++) {
    largeHistory.push({ role: "user", content: `Question ${i}: What is NexSpace?` });
    largeHistory.push({ role: "assistant", content: `Answer ${i}: NexSpace is an AI platform.` });
  }

  const { status, body } = await callAssistant("What is Sentinel-2?", largeHistory);
  assert(status === 200, `Expected 200, got ${status}`);
  assert(body.status === "success", "Expected success status for long history");
});

// ---------------------------------------------------------------------------
// 12. SECURITY & REGRESSION AUDIT TESTS
// ---------------------------------------------------------------------------

test("12a. SECURITY — Server-side API key protection audit", async () => {
  const fs = await import("fs");
  const routeContent = fs.readFileSync("./app/api/assistant/chat/route.ts", "utf-8");
  assert(!routeContent.includes("NEXT_PUBLIC_OPENAI_API_KEY"), "Never use NEXT_PUBLIC_OPENAI_API_KEY");
  assert(!routeContent.includes("NEXT_PUBLIC_ANTHROPIC_API_KEY"), "Never use NEXT_PUBLIC_ANTHROPIC_API_KEY");
});

test("12b. UI AUDIT — Required element IDs present in component", async () => {
  const fs = await import("fs");
  const compContent = fs.readFileSync("./app/components/NexSpaceAssistant.tsx", "utf-8");
  assert(compContent.includes('id="nexspace-assistant-trigger"'), "Missing trigger ID");
  assert(compContent.includes('id="nexspace-assistant-panel"'), "Missing panel ID");
  assert(compContent.includes('id="nexspace-assistant-close-btn"'), "Missing close button ID");
  assert(compContent.includes('id="nexspace-assistant-input"'), "Missing input ID");
  assert(compContent.includes('id="nexspace-assistant-send-btn"'), "Missing send button ID");
});

test("12c. REGRESSION — /api/query endpoint untouched", async () => {
  const fs = await import("fs");
  const queryRoute = fs.readFileSync("./app/api/query/route.ts", "utf-8");
  assert(queryRoute.includes("runInvestigation") || queryRoute.includes("investigation"), "Investigation pipeline intact");
  assert(!queryRoute.includes("NexSpace AI Assistant"), "Assistant logic must NOT bleed into investigation endpoint");
});

// ---------------------------------------------------------------------------
// 13. KNOWLEDGE BASE INTEGRITY TESTS
// ---------------------------------------------------------------------------

test("13a. KB — Capabilities integrity", async () => {
  assert(CAPABILITIES.length >= 10, "Should have verified capabilities");
  for (const cap of CAPABILITIES) {
    assert(Boolean(cap.name && cap.status && cap.detail), "Capability missing required fields");
  }
});

test("13b. KB — Quick-action chips integrity", async () => {
  assert(QUICK_ACTION_CHIPS.length >= 4, "Should have at least 4 quick-action chips");
  for (const chip of QUICK_ACTION_CHIPS) {
    assert(chip.trim().length > 0, "Chip cannot be empty");
  }
});

test("13c. KB — V2 System prompt comprehensiveness", async () => {
  const prompt = buildSystemPrompt();
  assertIncludes(prompt, "NexSpace AI");
  assertIncludes(prompt, "conversational AI assistant");
  assertIncludes(prompt, "LANGUAGE MIRRORING");
  assertIncludes(prompt, "Hinglish");
});

test("13d. KB — FAQ entries integrity", async () => {
  assert(FAQ_ENTRIES.length >= 10, "Should have verified FAQ entries");
  for (const faq of FAQ_ENTRIES) {
    assert(Boolean(faq.question && faq.answer), "FAQ entry missing required fields");
  }
});

// ---------------------------------------------------------------------------
// Execution runner
// ---------------------------------------------------------------------------

async function runAll(): Promise<void> {
  console.log("\n╔══════════════════════════════════════════════════════════════╗");
  console.log("║    NexSpace Conversational AI Assistant V2 — Test Suite     ║");
  console.log("╚══════════════════════════════════════════════════════════════╝\n");

  for (const fn of testQueue) {
    await fn();
  }

  console.log("\n──────────────────────────────────────────────────────────────");
  for (const res of results) {
    if (res.passed) {
      console.log(`✅ ${res.name}`);
    } else {
      console.log(`❌ ${res.name}`);
      console.log(`   ${res.message}`);
    }
  }

  console.log("\n──────────────────────────────────────────────────────────────");
  console.log(`Total: ${total}  Passed: ${passed}  Failed: ${failed}`);
  console.log("──────────────────────────────────────────────────────────────\n");

  if (failed > 0) {
    process.exit(1);
  } else {
    process.exit(0);
  }
}

runAll().catch((e) => {
  console.error("Fatal test runner error:", e);
  process.exit(1);
});
