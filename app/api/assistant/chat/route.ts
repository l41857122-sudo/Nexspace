import { NextRequest, NextResponse } from "next/server";
import OpenAI from "openai";
import Anthropic from "@anthropic-ai/sdk";
import { buildSystemPrompt, LIMITS } from "../../../../lib/nexspace-ai/knowledge";
import { answerFromKnowledgeBase } from "../../../../lib/nexspace-ai/fallback";

// ---------------------------------------------------------------------------
// Isolated NexSpace Conversational AI Assistant endpoint
// POST /api/assistant/chat
//
// This endpoint is COMPLETELY SEPARATE from the satellite investigation
// pipeline at /api/query. It MUST NOT be modified for investigation use.
// ---------------------------------------------------------------------------


/** Conversation message shape expected from / returned to the client */
interface ConversationMessage {
  role: "user" | "assistant";
  content: string;
}

/** Validated request body */
interface ChatRequestBody {
  message: string;
  conversation?: ConversationMessage[];
}


// ---------------------------------------------------------------------------
// Rate-limit (simple in-memory, per-IP, sliding window)
// For production use a Redis-backed implementation.
// ---------------------------------------------------------------------------
const rateLimitMap = new Map<string, { count: number; windowStart: number }>();
const RATE_LIMIT_WINDOW_MS = 60_000; // 1 minute
const RATE_LIMIT_MAX_REQUESTS = 20;

function checkRateLimit(ip: string): boolean {
  // Exempt local loopback / testing environments from aggressive rate limiting
  if (
    process.env.NODE_ENV === "test" ||
    ip === "127.0.0.1" ||
    ip === "::1" ||
    ip === "localhost" ||
    ip === "unknown"
  ) {
    return true;
  }

  const now = Date.now();
  const entry = rateLimitMap.get(ip);
  if (!entry || now - entry.windowStart > RATE_LIMIT_WINDOW_MS) {
    rateLimitMap.set(ip, { count: 1, windowStart: now });
    return true;
  }
  if (entry.count >= RATE_LIMIT_MAX_REQUESTS) {
    return false;
  }
  entry.count += 1;
  return true;
}

// Prune old entries periodically to prevent unbounded memory growth
setInterval(() => {
  const now = Date.now();
  for (const [ip, entry] of rateLimitMap.entries()) {
    if (now - entry.windowStart > RATE_LIMIT_WINDOW_MS * 2) {
      rateLimitMap.delete(ip);
    }
  }
}, 5 * 60_000);

// ---------------------------------------------------------------------------
// Route handler
// ---------------------------------------------------------------------------

export async function POST(req: NextRequest): Promise<NextResponse> {
  // --- Rate limiting ---
  const ip =
    req.headers.get("x-forwarded-for")?.split(",")[0]?.trim() ||
    req.headers.get("x-real-ip") ||
    "unknown";

  if (!checkRateLimit(ip)) {
    return NextResponse.json(
      { error: "Too many requests. Please wait a moment.", status: "rate_limited" },
      { status: 429 }
    );
  }

  // --- Parse body ---
  let body: ChatRequestBody;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json(
      { error: "Invalid request body.", status: "error" },
      { status: 400 }
    );
  }

  const { message, conversation = [] } = body;

  // --- Validate message ---
  if (!message || typeof message !== "string") {
    return NextResponse.json(
      { error: "Message is required.", status: "error" },
      { status: 400 }
    );
  }

  const trimmed = message.trim();
  if (trimmed.length === 0) {
    return NextResponse.json(
      { error: "Message cannot be empty.", status: "error" },
      { status: 400 }
    );
  }

  if (trimmed.length > LIMITS.maxMessageLength) {
    return NextResponse.json(
      {
        error: `Message too long. Maximum ${LIMITS.maxMessageLength} characters.`,
        status: "error",
      },
      { status: 400 }
    );
  }

  // --- Validate conversation history ---
  if (!Array.isArray(conversation)) {
    return NextResponse.json(
      { error: "Conversation must be an array.", status: "error" },
      { status: 400 }
    );
  }

  // Enforce turn limit and strip to allowed shape
  const sanitizedHistory: ConversationMessage[] = conversation
    .filter(
      (m): m is ConversationMessage =>
        m &&
        typeof m === "object" &&
        (m.role === "user" || m.role === "assistant") &&
        typeof m.content === "string"
    )
    .slice(-LIMITS.maxConversationTurns * 2) // keep last N turn pairs
    .map((m) => ({
      role: m.role,
      content: m.content.slice(0, LIMITS.maxMessageLength),
    }));

  // Enforce total history character budget
  let totalHistoryChars = sanitizedHistory.reduce(
    (sum, m) => sum + m.content.length,
    0
  );
  while (
    totalHistoryChars > LIMITS.maxConversationHistoryChars &&
    sanitizedHistory.length > 0
  ) {
    const removed = sanitizedHistory.shift();
    totalHistoryChars -= removed?.content.length ?? 0;
  }

  // --- Call OpenAI or Anthropic (or fallback to Conversational Engine if offline / key unset) ---
  const openaiKey = process.env.OPENAI_API_KEY;
  const anthropicKey = process.env.ANTHROPIC_API_KEY;

  if (!openaiKey && !anthropicKey) {
    const fallbackAnswer = answerFromKnowledgeBase(trimmed, sanitizedHistory);
    return NextResponse.json({
      message: fallbackAnswer,
      source: "NEXSPACE_CONVERSATIONAL_FALLBACK",
      status: "success",
    });
  }

  const systemPrompt = buildSystemPrompt();

  // Try OpenAI first if configured
  if (openaiKey) {
    try {
      const openai = new OpenAI({ apiKey: openaiKey });
      const model = process.env.NEXSPACE_ASSISTANT_MODEL || "gpt-4o-mini";

      const completion = await openai.chat.completions.create({
        model,
        messages: [
          { role: "system", content: systemPrompt },
          ...sanitizedHistory.map((m) => ({ role: m.role, content: m.content })),
          { role: "user", content: trimmed },
        ],
        max_tokens: 1024,
      });

      const assistantText = completion.choices[0]?.message?.content?.trim();
      if (assistantText) {
        return NextResponse.json({
          message: assistantText,
          source: "OPENAI_RESPONSES_API",
          status: "success",
        });
      }
    } catch (err: unknown) {
      console.error("[NexSpace AI] OpenAI call error, trying fallback:", err);
      // Fall through to Anthropic or knowledge fallback
    }
  }

  // Try Anthropic if configured
  if (anthropicKey) {
    try {
      const anthropic = new Anthropic({ apiKey: anthropicKey });
      const model = process.env.NEXSPACE_ASSISTANT_MODEL || "claude-3-5-haiku-20241022";

      const response = await anthropic.messages.create({
        model,
        max_tokens: 1024,
        system: systemPrompt,
        messages: [
          ...sanitizedHistory,
          { role: "user", content: trimmed },
        ],
      });

      const assistantText =
        response.content[0]?.type === "text" ? response.content[0].text.trim() : "";
      if (assistantText) {
        return NextResponse.json({
          message: assistantText,
          source: "ANTHROPIC_API",
          status: "success",
        });
      }
    } catch (err: unknown) {
      console.error("[NexSpace AI] Anthropic call error, trying fallback:", err);
    }
  }

  // Safe conversational fallback if API fails or returns empty
  try {
    const fallbackAnswer = answerFromKnowledgeBase(trimmed, sanitizedHistory);
    return NextResponse.json({
      message: fallbackAnswer,
      source: "NEXSPACE_CONVERSATIONAL_FALLBACK",
      status: "success",
    });
  } catch (err: unknown) {
    console.error("[NexSpace AI] Fallback error:", err);
    return NextResponse.json({
      message: "Oops 😭 kuch technical issue aa gaya. Ek baar phir try kar.",
      source: "SAFE_ERROR_RECOVERY",
      status: "success",
    });
  }
}

// Block all non-POST methods
export function GET(): NextResponse {
  return NextResponse.json({ error: "Method not allowed." }, { status: 405 });
}
