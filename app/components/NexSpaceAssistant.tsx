"use client";

import { useState, useRef, useEffect, useCallback, KeyboardEvent } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { X, Send, ChevronDown, Copy, Check, Sparkles, Terminal } from "lucide-react";
import { QUICK_ACTION_CHIPS } from "../../lib/nexspace-ai/knowledge";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
}

type AssistantStatus = "idle" | "thinking" | "error";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const MAX_INPUT_LENGTH = 1000;

const WELCOME_MESSAGE: Message = {
  id: "welcome",
  role: "assistant",
  content:
    "Hey 👋 I'm NexSpace AI.\n\nYou can ask me about NexSpace, satellite intelligence, tech, coding — or literally just say hi.",
  timestamp: new Date(),
};

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

function generateId(): string {
  return Math.random().toString(36).slice(2, 11);
}

// ---------------------------------------------------------------------------
// Code Block with Copy Button
// ---------------------------------------------------------------------------

function CodeBlock({ code, language }: { code: string; language?: string }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // ignore clipboard error
    }
  };

  return (
    <div className="my-2.5 rounded-lg overflow-hidden border border-cyan-500/20 bg-[#040810] shadow-inner font-mono text-[11.5px]">
      <div className="flex items-center justify-between px-3 py-1.5 bg-cyan-950/30 border-b border-cyan-500/15 text-[10px] text-cyan-400/80">
        <span className="flex items-center gap-1.5 uppercase tracking-wider font-semibold">
          <Terminal size={11} className="text-cyan-400" />
          {language || "code"}
        </span>
        <button
          onClick={handleCopy}
          className="flex items-center gap-1 text-slate-400 hover:text-cyan-300 transition-colors px-1.5 py-0.5 rounded hover:bg-white/5"
          aria-label="Copy code"
        >
          {copied ? <Check size={11} className="text-emerald-400" /> : <Copy size={11} />}
          <span>{copied ? "Copied" : "Copy"}</span>
        </button>
      </div>
      <pre className="p-3 overflow-x-auto text-slate-200 leading-relaxed font-mono selection:bg-cyan-500/30">
        <code>{code}</code>
      </pre>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Rich Markdown & Text Formatter
// ---------------------------------------------------------------------------

function FormattedContent({ content }: { content: string }) {
  // Parse code blocks with ```lang ... ```
  const codeBlockRegex = /```([a-zA-Z0-9_-]*)\n?([\s\S]*?)```/g;
  const parts: React.ReactNode[] = [];
  let lastIndex = 0;
  let match: RegExpExecArray | null;

  while ((match = codeBlockRegex.exec(content)) !== null) {
    if (match.index > lastIndex) {
      parts.push(
        <TextSegment key={`text-${lastIndex}`} text={content.slice(lastIndex, match.index)} />
      );
    }
    const lang = match[1] || "";
    const code = match[2]?.trim() || "";
    parts.push(<CodeBlock key={`code-${match.index}`} code={code} language={lang} />);
    lastIndex = match.index + match[0].length;
  }

  if (lastIndex < content.length) {
    parts.push(<TextSegment key={`text-${lastIndex}`} text={content.slice(lastIndex)} />);
  }

  return <div className="space-y-1">{parts}</div>;
}

function TextSegment({ text }: { text: string }) {
  const lines = text.split("\n");

  return (
    <>
      {lines.map((line, i) => {
        // Format bullet points
        const isBullet = line.trim().startsWith("•") || line.trim().startsWith("- ");
        const isNumbered = /^\d+\.\s/.test(line.trim());

        // Format bold text (**bold**)
        const formattedLine = line.split(/(\*\*[^*]+\*\*)/g).map((chunk, chunkIdx) => {
          if (chunk.startsWith("**") && chunk.endsWith("**")) {
            return (
              <strong key={chunkIdx} className="font-semibold text-white">
                {chunk.slice(2, -2)}
              </strong>
            );
          }
          // Inline code `code`
          return chunk.split(/(`[^`]+`)/g).map((subChunk, subIdx) => {
            if (subChunk.startsWith("`") && subChunk.endsWith("`")) {
              return (
                <code
                  key={subIdx}
                  className="px-1.5 py-0.5 mx-0.5 rounded bg-cyan-950/50 border border-cyan-500/20 text-cyan-300 font-mono text-[11px]"
                >
                  {subChunk.slice(1, -1)}
                </code>
              );
            }
            return subChunk;
          });
        });

        return (
          <span key={i} className="block leading-relaxed">
            {isBullet ? (
              <span className="flex items-start gap-1.5 text-slate-200">
                <span className="text-cyan-400 flex-shrink-0 mt-1">•</span>
                <span>{formattedLine}</span>
              </span>
            ) : isNumbered ? (
              <span className="text-cyan-300/90 font-medium">{formattedLine}</span>
            ) : (
              <span>{formattedLine}</span>
            )}
          </span>
        );
      })}
    </>
  );
}

// ---------------------------------------------------------------------------
// Thinking Indicator
// ---------------------------------------------------------------------------

function ThinkingIndicator() {
  return (
    <div className="flex items-center gap-2 text-cyan-400 text-[11px] font-mono py-1.5">
      <span className="inline-block w-2 h-2 rounded-full bg-cyan-400 animate-ping" />
      <span className="flex items-center gap-1 text-cyan-300 font-semibold tracking-wider">
        ◉ thinking...
      </span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Message Bubble Component
// ---------------------------------------------------------------------------

function MessageBubble({ message }: { message: Message }) {
  const isUser = message.role === "user";

  if (isUser) {
    return (
      <motion.div
        initial={{ opacity: 0, y: 6 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.18 }}
        className="flex justify-end my-2"
      >
        <div
          className="
            max-w-[85%] text-[13px] leading-relaxed px-4 py-2.5 rounded-2xl rounded-br-sm
            bg-cyan-500/15 border border-cyan-500/30 text-cyan-100 shadow-sm
          "
        >
          {message.content}
        </div>
      </motion.div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.2 }}
      className="flex flex-col items-start my-2 w-full"
    >
      <div className="flex items-center gap-1.5 mb-1 ml-1 text-[10px] font-mono tracking-wider uppercase text-cyan-400/90">
        <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 shadow-[0_0_8px_rgba(6,182,212,0.8)]" />
        <span className="font-semibold">◉ AI</span>
      </div>

      <div
        className="
          w-full text-[13px] leading-relaxed px-4 py-3 rounded-xl
          bg-white/[0.03] border border-white/[0.07] text-slate-200/95
          shadow-[0_4px_20px_rgba(0,0,0,0.2)]
        "
      >
        <FormattedContent content={message.content} />
      </div>
    </motion.div>
  );
}

// ---------------------------------------------------------------------------
// Main Component: NexSpaceAssistant
// ---------------------------------------------------------------------------

export default function NexSpaceAssistant() {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([WELCOME_MESSAGE]);
  const [inputValue, setInputValue] = useState("");
  const [status, setStatus] = useState<AssistantStatus>("idle");
  const [showScrollDown, setShowScrollDown] = useState(false);
  const [chipsVisible, setChipsVisible] = useState(true);

  const chatEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const chatScrollRef = useRef<HTMLDivElement>(null);

  // Conversation history for context (last 20 messages)
  const conversationRef = useRef<Array<{ role: "user" | "assistant"; content: string }>>([]);

  const scrollToBottom = useCallback((smooth = true) => {
    chatEndRef.current?.scrollIntoView({ behavior: smooth ? "smooth" : "auto" });
  }, []);

  useEffect(() => {
    if (isOpen) {
      setTimeout(() => scrollToBottom(false), 50);
      setTimeout(() => inputRef.current?.focus(), 80);
    }
  }, [isOpen, scrollToBottom]);

  useEffect(() => {
    if (status !== "thinking") {
      scrollToBottom();
    }
  }, [messages, status, scrollToBottom]);

  const handleChatScroll = () => {
    const el = chatScrollRef.current;
    if (!el) return;
    const distFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
    setShowScrollDown(distFromBottom > 80);
  };

  const sendMessage = useCallback(
    async (text: string) => {
      const trimmed = text.trim();
      if (!trimmed || status === "thinking") return;

      setChipsVisible(false);

      const userMsg: Message = {
        id: generateId(),
        role: "user",
        content: trimmed,
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, userMsg]);
      setInputValue("");
      setStatus("thinking");

      const history = conversationRef.current.slice(-20);

      try {
        const res = await fetch("/api/assistant/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            message: trimmed,
            conversation: history,
          }),
          signal: AbortSignal.timeout(35_000),
        });

        const data = await res.json();

        if (!res.ok || data.status === "error") {
          const errMsg: Message = {
            id: generateId(),
            role: "assistant",
            content: data.error || "Oops 😭 kuch technical issue aa gaya. Ek baar phir try kar.",
            timestamp: new Date(),
          };
          setMessages((prev) => [...prev, errMsg]);
          setStatus("error");
          setTimeout(() => setStatus("idle"), 3000);
          return;
        }

        const assistantMsg: Message = {
          id: generateId(),
          role: "assistant",
          content: data.message || "Oops 😭 kuch technical issue aa gaya. Ek baar phir try kar.",
          timestamp: new Date(),
        };

        setMessages((prev) => [...prev, assistantMsg]);
        setStatus("idle");

        // Update conversation context
        conversationRef.current = [
          ...conversationRef.current,
          { role: "user" as const, content: trimmed },
          { role: "assistant" as const, content: assistantMsg.content },
        ].slice(-20);
      } catch {
        const errMsg: Message = {
          id: generateId(),
          role: "assistant",
          content: "Oops 😭 kuch technical issue aa gaya. Ek baar phir try kar.",
          timestamp: new Date(),
        };
        setMessages((prev) => [...prev, errMsg]);
        setStatus("error");
        setTimeout(() => setStatus("idle"), 3000);
      }
    },
    [status]
  );

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage(inputValue);
    }
  };

  const handleChipClick = (chip: string) => {
    sendMessage(chip);
  };

  const handleToggle = () => {
    setIsOpen((prev) => !prev);
  };

  const handleClose = () => {
    setIsOpen(false);
  };

  useEffect(() => {
    const handleGlobalKey = (e: globalThis.KeyboardEvent) => {
      if (e.key === "Escape" && isOpen) {
        setIsOpen(false);
      }
    };
    window.addEventListener("keydown", handleGlobalKey);
    return () => window.removeEventListener("keydown", handleGlobalKey);
  }, [isOpen]);

  const charsLeft = MAX_INPUT_LENGTH - inputValue.length;
  const isOverLimit = charsLeft < 0;

  return (
    <div
      id="nexspace-assistant-container"
      className="fixed z-50 bottom-6 right-6 font-sans pointer-events-auto"
    >
      {/* ------------------------------------------------------------------ */}
      {/* Main Chat Panel (V2 Mission-Control HUD)                            */}
      {/* ------------------------------------------------------------------ */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            id="nexspace-assistant-panel"
            initial={{ opacity: 0, scale: 0.94, y: 16 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.94, y: 16 }}
            transition={{ type: "spring", damping: 26, stiffness: 320 }}
            className="
              fixed sm:absolute bottom-0 right-0 sm:bottom-14 sm:right-0
              w-full sm:w-[410px] h-[580px] sm:h-[620px] max-h-[92vh] sm:max-h-[85vh]
              rounded-t-2xl sm:rounded-2xl
              bg-[#060d17]/95
              border border-cyan-500/25
              backdrop-blur-2xl
              shadow-[0_12px_48px_rgba(0,0,0,0.8),0_0_24px_rgba(6,182,212,0.12)]
              flex flex-col
              overflow-hidden
              z-50
            "
            role="region"
            aria-label="NexSpace Conversational AI"
          >
            {/* ---- Technical Header ---- */}
            <div className="flex-shrink-0 flex items-center justify-between px-4 py-3 bg-[#081322]/80 border-b border-cyan-500/20 backdrop-blur-md">
              <div className="flex items-center gap-2.5">
                <div className="relative flex-shrink-0 w-7 h-7 rounded-lg bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center shadow-[0_0_12px_rgba(6,182,212,0.2)]">
                  <Sparkles size={14} className="text-cyan-400 animate-pulse" />
                  <span className="absolute -top-0.5 -right-0.5 w-1.5 h-1.5 rounded-full bg-cyan-400 shadow-[0_0_6px_rgba(6,182,212,1)]" />
                </div>
                <div>
                  <div className="flex items-center gap-2">
                    <span className="text-[12px] font-bold tracking-wider font-mono text-white">
                      ◉ NEXSPACE AI
                    </span>
                    <span className="text-[8px] font-mono px-1.5 py-0.5 rounded bg-cyan-500/15 text-cyan-300 border border-cyan-500/30">
                      v2.0
                    </span>
                  </div>
                  <p className="text-[10px] text-cyan-400/80 font-mono tracking-wide">
                    Conversational Intelligence
                  </p>
                </div>
              </div>

              <div className="flex items-center gap-3">
                <div className="flex items-center gap-1.5 px-2 py-0.5 rounded-full bg-emerald-500/10 border border-emerald-500/25">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                  <span className="text-[9px] font-mono tracking-wider text-emerald-400 font-semibold uppercase">
                    ● Online
                  </span>
                </div>
                <button
                  id="nexspace-assistant-close-btn"
                  onClick={handleClose}
                  aria-label="Close NexSpace AI"
                  className="text-slate-400 hover:text-white hover:bg-white/10 transition-colors p-1 rounded-lg"
                >
                  <X size={16} />
                </button>
              </div>
            </div>

            {/* ---- Message Scroll Area ---- */}
            <div
              ref={chatScrollRef}
              onScroll={handleChatScroll}
              className="flex-1 overflow-y-auto px-4 py-3 space-y-3 min-h-0 selection:bg-cyan-500/20"
              style={{ scrollbarWidth: "thin" }}
            >
              {messages.map((msg) => (
                <MessageBubble key={msg.id} message={msg} />
              ))}

              {/* Quick action chips */}
              <AnimatePresence>
                {chipsVisible && messages.length === 1 && (
                  <motion.div
                    initial={{ opacity: 0, y: 6 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -4 }}
                    transition={{ duration: 0.2, delay: 0.1 }}
                    className="flex flex-wrap gap-1.5 pt-2"
                    role="group"
                    aria-label="Suggested questions"
                  >
                    <div className="w-full text-[10px] font-mono text-cyan-400/70 tracking-wider mb-0.5">
                      SUGGESTED TOPICS:
                    </div>
                    {QUICK_ACTION_CHIPS.map((chip) => (
                      <button
                        key={chip}
                        id={`nexspace-chip-${chip.toLowerCase().replace(/[^a-z0-9]/g, "-")}`}
                        onClick={() => handleChipClick(chip)}
                        disabled={status === "thinking"}
                        className="
                          text-[11px] px-3 py-1.5 rounded-full
                          border border-cyan-500/25 bg-cyan-500/[0.08]
                          text-cyan-300 hover:text-white hover:bg-cyan-500/20 hover:border-cyan-400/50
                          transition-all duration-150 font-mono tracking-wide
                          disabled:opacity-40 disabled:cursor-not-allowed
                          shadow-sm
                        "
                      >
                        {chip}
                      </button>
                    ))}
                  </motion.div>
                )}
              </AnimatePresence>

              {/* Thinking Indicator */}
              {status === "thinking" && (
                <motion.div
                  initial={{ opacity: 0, y: 4 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="flex justify-start my-2"
                >
                  <ThinkingIndicator />
                </motion.div>
              )}

              <div ref={chatEndRef} />
            </div>

            {/* Scroll-down button */}
            <AnimatePresence>
              {showScrollDown && (
                <motion.button
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  onClick={() => scrollToBottom()}
                  className="absolute bottom-[76px] right-4 w-7 h-7 rounded-full bg-cyan-950/80 border border-cyan-500/40 text-cyan-400 flex items-center justify-center hover:bg-cyan-900 transition-colors shadow-lg"
                  aria-label="Scroll to bottom"
                >
                  <ChevronDown size={14} />
                </motion.button>
              )}
            </AnimatePresence>

            {/* ---- Footer Input Bar ---- */}
            <div className="flex-shrink-0 border-t border-cyan-500/20 px-3 py-2.5 bg-[#07101c]/95">
              <div className="flex items-end gap-2">
                <div className="flex-1 relative">
                  <textarea
                    ref={inputRef}
                    id="nexspace-assistant-input"
                    value={inputValue}
                    onChange={(e) => setInputValue(e.target.value.slice(0, MAX_INPUT_LENGTH + 50))}
                    onKeyDown={handleKeyDown}
                    placeholder="Ask anything..."
                    rows={1}
                    aria-label="Ask NexSpace AI anything"
                    disabled={status === "thinking"}
                    className="
                      w-full resize-none bg-white/[0.04] border border-white/[0.1] rounded-xl
                      px-3.5 py-2.5 text-[12.5px] text-slate-100 placeholder-slate-500
                      focus:outline-none focus:border-cyan-400 focus:bg-white/[0.06] focus:shadow-[0_0_12px_rgba(6,182,212,0.15)]
                      transition-all duration-150 leading-relaxed
                      disabled:opacity-50 disabled:cursor-not-allowed
                      max-h-[100px] overflow-y-auto font-sans
                    "
                    style={{ scrollbarWidth: "thin" }}
                  />
                  {isOverLimit && (
                    <span className="absolute bottom-2 right-2 text-[9px] text-red-400 font-mono">
                      {Math.abs(charsLeft)} over
                    </span>
                  )}
                </div>
                <button
                  id="nexspace-assistant-send-btn"
                  onClick={() => sendMessage(inputValue)}
                  disabled={status === "thinking" || !inputValue.trim() || isOverLimit}
                  aria-label="Send message"
                  className="
                    flex-shrink-0 w-9 h-9 rounded-xl
                    bg-cyan-500/25 border border-cyan-400/40
                    text-cyan-300 hover:bg-cyan-500 hover:text-black hover:border-cyan-300
                    flex items-center justify-center
                    transition-all duration-150
                    disabled:opacity-40 disabled:cursor-not-allowed
                    shadow-[0_0_12px_rgba(6,182,212,0.15)]
                  "
                >
                  <Send size={14} />
                </button>
              </div>

              <div className="flex justify-between items-center mt-1 px-1">
                <span className="text-[9px] font-mono text-cyan-400/60">
                  NexSpace AI · Multilingual & General Intelligence
                </span>
                <span
                  className={`text-[9px] font-mono ${
                    isOverLimit
                      ? "text-red-400"
                      : charsLeft < 100
                      ? "text-amber-400"
                      : "text-slate-500/60"
                  }`}
                >
                  {Math.max(0, charsLeft)}
                </span>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ------------------------------------------------------------------ */}
      {/* Floating Launcher Button (Mission-Control V2)                        */}
      {/* ------------------------------------------------------------------ */}
      <motion.button
        id="nexspace-assistant-trigger"
        onClick={handleToggle}
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
        aria-label="Ask NexSpace AI"
        aria-expanded={isOpen}
        aria-controls="nexspace-assistant-panel"
        className="
          relative flex items-center gap-2.5
          px-4 py-2.5
          rounded-full
          border border-cyan-500/40
          bg-[#060e1a]/95
          backdrop-blur-xl
          text-cyan-300
          hover:border-cyan-400 hover:bg-cyan-950/40
          transition-all duration-200
          select-none
          group
        "
        style={{
          boxShadow:
            "0 6px 28px rgba(0,0,0,0.7), 0 0 0 1px rgba(6,182,212,0.2), 0 0 24px rgba(6,182,212,0.15)",
        }}
      >
        {!isOpen && (
          <span className="absolute -top-1 -right-1 flex h-3 w-3">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan-400 opacity-30" />
            <span className="relative inline-flex rounded-full h-3 w-3 bg-cyan-500/50 border border-cyan-300" />
          </span>
        )}

        <div className="relative flex-shrink-0 w-6 h-6 rounded-full border border-cyan-400/50 bg-cyan-500/20 flex items-center justify-center">
          {isOpen ? (
            <X size={12} className="text-cyan-300" />
          ) : (
            <span className="text-[10px] font-bold font-mono text-cyan-300 animate-pulse">◉</span>
          )}
        </div>

        <div className="text-left font-mono">
          <p className="text-[11px] font-bold tracking-[0.16em] uppercase text-cyan-200 leading-tight">
            {isOpen ? "Close" : "NEXSPACE AI"}
          </p>
          {!isOpen && (
            <p className="text-[9px] text-cyan-400/70 leading-tight tracking-wide font-sans">
              Ask anything
            </p>
          )}
        </div>
      </motion.button>
    </div>
  );
}
