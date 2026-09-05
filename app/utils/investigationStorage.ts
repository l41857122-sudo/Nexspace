/**
 * app/utils/investigationStorage.ts
 * ---------------------------------
 * Single Canonical Investigation State and Image Identity Manager.
 * Ensures zero silent overrides of user selections/uploads and provides
 * reactive synchronization across all pages.
 */

import { CanonicalSourceImage, CanonicalInvestigationState } from "../types/nexspace";
import { SAMPLE_OPTICAL_PORT } from "./sampleImages";

const STORAGE_KEY_INVESTIGATION = "nexspace_current_investigation";
const STORAGE_KEY_ACTIVE_SOURCE = "nexspace_active_source_image";
const STORAGE_KEY_HISTORY = "nexspace_investigation_history";

export const DEFAULT_DEMO_SOURCE: CanonicalSourceImage = {
  id: "src-demo-port",
  filename: "port.png",
  mediaType: "image/png",
  dataUrl: SAMPLE_OPTICAL_PORT,
  source: "demo",
};

/**
 * Safe storage helper with memory fallback in case quota is exceeded or storage unavailable
 */
const memoryStore: Record<string, string> = {};

function safeSetItem(key: string, value: string): void {
  memoryStore[key] = value;
  if (typeof window === "undefined") return;
  try {
    sessionStorage.setItem(key, value);
  } catch {
    try {
      localStorage.setItem(key, value);
    } catch {
      // Memory store is already populated
    }
  }
}

function safeGetItem(key: string): string | null {
  if (typeof window === "undefined") return memoryStore[key] || null;
  try {
    const sVal = sessionStorage.getItem(key);
    if (sVal) return sVal;
  } catch {}
  try {
    const lVal = localStorage.getItem(key);
    if (lVal) return lVal;
  } catch {}
  return memoryStore[key] || null;
}

function safeRemoveItem(key: string): void {
  delete memoryStore[key];
  if (typeof window === "undefined") return;
  try { sessionStorage.removeItem(key); } catch {}
  try { localStorage.removeItem(key); } catch {}
}

export function getActiveSourceImage(fallbackToDemo = true): CanonicalSourceImage | null {
  try {
    const raw = safeGetItem(STORAGE_KEY_ACTIVE_SOURCE);
    if (raw) {
      const parsed = JSON.parse(raw) as CanonicalSourceImage;
      if (parsed && parsed.dataUrl && parsed.filename) {
        return parsed;
      }
    }
  } catch (err) {
    console.warn("[investigationStorage] Error reading active source image:", err);
  }
  return fallbackToDemo ? DEFAULT_DEMO_SOURCE : null;
}

export function setActiveSourceImage(source: CanonicalSourceImage): void {
  try {
    safeSetItem(STORAGE_KEY_ACTIVE_SOURCE, JSON.stringify(source));
    if (typeof window !== "undefined") {
      window.dispatchEvent(new CustomEvent("nexspace-source-changed", { detail: source }));
    }
  } catch (err) {
    console.error("[investigationStorage] Error saving active source image:", err);
  }
}

export function getCurrentInvestigation(): CanonicalInvestigationState | null {
  try {
    const raw = safeGetItem(STORAGE_KEY_INVESTIGATION);
    if (raw) {
      const parsed = JSON.parse(raw) as CanonicalInvestigationState;
      if (parsed && (parsed.investigation_id || parsed.query)) {
        // Ensure source_image has dataUrl from active source if missing
        if (!parsed.source_image?.dataUrl) {
          const activeSrc = getActiveSourceImage(false);
          if (activeSrc) {
            parsed.source_image = activeSrc;
          }
        }
        return parsed;
      }
    }
  } catch (err) {
    console.warn("[investigationStorage] Error reading current investigation:", err);
  }
  return null;
}

export function getInvestigationHistory(): CanonicalInvestigationState[] {
  try {
    const raw = safeGetItem(STORAGE_KEY_HISTORY);
    if (raw) {
      const parsed = JSON.parse(raw);
      if (Array.isArray(parsed)) {
        return parsed;
      }
    }
  } catch (err) {
    console.warn("[investigationStorage] Error reading investigation history:", err);
  }
  return [];
}

export function setCurrentInvestigation(inv: CanonicalInvestigationState): void {
  try {
    // 1. Persist current investigation
    safeSetItem(STORAGE_KEY_INVESTIGATION, JSON.stringify(inv));

    // 2. Persist active source image
    if (inv.source_image && inv.source_image.dataUrl) {
      setActiveSourceImage(inv.source_image);
    }

    // 3. Compact history item (omit duplicate heavy base64 dataUrl in history entries to prevent quota overflow)
    const compactInv: CanonicalInvestigationState = {
      ...inv,
      source_image: inv.source_image ? {
        id: inv.source_image.id,
        filename: inv.source_image.filename,
        mediaType: inv.source_image.mediaType,
        source: inv.source_image.source,
        dataUrl: "", // Omit large base64 dataUrl in history items to conserve quota
        sha256: inv.source_image.sha256,
        uploadedAt: inv.source_image.uploadedAt,
        width: inv.source_image.width,
        height: inv.source_image.height,
      } : inv.source_image
    };

    const history = getInvestigationHistory();
    const existingIndex = history.findIndex((h) => h.investigation_id === inv.investigation_id);
    if (existingIndex >= 0) {
      history[existingIndex] = compactInv;
    } else {
      history.unshift(compactInv);
      if (history.length > 25) {
        history.pop();
      }
    }
    safeSetItem(STORAGE_KEY_HISTORY, JSON.stringify(history));

    if (typeof window !== "undefined") {
      window.dispatchEvent(new CustomEvent("nexspace-investigation-changed", { detail: inv }));
    }
  } catch (err) {
    console.error("[investigationStorage] Error saving current investigation:", err);
  }
}

export function updateSelectedTarget(targetId: string | null): void {
  try {
    const current = getCurrentInvestigation();
    if (current) {
      current.selectedTargetId = targetId;
      safeSetItem(STORAGE_KEY_INVESTIGATION, JSON.stringify(current));
      if (typeof window !== "undefined") {
        window.dispatchEvent(new CustomEvent("nexspace-target-changed", { detail: { targetId } }));
      }
    }
  } catch (err) {
    console.error("[investigationStorage] Error updating selected target:", err);
  }
}

export function clearCurrentInvestigation(retainSourceImage = true): void {
  try {
    safeRemoveItem(STORAGE_KEY_INVESTIGATION);
    if (!retainSourceImage) {
      safeRemoveItem(STORAGE_KEY_ACTIVE_SOURCE);
    }
    if (typeof window !== "undefined") {
      window.dispatchEvent(new CustomEvent("nexspace-investigation-cleared"));
    }
  } catch (err) {
    console.error("[investigationStorage] Error clearing investigation:", err);
  }
}
