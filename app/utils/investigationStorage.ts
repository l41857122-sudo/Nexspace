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

export const DEFAULT_DEMO_SOURCE: CanonicalSourceImage = {
  id: "src-demo-port",
  filename: "port.png",
  mediaType: "image/png",
  dataUrl: SAMPLE_OPTICAL_PORT,
  source: "demo",
};

export function getActiveSourceImage(fallbackToDemo = true): CanonicalSourceImage | null {
  if (typeof window === "undefined") {
    return fallbackToDemo ? DEFAULT_DEMO_SOURCE : null;
  }
  try {
    const raw = sessionStorage.getItem(STORAGE_KEY_ACTIVE_SOURCE);
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
  if (typeof window === "undefined") return;
  try {
    sessionStorage.setItem(STORAGE_KEY_ACTIVE_SOURCE, JSON.stringify(source));
    window.dispatchEvent(new CustomEvent("nexspace-source-changed", { detail: source }));
  } catch (err) {
    console.error("[investigationStorage] Error saving active source image:", err);
  }
}

export function getCurrentInvestigation(): CanonicalInvestigationState | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = sessionStorage.getItem(STORAGE_KEY_INVESTIGATION);
    if (raw) {
      const parsed = JSON.parse(raw) as CanonicalInvestigationState;
      if (parsed && parsed.investigation_id) {
        return parsed;
      }
    }
  } catch (err) {
    console.warn("[investigationStorage] Error reading current investigation:", err);
  }
  return null;
}

export function setCurrentInvestigation(inv: CanonicalInvestigationState): void {
  if (typeof window === "undefined") return;
  try {
    sessionStorage.setItem(STORAGE_KEY_INVESTIGATION, JSON.stringify(inv));
    if (inv.source_image) {
      sessionStorage.setItem(STORAGE_KEY_ACTIVE_SOURCE, JSON.stringify(inv.source_image));
    }
    window.dispatchEvent(new CustomEvent("nexspace-investigation-changed", { detail: inv }));
  } catch (err) {
    console.error("[investigationStorage] Error saving current investigation:", err);
  }
}

export function updateSelectedTarget(targetId: string | null): void {
  if (typeof window === "undefined") return;
  try {
    const current = getCurrentInvestigation();
    if (current) {
      current.selectedTargetId = targetId;
      sessionStorage.setItem(STORAGE_KEY_INVESTIGATION, JSON.stringify(current));
      window.dispatchEvent(new CustomEvent("nexspace-target-changed", { detail: { targetId } }));
    }
  } catch (err) {
    console.error("[investigationStorage] Error updating selected target:", err);
  }
}

export function clearCurrentInvestigation(retainSourceImage = true): void {
  if (typeof window === "undefined") return;
  try {
    sessionStorage.removeItem(STORAGE_KEY_INVESTIGATION);
    if (!retainSourceImage) {
      sessionStorage.removeItem(STORAGE_KEY_ACTIVE_SOURCE);
    }
    window.dispatchEvent(new CustomEvent("nexspace-investigation-cleared"));
  } catch (err) {
    console.error("[investigationStorage] Error clearing investigation:", err);
  }
}
