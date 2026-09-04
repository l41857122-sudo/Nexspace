/**
 * imageValidation.ts
 * -------------------
 * Universal image format validation, MIME/extension normalization,
 * and canonical source image constructor for NexSpace.
 * 
 * Supports: JPG, JPEG, PNG, WEBP, TIFF (.tif, .tiff), GeoTIFF.
 */

import type { CanonicalSourceImage } from "../types/nexspace";

export interface ImageValidationResult {
  valid: boolean;
  error?: string;
  source?: CanonicalSourceImage;
}

export const ACCEPTED_IMAGE_MIME_TYPES = [
  "image/jpeg",
  "image/jpg",
  "image/pjpeg",
  "image/png",
  "image/webp",
  "image/tiff",
  "image/tif",
  "image/x-tiff",
];

export const ACCEPTED_IMAGE_EXTENSIONS = [
  ".jpg",
  ".jpeg",
  ".png",
  ".webp",
  ".tif",
  ".tiff",
];

export const ACCEPT_FILE_ATTR =
  "image/png,image/jpeg,image/jpg,image/pjpeg,image/webp,image/tiff,image/tif,image/x-tiff,.jpg,.jpeg,.png,.webp,.tif,.tiff";

/**
 * Validates a File object against size, extension, and MIME type rules.
 */
export async function validateAndProcessImageFile(
  file: File,
  maxSizeBytes: number = 25 * 1024 * 1024
): Promise<ImageValidationResult> {
  if (!file) {
    return { valid: false, error: "No file provided." };
  }

  // 1. Check size limit
  if (file.size > maxSizeBytes) {
    const sizeMb = (file.size / (1024 * 1024)).toFixed(1);
    const maxMb = (maxSizeBytes / (1024 * 1024)).toFixed(0);
    return {
      valid: false,
      error: `File '${file.name}' (${sizeMb} MB) exceeds maximum permitted size of ${maxMb} MB.`,
    };
  }

  // 2. Normalize extension and MIME
  const nameLower = file.name.toLowerCase();
  const hasValidExt = ACCEPTED_IMAGE_EXTENSIONS.some((ext) => nameLower.endsWith(ext));
  const rawType = (file.type || "").toLowerCase();
  const hasValidMime = ACCEPTED_IMAGE_MIME_TYPES.includes(rawType);

  // Accept if extension is valid, even if browser failed to report MIME type (common on Windows for TIFF/raw JPEGs)
  if (!hasValidExt && !hasValidMime) {
    return {
      valid: false,
      error: `Unsupported file format '${file.name}'. Please upload a valid JPG, JPEG, PNG, WEBP, or TIFF raster.`,
    };
  }

  // Determine normalized media type
  let mediaType = file.type || "image/jpeg";
  if (nameLower.endsWith(".jpg") || nameLower.endsWith(".jpeg")) {
    mediaType = "image/jpeg";
  } else if (nameLower.endsWith(".png")) {
    mediaType = "image/png";
  } else if (nameLower.endsWith(".webp")) {
    mediaType = "image/webp";
  } else if (nameLower.endsWith(".tif") || nameLower.endsWith(".tiff")) {
    mediaType = "image/tiff";
  }

  // 3. Read file as Data URL and compute SHA-256
  return new Promise((resolve) => {
    const reader = new FileReader();

    reader.onload = async () => {
      if (typeof reader.result !== "string") {
        resolve({ valid: false, error: "Failed to read image data stream." });
        return;
      }

      let sha256Hex = "";
      try {
        const arrayBuffer = await file.arrayBuffer();
        const hashBuffer = await crypto.subtle.digest("SHA-256", arrayBuffer);
        const hashArray = Array.from(new Uint8Array(hashBuffer));
        sha256Hex = hashArray.map((b) => b.toString(16).padStart(2, "0")).join("");
      } catch {
        // Fallback SHA-256 calculation if SubtleCrypto is unavailable in insecure context
        let hash = 0;
        for (let i = 0; i < reader.result.length; i++) {
          hash = (hash << 5) - hash + reader.result.charCodeAt(i);
          hash |= 0;
        }
        sha256Hex = `pseudo_${Math.abs(hash).toString(16)}`;
      }

      const canonicalSource: CanonicalSourceImage = {
        id: `src-upload-${Date.now()}`,
        filename: file.name,
        mediaType,
        dataUrl: reader.result,
        source: "upload",
        sha256: sha256Hex,
        uploadedAt: new Date().toISOString(),
      };

      resolve({
        valid: true,
        source: canonicalSource,
      });
    };

    reader.onerror = () => {
      resolve({ valid: false, error: `Failed to read file '${file.name}'.` });
    };

    reader.readAsDataURL(file);
  });
}
