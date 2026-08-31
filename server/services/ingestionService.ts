export interface IngestionJobStatus {
  id: string;
  fileName: string;
  fileSize: string;
  progress: number;
  status: "uploading" | "completed" | "error";
  statusMessage: string;
  validationChecklist: {
    crs: string;
    resolution: string;
    cloudCover: string;
  };
}

const activeUploads = new Map<string, IngestionJobStatus>();

export function createUploadJob(fileName: string, fileSize: string): IngestionJobStatus {
  const id = `upload_${Date.now()}_${Math.random().toString(36).substring(2, 7)}`;
  const job: IngestionJobStatus = {
    id,
    fileName,
    fileSize,
    progress: 10,
    status: "uploading",
    statusMessage: "Reading geospatial raster header...",
    validationChecklist: {
      crs: "Detected · Valid UTM (EPSG:32651)",
      resolution: "Analyzing · Extraction",
      cloudCover: "Awaiting QA · Band Analysis"
    }
  };
  activeUploads.set(id, job);
  return job;
}

export function getUploadJobStatus(id: string): IngestionJobStatus {
  const job = activeUploads.get(id);
  if (!job) {
    // Default fallback mock status if job ID not found
    return {
      id,
      fileName: "S2A_MSIL2A_20240315T105341.zip",
      fileSize: "4.8 GB",
      progress: 100,
      status: "completed",
      statusMessage: "Dataset successfully indexed into Vector Database",
      validationChecklist: {
        crs: "Detected · Valid UTM (EPSG:32651)",
        resolution: "Extracted · 10m GSD",
        cloudCover: "Verified · 4.2% Cloud"
      }
    };
  }

  // Advance progress simulation
  if (job.status === "uploading") {
    job.progress = Math.min(100, job.progress + 30);
    if (job.progress >= 30 && job.progress < 60) {
      job.statusMessage = "Extracting multi-spectral raster layers...";
      job.validationChecklist.resolution = "Analyzing · Extraction";
    } else if (job.progress >= 60 && job.progress < 90) {
      job.statusMessage = "Calculating NDVI & Cloud Occlusion matrix...";
      job.validationChecklist.resolution = "Extracted · 10m GSD";
    } else if (job.progress >= 90 && job.progress < 100) {
      job.statusMessage = "Building spatial tile pyramid (COG format)...";
    } else if (job.progress >= 100) {
      job.progress = 100;
      job.status = "completed";
      job.statusMessage = "Dataset successfully indexed into Vector Database";
      job.validationChecklist.cloudCover = "Verified · 4.2% Cloud";
    }
  }

  return job;
}
