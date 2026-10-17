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
  const ext = fileName.toLowerCase().split('.').pop() || '';
  const isGeoRaster = ['tif', 'tiff', 'jp2', 'safe', 'zip'].includes(ext);

  const job: IngestionJobStatus = {
    id,
    fileName,
    fileSize,
    progress: 10,
    status: "uploading",
    statusMessage: isGeoRaster ? "Reading geospatial raster header..." : "Reading image header...",
    validationChecklist: {
      crs: isGeoRaster ? "Detecting · Validating GeoTIFF CRS..." : "None detected (Standard RGB)",
      resolution: isGeoRaster ? "Analyzing · Extraction" : "Pixel Resolution",
      cloudCover: isGeoRaster ? "Awaiting QA · Band Analysis" : "Not applicable (RGB)"
    }
  };
  activeUploads.set(id, job);
  return job;
}

export function getUploadJobStatus(id: string): IngestionJobStatus {
  const job = activeUploads.get(id);
  if (!job) {
    return {
      id,
      fileName: "imagery_upload.png",
      fileSize: "2.4 MB",
      progress: 100,
      status: "completed",
      statusMessage: "Dataset successfully indexed into Vector Database",
      validationChecklist: {
        crs: "None detected (Standard RGB)",
        resolution: "Pixel Resolution",
        cloudCover: "Not applicable (RGB)"
      }
    };
  }

  const ext = job.fileName.toLowerCase().split('.').pop() || '';
  const isGeoRaster = ['tif', 'tiff', 'jp2', 'safe', 'zip'].includes(ext);

  // Advance progress simulation
  if (job.status === "uploading") {
    job.progress = Math.min(100, job.progress + 30);
    if (job.progress >= 30 && job.progress < 60) {
      job.statusMessage = isGeoRaster ? "Extracting multi-spectral raster layers..." : "Extracting image layers...";
      job.validationChecklist.resolution = isGeoRaster ? "Analyzing · Extraction" : "Pixel Resolution";
    } else if (job.progress >= 60 && job.progress < 90) {
      job.statusMessage = isGeoRaster ? "Calculating NDVI & Cloud Occlusion matrix..." : "Calculating visual feature embeddings...";
      job.validationChecklist.resolution = isGeoRaster ? "Extracted · 10m GSD" : "Pixel Dimensions";
      if (isGeoRaster) {
        job.validationChecklist.crs = "Detected · Valid UTM (EPSG:32651)";
      }
    } else if (job.progress >= 90 && job.progress < 100) {
      job.statusMessage = isGeoRaster ? "Building spatial tile pyramid (COG format)..." : "Normalizing RGB visual asset...";
    } else if (job.progress >= 100) {
      job.progress = 100;
      job.status = "completed";
      job.statusMessage = "Dataset successfully indexed into Vector Database";
      job.validationChecklist.cloudCover = isGeoRaster ? "Verified · 4.2% Cloud" : "Not applicable (RGB)";
    }
  }

  return job;
}
