import { NextResponse } from "next/server";
import { createUploadJob } from "@/server/services/ingestionService";

export async function POST(req: Request) {
  try {
    let fileName = "S2A_MSIL2A_20240315T105341.zip";
    let fileSize = "4.8 GB";

    if (req.headers.get("content-type")?.includes("multipart/form-data")) {
      const formData = await req.formData();
      const file = formData.get("file") as File;
      if (file) {
        fileName = file.name;
        fileSize = (file.size / (1024 * 1024 * 1024)).toFixed(1) + " GB";
      }
    } else {
      try {
        const body = await req.json();
        if (body.fileName) fileName = body.fileName;
        if (body.fileSize) fileSize = body.fileSize;
      } catch (e) {}
    }

    const job = createUploadJob(fileName, fileSize);
    return NextResponse.json(job);
  } catch (err: any) {
    return NextResponse.json({ error: { code: "SERVER_ERROR", message: err.message } }, { status: 500 });
  }
}
