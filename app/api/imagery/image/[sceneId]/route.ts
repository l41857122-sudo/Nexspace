import { NextResponse } from "next/server";

const ML_BACKEND_URL = process.env.ML_BACKEND_URL || "http://127.0.0.1:8000";

export async function GET(
  req: Request,
  { params }: { params: Promise<{ sceneId: string }> | { sceneId: string } }
) {
  const resolvedParams = await Promise.resolve(params);
  const sceneId = resolvedParams?.sceneId;

  if (!sceneId) {
    return new NextResponse("Scene ID is required", { status: 400 });
  }

  try {
    const res = await fetch(`${ML_BACKEND_URL}/api/satellite/image/${encodeURIComponent(sceneId)}`, {
      method: "GET",
      cache: "force-cache",
    });

    if (!res.ok) {
      return new NextResponse(`Image not found on backend: ${res.status}`, { status: res.status });
    }

    const contentType = res.headers.get("content-type") || "image/jpeg";
    const imageBuffer = await res.arrayBuffer();

    return new NextResponse(imageBuffer, {
      status: 200,
      headers: {
        "Content-Type": contentType,
        "Cache-Control": "public, max-age=86400, immutable",
      },
    });
  } catch (err: any) {
    console.error("[Imagery Image Proxy] Error fetching satellite raster:", err);
    return new NextResponse("Error fetching image from ML backend", { status: 502 });
  }
}
