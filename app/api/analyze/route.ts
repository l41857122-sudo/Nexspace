import { NextResponse } from "next/server";

const ML_BACKEND_URL = process.env.ML_BACKEND_URL || "http://127.0.0.1:8000";

export async function POST(req: Request) {
  const contentType = req.headers.get("content-type") || "";

  try {
    let fetchOptions: RequestInit;

    if (contentType.includes("multipart/form-data")) {
      // Reconstruct FormData for FastAPI without manually setting Content-Type header
      // This allows the runtime fetch implementation to generate the proper multipart boundary
      const clientFormData = await req.formData();
      const proxyFormData = new FormData();

      for (const [key, value] of clientFormData.entries()) {
        proxyFormData.append(key, value);
      }

      fetchOptions = {
        method: "POST",
        body: proxyFormData,
        signal: AbortSignal.timeout(120000),
      };
    } else {
      // JSON or raw body
      const bodyText = await req.text();
      fetchOptions = {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: bodyText,
        signal: AbortSignal.timeout(120000),
      };
    }

    const res = await fetch(`${ML_BACKEND_URL}/analyze`, fetchOptions);
    const data = await res.json().catch(() => ({ detail: "ML Backend Error" }));
    return NextResponse.json(data, { status: res.status });
  } catch (err: any) {
    return NextResponse.json(
      {
        detail: `FastAPI ML backend unreachable at ${ML_BACKEND_URL}: ${err?.message || "Connection failed"}`,
        status: "offline",
      },
      { status: 503 }
    );
  }
}
