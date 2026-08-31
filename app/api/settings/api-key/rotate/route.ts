import { NextResponse } from "next/server";
import crypto from "crypto";

export async function POST() {
  const newApiKey = `sq_live_${crypto.randomBytes(12).toString("hex")}`;
  return NextResponse.json({
    apiKey: newApiKey,
    message: "API key rotated successfully. Store the new key securely."
  });
}
