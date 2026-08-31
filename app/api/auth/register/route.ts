import { NextResponse } from "next/server";
import bcrypt from "bcryptjs";
import crypto from "crypto";

export async function POST(req: Request) {
  try {
    const body = await req.json();
    const { email, password } = body;

    if (!email || !password) {
      return NextResponse.json({ error: { code: "BAD_REQUEST", message: "Email and password are required." } }, { status: 400 });
    }

    const apiKey = `sq_live_${crypto.randomBytes(12).toString("hex")}`;
    const passwordHash = await bcrypt.hash(password, 10);

    return NextResponse.json({
      user: {
        id: `usr_${Date.now()}`,
        email,
        apiKey,
        twoFactorOn: false
      },
      token: "mock_jwt_token_session"
    });
  } catch (err: any) {
    return NextResponse.json({ error: { code: "SERVER_ERROR", message: err.message } }, { status: 500 });
  }
}
