import { NextResponse } from "next/server";

export async function POST(req: Request) {
  try {
    const body = await req.json();
    const { email, password } = body;

    return NextResponse.json({
      user: {
        id: "usr_active_admin",
        email: email || "operator@nexspace.ai",
        apiKey: "nx_live_8f92a4b928104719x921k",
        twoFactorOn: true
      },
      token: "mock_session_token_valid"
    });
  } catch (err: any) {
    return NextResponse.json({ error: { code: "SERVER_ERROR", message: err.message } }, { status: 500 });
  }
}
