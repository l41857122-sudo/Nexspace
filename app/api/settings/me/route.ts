import { NextResponse } from "next/server";

let currentSettings = {
  id: "usr_active_admin",
  email: "operator@nexspace.ai",
  apiKey: "nx_live_8f92a4b928104719x921k",
  twoFactorOn: true,
  credits: { used: 45200, total: 190000, percent: 24 },
  gpuHours: { used: 112.5, total: 250, percent: 45 },
  plan: "Orbital Plus Plan",
  renewDate: "Oct 12, 2024"
};

export async function GET() {
  return NextResponse.json(currentSettings);
}

export async function PATCH(req: Request) {
  try {
    const body = await req.json();
    currentSettings = { ...currentSettings, ...body };
    return NextResponse.json(currentSettings);
  } catch (err: any) {
    return NextResponse.json({ error: { code: "BAD_REQUEST", message: err.message } }, { status: 400 });
  }
}
