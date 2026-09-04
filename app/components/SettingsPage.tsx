"use client";

import { useState, useEffect } from "react";
import {
  KeyRound,
  FileText,
  Gauge,
  ShieldCheck,
  Satellite,
  Cpu,
  CreditCard,
  Settings as SettingsIcon,
  Copy,
  Check,
} from "lucide-react";
import Sidebar from "./Sidebar";

export default function SettingsPage() {
  const [twoFactor, setTwoFactor] = useState(true);
  const [showKey, setShowKey] = useState(false);
  const [copied, setCopied] = useState(false);
  const [apiKey, setApiKey] = useState("sq_live_8f92a4b928104719x921k");

  useEffect(() => {
    fetch("/api/settings/me")
      .then((res) => res.json())
      .then((data) => {
        if (data.apiKey) setApiKey(data.apiKey);
        if (typeof data.twoFactorOn === "boolean") setTwoFactor(data.twoFactorOn);
      })
      .catch((err) => console.error(err));
  }, []);

  const handleCopy = () => {
    navigator.clipboard?.writeText(apiKey);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleRotateKey = async () => {
    try {
      const res = await fetch("/api/settings/api-key/rotate", { method: "POST" });
      const data = await res.json();
      if (data.apiKey) {
        setApiKey(data.apiKey);
        setShowKey(true);
      }
    } catch (e) {
      console.error(e);
    }
  };

  const handleToggle2FA = async () => {
    const nextVal = !twoFactor;
    setTwoFactor(nextVal);
    try {
      await fetch("/api/settings/me", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ twoFactorOn: nextVal })
      });
    } catch (e) {
      console.error(e);
    }
  };

  return (
    <div className="flex flex-col lg:flex-row min-h-screen bg-[#0a1420] text-white font-sans overflow-x-hidden">
      <Sidebar />

      <div className="flex-1 flex flex-col min-w-0 overflow-y-auto">
        {/* Header */}
        <div className="px-4 sm:px-8 py-5 border-b border-slate-800/80 bg-[#0d1826]/40 backdrop-blur-md flex items-center justify-between">
          <div>
            <div className="flex items-center gap-2">
              <SettingsIcon size={15} className="text-cyan-400" />
              <h1 className="text-base sm:text-lg font-semibold text-white tracking-tight">
                System Settings &amp; Credentials
              </h1>
            </div>
            <p className="mt-1 text-[11px] sm:text-xs text-slate-400 font-mono">
              Manage API credentials, monitor compute utilization, and configure your Orbital Plus environment.
            </p>
          </div>
        </div>

        <div className="px-4 sm:px-8 py-6 sm:py-8">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Left / main column */}
            <div className="lg:col-span-2 flex flex-col gap-6">
              {/* API Management */}
              <Panel icon={<KeyRound size={15} className="text-cyan-400" />} title="API Management">
                <div className="text-[10px] uppercase tracking-widest font-mono text-slate-400 mb-2">
                  Production API Key
                </div>
                <div className="flex items-center justify-between bg-slate-900/80 border border-slate-800 rounded-lg px-4 py-3">
                  <code className="text-xs sm:text-sm text-cyan-300 font-mono">
                    {showKey ? apiKey : "sq_live_••••••••••••x921k"}
                  </code>
                  <div className="flex items-center gap-3">
                    <button
                      onClick={handleCopy}
                      className="text-xs font-mono text-slate-400 hover:text-cyan-300 transition-colors flex items-center gap-1 cursor-pointer"
                    >
                      {copied ? <Check size={12} className="text-emerald-400" /> : <Copy size={12} />}
                      <span>{copied ? "Copied" : "Copy"}</span>
                    </button>
                    <button
                      onClick={() => setShowKey((s) => !s)}
                      className="text-xs font-medium text-cyan-400 hover:text-cyan-300 transition-colors cursor-pointer"
                    >
                      {showKey ? "Hide" : "Reveal Key"}
                    </button>
                    <button
                      onClick={handleRotateKey}
                      className="text-xs font-medium text-amber-400 hover:text-amber-300 transition-colors cursor-pointer border border-amber-500/30 px-2 py-0.5 rounded bg-amber-500/10"
                    >
                      Rotate Key
                    </button>
                  </div>
                </div>

                <div className="mt-4 flex items-center justify-between bg-slate-900/50 border border-slate-800/80 rounded-lg px-4 py-3">
                  <div>
                    <div className="text-xs text-slate-200 font-medium">
                      Developer Documentation
                    </div>
                    <div className="text-[11px] text-slate-400 mt-0.5 font-mono">
                      Access endpoints for raster analysis and vector extraction.
                    </div>
                  </div>
                  <a
                    href="#"
                    className="flex items-center gap-1 text-xs font-medium text-cyan-400 hover:text-cyan-300 transition-colors whitespace-nowrap cursor-pointer"
                  >
                    <FileText size={13} />
                    View Docs
                  </a>
                </div>
              </Panel>

              {/* Compute & Utilization */}
              <Panel icon={<Gauge size={15} className="text-cyan-400" />} title="Compute & Utilization">
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  <UsageStat
                    icon={<Satellite size={14} className="text-cyan-400" />}
                    label="Satellite Credits"
                    value="45,200"
                    max="190,000"
                    percent={24}
                    barColor="bg-cyan-400"
                  />
                  <UsageStat
                    icon={<Cpu size={14} className="text-amber-400" />}
                    label="Neural Processing (GPU)"
                    value="112.5"
                    max="250 hrs"
                    percent={45}
                    barColor="bg-amber-400"
                  />
                </div>
                <button className="mt-4 w-full rounded-lg border border-slate-800 bg-slate-900/60 py-2.5 text-xs font-medium text-slate-300 hover:text-cyan-300 hover:border-cyan-500/40 hover:bg-slate-900/90 transition-all cursor-pointer">
                  Request Quota Increase
                </button>
              </Panel>

              {/* Account & Security */}
              <Panel icon={<ShieldCheck size={15} className="text-cyan-400" />} title="Account & Security">
                <div className="flex items-center justify-between py-2">
                  <div className="flex items-center gap-3">
                    <CreditCard size={16} className="text-slate-400" />
                    <div>
                      <div className="text-sm text-slate-200 font-medium flex items-center gap-2">
                        Orbital Plus Plan
                        <span className="text-[10px] font-mono font-semibold tracking-wide bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 px-2 py-0.5 rounded-full">
                          ACTIVE
                        </span>
                      </div>
                      <div className="text-xs text-slate-400 font-mono mt-0.5">
                        Renews on Oct 12, 2024
                      </div>
                    </div>
                  </div>
                  <button className="text-xs font-medium text-cyan-400 hover:text-cyan-300 transition-colors cursor-pointer">
                    Manage Billing
                  </button>
                </div>

                <div className="h-px bg-slate-800/80 my-3" />

                <div className="flex items-center justify-between py-2">
                  <div>
                    <div className="text-sm text-slate-200 font-medium">
                      Two-Factor Authentication
                    </div>
                    <div className="text-xs text-slate-400 mt-0.5">
                      Secure your account with an authenticator app.
                    </div>
                  </div>
                  <button
                    onClick={handleToggle2FA}
                    aria-pressed={twoFactor}
                    aria-label="Toggle two-factor authentication"
                    className={`relative w-10 h-6 rounded-full transition-colors cursor-pointer ${
                      twoFactor ? "bg-cyan-500 shadow-[0_0_8px_rgba(6,182,212,0.4)]" : "bg-slate-800"
                    }`}
                  >
                    <span
                      className={`absolute top-0.5 w-5 h-5 rounded-full bg-white shadow transition-transform ${
                        twoFactor ? "left-4.5" : "left-0.5"
                      }`}
                    />
                  </button>
                </div>
              </Panel>
            </div>

              {/* Right column */}
            <div className="flex flex-col gap-6">
              <div className="w-full border border-slate-800/90 bg-[#0c1624]/60 backdrop-blur-md rounded-xl p-6 flex flex-col items-center text-center shadow-[0_4px_24px_rgba(0,0,0,0.3)]">
                <div className="w-14 h-14 rounded-full bg-cyan-500/10 border border-cyan-500/20 flex items-center justify-center mb-4 text-cyan-400 shadow-[0_0_15px_rgba(6,182,212,0.15)]">
                  <Satellite size={24} />
                </div>
                <div className="text-sm font-bold text-white tracking-tight">
                  NexSpace Platform
                </div>
                <div className="text-xs text-slate-400 font-mono mt-1 mb-5">
                  Version v2.4.1 (Build 8902)
                </div>
                <div className="w-full flex flex-col divide-y divide-slate-800/60 border-t border-slate-800/80">
                  <FooterLink label="Terms of Service" />
                  <FooterLink label="Privacy Policy" />
                  <FooterLink label="Data Processing Agreement" />
                </div>
              </div>

              <Panel icon={<ShieldCheck size={15} className="text-cyan-400" />} title="Scientific Credits">
                <p className="text-xs text-slate-400 leading-relaxed font-sans">
                  Neural models trained on open datasets provided by the European
                  Space Agency (ESA) Copernicus program and USGS Landsat
                  archives. Core computer vision architecture developed in
                  collaboration with NexSpace Research Labs.
                </p>
              </Panel>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function Panel({
  icon,
  title,
  children,
}: {
  icon: React.ReactNode;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div className="w-full border border-slate-800/90 bg-[#0c1624]/60 backdrop-blur-md rounded-xl p-4 sm:p-5 shadow-[0_4px_24px_rgba(0,0,0,0.3)]">
      <div className="flex items-center gap-2 mb-4">
        {icon}
        <h2 className="text-sm font-semibold text-white tracking-tight">{title}</h2>
      </div>
      {children}
    </div>
  );
}

function UsageStat({
  icon,
  label,
  value,
  max,
  percent,
  barColor,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  max: string;
  percent: number;
  barColor: string;
}) {
  return (
    <div>
      <div className="flex items-center gap-1.5 text-xs text-slate-400 mb-2">
        {icon}
        <span>{label}</span>
      </div>
      <div className="text-sm text-slate-200 font-medium mb-2 font-mono">
        {value}{" "}
        <span className="text-slate-500 font-normal">/ {max}</span>
      </div>
      <div className="h-1.5 w-full bg-slate-800 rounded-full overflow-hidden">
        <div
          className={`h-full ${barColor} rounded-full`}
          style={{ width: `${percent}%` }}
        />
      </div>
    </div>
  );
}

function FooterLink({ label }: { label: string }) {
  return (
    <a
      href="#"
      className="flex items-center justify-between py-2.5 text-xs text-slate-400 hover:text-cyan-300 transition-colors"
    >
      <span>{label}</span>
      <span className="text-slate-600 font-mono">›</span>
    </a>
  );
}