"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import {
  Satellite,
  LayoutGrid,
  Search,
  GitCompare,
  Upload,
  FileText,
  Settings,
  Activity,
  FileCode2,
  Scan,
  Eye,
  Menu,
  X,
  Radio,
  ChevronRight,
} from "lucide-react";

const querySubmenu = [
  { label: "Execution Trace", icon: Activity, href: "/execution" },
  { label: "Execution Log", icon: FileCode2, href: "/execution-log" },
  { label: "Scan Results", icon: Scan, href: "/results" },
  { label: "Evidence Viewer", icon: Eye, href: "/evidence" },
];

export default function Sidebar() {
  const [mobileOpen, setMobileOpen] = useState(false);
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);
  const pathname = usePathname();
  const router = useRouter();

  const isQuerySubRoute = querySubmenu.some((item) => item.href === pathname);
  const isQueryActive = pathname === "/query";
  const [queryExpanded, setQueryExpanded] = useState(isQueryActive || isQuerySubRoute);

  // Poll backend health status
  useEffect(() => {
    const checkHealth = () => {
      fetch("/api/health")
        .then((res) => {
          if (res.ok) {
            setBackendOnline(true);
          } else {
            setBackendOnline(false);
          }
        })
        .catch(() => {
          setBackendOnline(false);
        });
    };
    checkHealth();
    const interval = setInterval(checkHealth, 15000);
    return () => clearInterval(interval);
  }, []);

  // Auto-expand Query submenu if user navigates to or refreshes on any sub-route or /query
  useEffect(() => {
    if (isQuerySubRoute || isQueryActive) {
      setQueryExpanded(true);
    }
  }, [pathname, isQuerySubRoute, isQueryActive]);

  // Close mobile drawer on route change
  useEffect(() => {
    setMobileOpen(false);
  }, [pathname]);

  // Lock body scroll when mobile menu is open
  useEffect(() => {
    if (mobileOpen) {
      document.body.style.overflow = "hidden";
    } else {
      document.body.style.overflow = "unset";
    }
    return () => {
      document.body.style.overflow = "unset";
    };
  }, [mobileOpen]);

  const handleQueryClick = (e: React.MouseEvent) => {
    e.preventDefault();
    setQueryExpanded((prev) => !prev);
    if (pathname !== "/query" && !isQuerySubRoute) {
      router.push("/query");
    }
  };

  const renderNavList = () => (
    <div className="flex-1 flex flex-col justify-between overflow-y-auto px-3 py-3 space-y-6 select-none">
      <div>
        <p className="px-2.5 text-[10px] uppercase font-mono tracking-widest text-slate-400 mb-2">
          Core Operations
        </p>

        <nav className="space-y-1">
          {/* 1. Dashboard */}
          <Link
            href="/dashboard"
            className={`group w-full flex items-center justify-between rounded-lg px-3 py-2 text-xs font-medium transition-all duration-180 ease-out border ${
              pathname === "/dashboard"
                ? "bg-gradient-to-r from-cyan-500 to-sky-400 text-[#071320] border-cyan-300 font-bold shadow-[0_0_16px_rgba(6,182,212,0.4)] scale-[1.01]"
                : "border-transparent text-slate-400 hover:bg-slate-800/60 hover:text-cyan-300 hover:border-cyan-500/20 hover:translate-x-0.5 active:scale-[0.99]"
            }`}
          >
            <div className="flex items-center gap-2.5">
              <LayoutGrid
                size={15}
                className={`transition-transform duration-180 group-hover:scale-105 ${
                  pathname === "/dashboard" ? "text-[#071320]" : "text-slate-400 group-hover:text-cyan-400"
                }`}
              />
              <span>Dashboard</span>
            </div>
            {pathname === "/dashboard" && <ChevronRight size={14} className="opacity-80" />}
          </Link>

          {/* 2. Query with Expandable Submenu */}
          <div>
            <button
              onClick={handleQueryClick}
              className={`group w-full flex items-center justify-between rounded-lg px-3 py-2 text-xs font-medium transition-all duration-180 ease-out border cursor-pointer ${
                isQueryActive
                  ? "bg-gradient-to-r from-cyan-500 to-sky-400 text-[#071320] border-cyan-300 font-bold shadow-[0_0_16px_rgba(6,182,212,0.4)] scale-[1.01]"
                  : isQuerySubRoute
                  ? "bg-cyan-500/15 text-cyan-300 border-cyan-500/35 font-semibold"
                  : "border-transparent text-slate-400 hover:bg-slate-800/60 hover:text-cyan-300 hover:border-cyan-500/20 hover:translate-x-0.5 active:scale-[0.99]"
              }`}
            >
              <div className="flex items-center gap-2.5">
                <Search
                  size={15}
                  className={`transition-transform duration-180 group-hover:scale-105 ${
                    isQueryActive ? "text-[#071320]" : "text-slate-400 group-hover:text-cyan-400"
                  }`}
                />
                <span>Query</span>
              </div>

              {/* Animated Chevron Indicator */}
              <div
                className={`transition-transform duration-180 ease-out ${
                  queryExpanded ? "rotate-90" : "rotate-0"
                }`}
              >
                <ChevronRight
                  size={14}
                  className={isQueryActive ? "text-[#071320]" : "text-slate-500 group-hover:text-cyan-300"}
                />
              </div>
            </button>

            {/* Expandable Submenu */}
            <AnimatePresence initial={false}>
              {queryExpanded && (
                <motion.div
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: "auto", opacity: 1 }}
                  exit={{ height: 0, opacity: 0 }}
                  transition={{ duration: 0.18, ease: "easeOut" }}
                  className="overflow-hidden"
                >
                  <div className="border-l border-slate-800/90 ml-4 pl-2 space-y-1 my-1">
                    {querySubmenu.map(({ label, icon: Icon, href }) => {
                      const active = pathname === href;
                      return (
                        <Link
                          key={href}
                          href={href}
                          className={`group w-full flex items-center justify-between rounded-lg px-2.5 py-1.5 text-[11px] font-medium transition-all duration-180 ease-out border ${
                            active
                              ? "bg-cyan-500/15 text-cyan-300 border-cyan-500/40 font-semibold shadow-[0_0_10px_rgba(6,182,212,0.2)]"
                              : "border-transparent text-slate-400 hover:bg-slate-800/50 hover:text-cyan-300 hover:translate-x-0.5"
                          }`}
                        >
                          <div className="flex items-center gap-2">
                            <Icon
                              size={13}
                              className={`transition-transform duration-180 group-hover:scale-105 ${
                                active ? "text-cyan-400" : "text-slate-500 group-hover:text-cyan-400"
                              }`}
                            />
                            <span>{label}</span>
                          </div>
                          {active && (
                            <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 shadow-[0_0_6px_rgba(6,182,212,0.8)]" />
                          )}
                        </Link>
                      );
                    })}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* 3. Comparison */}
          <Link
            href="/comparison"
            className={`group w-full flex items-center justify-between rounded-lg px-3 py-2 text-xs font-medium transition-all duration-180 ease-out border ${
              pathname === "/comparison"
                ? "bg-gradient-to-r from-cyan-500 to-sky-400 text-[#071320] border-cyan-300 font-bold shadow-[0_0_16px_rgba(6,182,212,0.4)] scale-[1.01]"
                : "border-transparent text-slate-400 hover:bg-slate-800/60 hover:text-cyan-300 hover:border-cyan-500/20 hover:translate-x-0.5 active:scale-[0.99]"
            }`}
          >
            <div className="flex items-center gap-2.5">
              <GitCompare
                size={15}
                className={`transition-transform duration-180 group-hover:scale-105 ${
                  pathname === "/comparison" ? "text-[#071320]" : "text-slate-400 group-hover:text-cyan-400"
                }`}
              />
              <span>Comparison</span>
            </div>
            {pathname === "/comparison" && <ChevronRight size={14} className="opacity-80" />}
          </Link>

          {/* 4. Upload Ingest */}
          <Link
            href="/upload"
            className={`group w-full flex items-center justify-between rounded-lg px-3 py-2 text-xs font-medium transition-all duration-180 ease-out border ${
              pathname === "/upload"
                ? "bg-gradient-to-r from-cyan-500 to-sky-400 text-[#071320] border-cyan-300 font-bold shadow-[0_0_16px_rgba(6,182,212,0.4)] scale-[1.01]"
                : "border-transparent text-slate-400 hover:bg-slate-800/60 hover:text-cyan-300 hover:border-cyan-500/20 hover:translate-x-0.5 active:scale-[0.99]"
            }`}
          >
            <div className="flex items-center gap-2.5">
              <Upload
                size={15}
                className={`transition-transform duration-180 group-hover:scale-105 ${
                  pathname === "/upload" ? "text-[#071320]" : "text-slate-400 group-hover:text-cyan-400"
                }`}
              />
              <span>Upload Ingest</span>
            </div>
            {pathname === "/upload" && <ChevronRight size={14} className="opacity-80" />}
          </Link>

          {/* 5. Reports */}
          <Link
            href="/reports"
            className={`group w-full flex items-center justify-between rounded-lg px-3 py-2 text-xs font-medium transition-all duration-180 ease-out border ${
              pathname === "/reports"
                ? "bg-gradient-to-r from-cyan-500 to-sky-400 text-[#071320] border-cyan-300 font-bold shadow-[0_0_16px_rgba(6,182,212,0.4)] scale-[1.01]"
                : "border-transparent text-slate-400 hover:bg-slate-800/60 hover:text-cyan-300 hover:border-cyan-500/20 hover:translate-x-0.5 active:scale-[0.99]"
            }`}
          >
            <div className="flex items-center gap-2.5">
              <FileText
                size={15}
                className={`transition-transform duration-180 group-hover:scale-105 ${
                  pathname === "/reports" ? "text-[#071320]" : "text-slate-400 group-hover:text-cyan-400"
                }`}
              />
              <span>Reports</span>
            </div>
            {pathname === "/reports" && <ChevronRight size={14} className="opacity-80" />}
          </Link>

          {/* 6. Settings */}
          <Link
            href="/settings"
            className={`group w-full flex items-center justify-between rounded-lg px-3 py-2 text-xs font-medium transition-all duration-180 ease-out border ${
              pathname === "/settings"
                ? "bg-gradient-to-r from-cyan-500 to-sky-400 text-[#071320] border-cyan-300 font-bold shadow-[0_0_16px_rgba(6,182,212,0.4)] scale-[1.01]"
                : "border-transparent text-slate-400 hover:bg-slate-800/60 hover:text-cyan-300 hover:border-cyan-500/20 hover:translate-x-0.5 active:scale-[0.99]"
            }`}
          >
            <div className="flex items-center gap-2.5">
              <Settings
                size={15}
                className={`transition-transform duration-180 group-hover:scale-105 ${
                  pathname === "/settings" ? "text-[#071320]" : "text-slate-400 group-hover:text-cyan-400"
                }`}
              />
              <span>Settings</span>
            </div>
            {pathname === "/settings" && <ChevronRight size={14} className="opacity-80" />}
          </Link>
        </nav>
      </div>

      {/* Clean Bottom Control Block */}
      <div className="pt-3.5 border-t border-slate-800/80 space-y-1.5 px-0.5">
        <Link
          href="/"
          className="flex items-center justify-between px-3 py-2 rounded-lg text-xs font-medium text-slate-400 hover:text-cyan-300 hover:bg-slate-800/50 transition-all duration-180 hover:translate-x-0.5 group"
        >
          <div className="flex items-center gap-2">
            <Radio size={14} className="text-cyan-400 group-hover:scale-110 transition-transform duration-180" />
            <span>Terminal Hub</span>
          </div>
          <span className="text-[10px] font-mono text-slate-500 group-hover:text-cyan-400">
            Home
          </span>
        </Link>

        <div className={`flex items-center justify-between px-3 py-2 rounded-lg border text-[11px] font-mono ${
          backendOnline === false
            ? "bg-rose-500/10 border-rose-500/25 text-rose-300"
            : backendOnline === true
            ? "bg-emerald-500/5 border-emerald-500/15 text-emerald-300"
            : "bg-slate-900 border-slate-800 text-slate-400"
        }`}>
          <div className="flex items-center gap-2">
            <span className={`w-1.5 h-1.5 rounded-full ${
              backendOnline === false
                ? "bg-rose-500 shadow-[0_0_6px_rgba(244,63,94,0.8)]"
                : backendOnline === true
                ? "bg-emerald-400 animate-pulse shadow-[0_0_6px_rgba(52,211,153,0.8)]"
                : "bg-amber-400 animate-ping"
            }`} />
            <span>AI Status</span>
          </div>
          <span className={`font-semibold ${
            backendOnline === false
              ? "text-rose-400"
              : backendOnline === true
              ? "text-emerald-400"
              : "text-slate-400"
          }`}>
            {backendOnline === false ? "AI Backend Offline" : backendOnline === true ? "Live ML Pipeline" : "Connecting..."}
          </span>
        </div>
      </div>
    </div>
  );

  return (
    <>
      {/* Mobile Top Header */}
      <div className="lg:hidden w-full bg-[#09131f]/95 backdrop-blur-md border-b border-slate-800/80 px-4 py-3 flex items-center justify-between sticky top-0 z-40">
        <Link href="/" className="flex items-center gap-2 group">
          <Satellite size={18} className="text-cyan-400 group-hover:rotate-45 transition-transform duration-200" />
          <span className="text-sm font-bold text-white tracking-tight group-hover:text-cyan-300 transition-colors">
            NexSpace
          </span>
          <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
            v2.4
          </span>
        </Link>

        <button
          onClick={() => setMobileOpen(!mobileOpen)}
          aria-label="Toggle navigation menu"
          className="p-1.5 rounded-lg bg-slate-900/80 border border-slate-800 text-slate-300 hover:text-white hover:border-cyan-500/40 transition-all duration-180 active:scale-95 cursor-pointer"
        >
          {mobileOpen ? <X size={20} /> : <Menu size={20} />}
        </button>
      </div>

      {/* Mobile Drawer with Framer Motion AnimatePresence */}
      <AnimatePresence>
        {mobileOpen && (
          <>
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.2 }}
              onClick={() => setMobileOpen(false)}
              className="lg:hidden fixed inset-0 z-40 bg-black/70 backdrop-blur-sm"
            />
            <motion.div
              initial={{ x: "-100%" }}
              animate={{ x: 0 }}
              exit={{ x: "-100%" }}
              transition={{ type: "spring", damping: 25, stiffness: 250 }}
              className="lg:hidden fixed inset-y-0 left-0 z-50 w-64 bg-[#09131f] border-r border-slate-800/90 flex flex-col shadow-2xl"
            >
              <div className="flex items-center justify-between px-5 py-4 border-b border-slate-800/80">
                <div className="flex items-center gap-2">
                  <Satellite size={20} className="text-cyan-400" />
                  <div>
                    <p className="text-sm font-bold text-white leading-tight">
                      NexSpace
                    </p>
                    <p className="text-[10px] tracking-widest text-slate-400 uppercase font-mono">
                      Orbital Intel
                    </p>
                  </div>
                </div>
                <button
                  onClick={() => setMobileOpen(false)}
                  aria-label="Close menu"
                  className="p-1 text-slate-400 hover:text-white transition-colors cursor-pointer"
                >
                  <X size={18} />
                </button>
              </div>

              {renderNavList()}
            </motion.div>
          </>
        )}
      </AnimatePresence>

      {/* Desktop Persistent Sidebar */}
      <aside className="hidden lg:flex w-56 shrink-0 bg-[#09131f] border-r border-slate-800/90 flex-col h-screen sticky top-0">
        <div className="flex items-center gap-2 px-4 py-4 border-b border-slate-800/80">
          <Satellite size={20} className="text-cyan-400" />
          <div>
            <p className="text-sm font-bold text-white leading-tight">
              NexSpace
            </p>
            <p className="text-[10px] tracking-widest text-slate-400 uppercase font-mono">
              Orbital Intel
            </p>
          </div>
        </div>

        {renderNavList()}
      </aside>
    </>
  );
}