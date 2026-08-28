"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import { Satellite, ArrowRight, Menu, X, Sparkles } from "lucide-react";

export default function TopNavigationBar() {
  const pathname = usePathname();
  const [scrolled, setScrolled] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [hoveredPath, setHoveredPath] = useState<string | null>(null);

  const navItems = [
    { label: "Dashboard", href: "/dashboard" },
    { label: "NLP Query", href: "/query" },
    { label: "Temporal Compare", href: "/comparison" },
    { label: "Data Ingest", href: "/upload" },
    { label: "Intel Reports", href: "/reports" },
  ];

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 20);
    };
    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  return (
    <header
      className={`fixed top-0 left-0 right-0 z-50 transition-all duration-200 ease-out ${
        scrolled
          ? "bg-[#09131f]/90 backdrop-blur-md border-b border-cyan-500/20 py-3 px-4 sm:px-8 shadow-[0_4px_30px_rgba(0,0,0,0.6)]"
          : "bg-transparent border-b border-transparent py-4 sm:py-5 px-4 sm:px-8"
      }`}
    >
      <div className="max-w-7xl mx-auto flex items-center justify-between">
        {/* Brand Logo */}
        <Link href="/" className="flex items-center gap-2.5 group select-none">
          <motion.div
            whileHover={{ rotate: 15, scale: 1.05 }}
            transition={{ duration: 0.18, ease: "easeOut" }}
            className="p-1.5 rounded-lg bg-cyan-500/10 border border-cyan-500/25 text-cyan-400 group-hover:border-cyan-400/50 group-hover:bg-cyan-500/20 transition-all duration-180 shadow-[0_0_12px_rgba(6,182,212,0.25)]"
          >
            <Satellite size={18} />
          </motion.div>

          <div className="flex flex-col">
            <div className="flex items-center gap-1.5">
              <span className="text-white font-bold text-base sm:text-lg tracking-tight group-hover:text-cyan-300 transition-colors duration-180">
                NexSpace
              </span>
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse shadow-[0_0_8px_rgba(52,211,153,0.8)]" />
            </div>
            <span className="text-[9px] font-mono tracking-wider text-cyan-400/80 uppercase hidden sm:block">
              NEXSPACE · ORBITAL INTELLIGENCE
            </span>
          </div>
        </Link>

        {/* Desktop Navigation Links */}
        <nav
          className="hidden lg:flex items-center gap-1 bg-[#09131f]/80 backdrop-blur-md border border-slate-800/80 rounded-xl px-2 py-1 shadow-inner"
          onMouseLeave={() => setHoveredPath(null)}
        >
          {navItems.map((item) => {
            const isActive = pathname === item.href;
            const isHovered = hoveredPath === item.href;
            const isHighlighted = hoveredPath ? isHovered : isActive;

            return (
              <Link
                key={item.label}
                href={item.href}
                onMouseEnter={() => setHoveredPath(item.href)}
                className={`relative px-3.5 py-1.5 text-xs sm:text-sm font-medium transition-all duration-180 hover:-translate-y-[1px] ${
                  isHighlighted
                    ? "text-cyan-300 font-semibold"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                <span className="relative z-10 flex items-center gap-1.5">
                  {item.label}
                </span>

                {/* Animated active/hover glassmorphism pill */}
                {isHighlighted && (
                  <motion.div
                    layoutId="navbar-glass-pill"
                    initial={{ opacity: 0, scale: 0.96 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.96 }}
                    className="absolute inset-0 bg-[#0c1827]/90 bg-gradient-to-r from-cyan-500/15 via-sky-500/10 to-emerald-500/15 border border-cyan-500/30 rounded-lg shadow-[0_0_14px_rgba(6,182,212,0.18)] backdrop-blur-md"
                    transition={{
                      duration: 0.18,
                      ease: "easeOut",
                    }}
                  />
                )}
              </Link>
            );
          })}
        </nav>

        {/* Right CTA Button & Mobile Toggle */}
        <div className="flex items-center gap-3">
          <Link
            href="/dashboard"
            className="group relative inline-flex items-center gap-1.5 bg-gradient-to-r from-cyan-500 to-sky-400 hover:from-cyan-400 hover:to-sky-300 text-[#071320] text-xs font-bold px-4 py-2 rounded-lg transition-all duration-180 shadow-[0_0_15px_rgba(6,182,212,0.35)] hover:shadow-[0_0_22px_rgba(6,182,212,0.6)] hover:scale-[1.02] active:scale-[0.98] cursor-pointer overflow-hidden"
          >
            <Sparkles size={13} className="text-[#071320]" />
            <span>Open Console</span>
            <ArrowRight
              size={13}
              className="group-hover:translate-x-0.5 transition-transform duration-180"
            />
          </Link>

          {/* Mobile Menu Hamburger Toggle */}
          <button
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            className="lg:hidden p-2 rounded-lg bg-slate-900/80 border border-slate-800 text-slate-300 hover:text-white hover:border-cyan-500/40 transition-colors cursor-pointer"
            aria-label="Toggle navigation menu"
          >
            {mobileMenuOpen ? <X size={20} /> : <Menu size={20} />}
          </button>
        </div>
      </div>

      {/* Mobile Animated Dropdown Drawer */}
      <AnimatePresence>
        {mobileMenuOpen && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.2, ease: "easeOut" }}
            className="lg:hidden overflow-hidden border-b border-cyan-500/20 bg-[#09131f]/95 backdrop-blur-xl px-6 py-5 mt-3 rounded-2xl shadow-2xl mx-2 border border-slate-800/80"
          >
            <nav className="flex flex-col gap-2">
              {navItems.map((item) => {
                const isActive = pathname === item.href;
                return (
                  <Link
                    key={item.label}
                    href={item.href}
                    onClick={() => setMobileMenuOpen(false)}
                    className={`flex items-center justify-between px-3.5 py-2.5 rounded-lg text-sm font-medium transition-all duration-180 ${
                      isActive
                        ? "bg-cyan-500/15 text-cyan-300 border border-cyan-500/40 font-semibold"
                        : "text-slate-300 hover:bg-white/5 hover:text-white"
                    }`}
                  >
                    <span>{item.label}</span>
                    {isActive && (
                      <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 shadow-[0_0_6px_rgba(6,182,212,0.8)]" />
                    )}
                  </Link>
                );
              })}

              <div className="pt-3 border-t border-slate-800/80 mt-1">
                <Link
                  href="/dashboard"
                  onClick={() => setMobileMenuOpen(false)}
                  className="flex items-center justify-center gap-2 w-full bg-gradient-to-r from-cyan-500 to-sky-400 text-[#071320] text-xs font-bold py-2.5 rounded-lg shadow-[0_0_15px_rgba(6,182,212,0.4)]"
                >
                  <span>Open Console</span>
                  <ArrowRight size={13} />
                </Link>
              </div>
            </nav>
          </motion.div>
        )}
      </AnimatePresence>
    </header>
  );
}
