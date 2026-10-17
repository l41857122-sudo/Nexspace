"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Menu, X } from "lucide-react";
import NexSpaceLogo from "./landing/NexSpaceLogo";
import { LANDING_SECTIONS } from "../lib/landing/scrollTimeline";

export default function TopNavigationBar() {
  const pathname = usePathname();
  const [scrolled, setScrolled] = useState(false);
  const [activeSectionId, setActiveSectionId] = useState("hero");
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const navItems = [
    { label: "Home", href: "/#hero", id: "hero" },
    { label: "Platform", href: "/#investigation", id: "investigation" },
    { label: "Intelligence", href: "/#pixels", id: "pixels" },
    { label: "About", href: "/#about", id: "about" },
  ];

  useEffect(() => {
    const handleScroll = () => {
      const scrollY = window.scrollY;
      setScrolled(scrollY > 40);

      // Check if About section is in active view
      const aboutEl = document.getElementById("about");
      if (aboutEl) {
        const rect = aboutEl.getBoundingClientRect();
        if (rect.top <= 300 && rect.bottom >= 100) {
          setActiveSectionId("about");
          return;
        }
      }

      // Determine active section from scroll position
      const scrollHeight = document.documentElement.scrollHeight - window.innerHeight;
      if (scrollHeight > 0) {
        const progress = Math.min(1, Math.max(0, scrollY / scrollHeight));
        if (progress < 0.25) {
          setActiveSectionId("hero");
        } else if (progress < 0.50) {
          setActiveSectionId("investigation");
        } else if (progress < 0.75) {
          setActiveSectionId("pixels");
        }
      }
    };

    window.addEventListener("scroll", handleScroll, { passive: true });
    handleScroll();
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  const handleNavClick = (href: string) => {
    setMobileMenuOpen(false);
    if (href.startsWith("/#")) {
      const id = href.replace("/#", "");
      if (id === "hero") {
        if (typeof window !== "undefined" && (window as any).__lenis) {
          (window as any).__lenis.scrollTo(0);
        } else {
          window.scrollTo({ top: 0, behavior: "smooth" });
        }
        return;
      }
      const el = document.getElementById(id);
      if (el) {
        if (typeof window !== "undefined" && (window as any).__lenis) {
          (window as any).__lenis.scrollTo(el);
        } else {
          el.scrollIntoView({ behavior: "smooth" });
        }
      }
    }
  };

  return (
    <header
      className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 select-none ${
        scrolled
          ? "bg-[#020712]/92 backdrop-blur-xl border-b border-cyan-500/20 py-3 px-6 sm:px-12 shadow-[0_4px_30px_rgba(0,0,0,0.85)]"
          : "bg-transparent border-b border-transparent py-5 sm:py-6 px-6 sm:px-12"
      }`}
    >
      <div className="max-w-[1440px] mx-auto flex items-center justify-between">
        {/* Left: NEXSPACE Logo matching reference storyboard */}
        <Link href="/" className="flex items-center gap-2 group cursor-pointer" data-cursor="interactive">
          <NexSpaceLogo size="sm" />
        </Link>

        {/* Center: Sleek Capsule Pill Navigation with Active Cyan Indicator & Center Underline Reveal */}
        <nav className="hidden md:flex items-center bg-[#071322]/85 border border-slate-800/80 rounded-full px-5 py-1.5 shadow-[0_0_20px_rgba(0,0,0,0.6)] backdrop-blur-md">
          <div className="flex items-center gap-6 lg:gap-7">
            {navItems.map((item) => {
              const isActive = activeSectionId === item.id;

              return (
                <Link
                  key={item.label}
                  href={item.href}
                  onClick={(e) => {
                    e.preventDefault();
                    handleNavClick(item.href);
                  }}
                  data-cursor="interactive"
                  className="relative py-1 text-xs lg:text-[13px] font-medium tracking-wide transition-all group flex items-center gap-1.5"
                >
                  {/* Active cyan dot */}
                  {isActive && (
                    <span className="w-1.5 h-1.5 rounded-full bg-[#00f0ff] shadow-[0_0_8px_#00f0ff]" />
                  )}

                  <span
                    className={`transition-colors duration-200 ${
                      isActive
                        ? "text-white font-semibold"
                        : "text-slate-400 group-hover:text-white"
                    }`}
                  >
                    {item.label}
                  </span>

                  {/* Underline grows from center on hover */}
                  <span
                    className={`absolute bottom-0 left-1/2 -translate-x-1/2 h-[2px] bg-[#00f0ff] rounded-full transition-all duration-300 pointer-events-none ${
                      isActive ? "w-full shadow-[0_0_8px_#00f0ff]" : "w-0 group-hover:w-full opacity-60"
                    }`}
                  />
                </Link>
              );
            })}
          </div>
        </nav>

        {/* Right: Log In + Cyan Pill Get Started (matching approved mockup) */}
        <div className="flex items-center gap-4 sm:gap-5">
          <Link
            href="/query"
            data-cursor="interactive"
            className="text-slate-300 hover:text-white text-xs lg:text-sm font-medium transition-colors cursor-pointer"
          >
            Log In
          </Link>

          <Link
            href="/query"
            data-cursor="interactive"
            className="inline-flex items-center justify-center bg-[#00f0ff] hover:bg-[#38f5ff] text-[#030914] text-xs lg:text-sm font-bold px-5 sm:px-6 py-2 rounded-full transition-all duration-200 shadow-[0_0_24px_rgba(0,240,255,0.65)] hover:shadow-[0_0_32px_rgba(0,240,255,0.85)] hover:scale-105 active:scale-95 cursor-pointer"
          >
            Get Started
          </Link>

          {/* Mobile menu toggle */}
          <button
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            className="md:hidden text-slate-300 hover:text-white p-1"
            aria-label="Toggle navigation menu"
          >
            {mobileMenuOpen ? <X size={20} /> : <Menu size={20} />}
          </button>
        </div>
      </div>

      {/* Mobile Drawer */}
      {mobileMenuOpen && (
        <div className="md:hidden bg-[#030914]/95 border-b border-cyan-500/20 px-6 py-4 mt-3 flex flex-col gap-3">
          {navItems.map((item) => (
            <button
              key={item.label}
              onClick={() => handleNavClick(item.href)}
              className="text-sm font-medium text-slate-300 hover:text-[#00f0ff] py-1.5 text-left flex items-center justify-between"
            >
              <span>{item.label}</span>
              {activeSectionId === item.id && (
                <span className="w-1.5 h-1.5 rounded-full bg-[#00f0ff] shadow-[0_0_8px_#00f0ff]" />
              )}
            </button>
          ))}
          <div className="pt-2 border-t border-slate-800 flex items-center justify-between">
            <Link href="/query" className="text-sm text-slate-300">
              Log In
            </Link>
            <Link
              href="/query"
              className="bg-[#00f0ff] text-black text-xs font-bold px-4 py-2 rounded-full"
            >
              Get Started
            </Link>
          </div>
        </div>
      )}
    </header>
  );
}
