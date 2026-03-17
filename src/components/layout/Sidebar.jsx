import React from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Terminal,
  Command,
  Monitor,
  Map,
  Github,
  Linkedin,
  Mail,
} from "lucide-react";
import { profile } from "../../data/profile";
import { NAVIGATION_ITEMS } from "../../data/navigation";
import { SOCIAL_LINKS } from "../../data/constants";
import { useUI } from "../../hooks/useUI";
import NavButton from "./NavButton";

export default function Sidebar() {
  const {
    isRetro,
    setIsRetro,
    setIsTerminalOpen,
    setTourStep,
    isMobileMenuOpen,
    setIsMobileMenuOpen,
  } = useUI();

  const closeMenu = () => setIsMobileMenuOpen(false);

  return (
    <>
      {/* Mobile Backdrop Overlay */}
      <AnimatePresence>
        {isMobileMenuOpen && (
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-slate-950/80 backdrop-blur-sm z-30 md:hidden"
            onClick={closeMenu}
            aria-hidden="true"
          />
        )}
      </AnimatePresence>

      <nav
        className={`
          fixed inset-y-0 left-0 transform ${isMobileMenuOpen ? "translate-x-0" : "-translate-x-full"}
          md:translate-x-0 transition-transform duration-300 ease-in-out
          w-64 bg-slate-950 md:bg-slate-900/30 border-r border-slate-800 flex flex-col justify-between z-40 md:z-10 h-full
        `}
        aria-label="Main navigation"
      >
        <div className="overflow-y-auto custom-scrollbar flex-1 pb-6">
          <div className="p-6 hidden md:block">
            <div className="w-12 h-12 bg-slate-800 rounded border border-slate-700 flex items-center justify-center mb-4">
              <Terminal className="text-emerald-500" />
            </div>
            <h1 className="font-bold text-lg tracking-wider">{profile.name}</h1>
            <p className="text-xs text-slate-500 font-mono mt-1">
              {profile.title}
            </p>
          </div>
          
          <div className="p-6 md:hidden">
            <div className="font-mono text-emerald-500 mb-2">{"// NAVIGATION"}</div>
          </div>

          <div className="flex flex-col gap-1 mt-0 md:mt-4">
            {NAVIGATION_ITEMS.map((item) => (
              <NavButton
                key={item.path}
                to={item.path}
                icon={item.icon}
                label={item.systemLabel}
                shortcut={item.shortcut}
                onClick={closeMenu}
              />
            ))}
          </div>
          <div className="flex flex-col gap-2 px-6 mt-6 pb-6">
            <button
              onClick={() => { setIsTerminalOpen(true); closeMenu(); }}
              aria-label="Open Terminal"
              className="w-full bg-slate-900 border border-slate-700 px-4 py-3 rounded text-xs font-mono text-slate-400 hover:text-emerald-500 hover:border-emerald-500/50 transition-all flex items-center justify-between group cursor-pointer focus:outline-none focus:ring-1 focus:ring-emerald-500/50"
            >
              <div className="flex items-center gap-2">
                <Command size={14} /> Terminal
              </div>
              <span className="bg-slate-800 px-1.5 py-0.5 rounded text-[10px] group-hover:text-white hidden lg:inline-block">
                ⌘K
              </span>
            </button>
            <button
              onClick={() => { setIsRetro((prev) => !prev); closeMenu(); }}
              aria-label="Toggle Retro Mode"
              className={`w-full border px-4 py-3 rounded text-xs font-mono transition-all flex items-center justify-center gap-2 cursor-pointer focus:outline-none focus:ring-1 focus:ring-emerald-500/50 ${
                isRetro
                  ? "bg-emerald-900/30 border-emerald-500 text-emerald-400"
                  : "bg-slate-900 border-slate-700 text-slate-400 hover:text-white"
              }`}
            >
              <Monitor size={14} />{" "}
              {isRetro ? "Retro Mode: ON" : "Retro Mode: OFF"}
            </button>
            <button
              onClick={() => { setTourStep(1); closeMenu(); }}
              aria-label="Start System Tour"
              className="w-full border border-slate-700 px-4 py-3 rounded text-xs font-mono text-slate-400 hover:text-emerald-500 hover:border-emerald-500/50 transition-all flex items-center justify-center gap-2 bg-slate-900 cursor-pointer focus:outline-none focus:ring-1 focus:ring-emerald-500/50"
            >
              <Map size={14} /> Start Tour
            </button>
          </div>
        </div>
        <div className="p-6 border-t border-slate-800 shrink-0">
          <div className="flex gap-4 justify-center md:justify-start">
            <a
              href={SOCIAL_LINKS.github}
              target="_blank"
              rel="noreferrer"
              aria-label="GitHub profile"
              className="text-slate-400 hover:text-white transition-colors"
            >
              <Github size={20} />
            </a>
            <a
              href={SOCIAL_LINKS.linkedin}
              target="_blank"
              rel="noreferrer"
              aria-label="LinkedIn profile"
              className="text-slate-400 hover:text-white transition-colors"
            >
              <Linkedin size={20} />
            </a>
            <a
              href={SOCIAL_LINKS.email}
              aria-label="Send email"
              className="text-slate-400 hover:text-white transition-colors"
            >
              <Mail size={20} />
            </a>
          </div>
        </div>
      </nav>
    </>
  );
}
