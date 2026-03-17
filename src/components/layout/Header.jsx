import React from "react";
import { useLocation } from "react-router-dom";
import { Download } from "lucide-react";
import { profile } from "../../data/profile";
import { getActiveItem } from "../../data/navigation";
import { CV_LINK, SYSTEM_CONFIG } from "../../data/constants";

export default function Header() {
  const location = useLocation();
  const activeItem = getActiveItem(location.pathname);
  const Icon = activeItem.icon;

  return (
    <header className="flex justify-between items-end mb-6 md:mb-12 border-b border-slate-800 pb-4 mt-2 md:mt-0">
      <div>
        <h2 className="text-2xl font-bold text-white mb-1 flex items-center gap-3">
          <Icon className="text-emerald-500" />
          {activeItem.label.toUpperCase()}
        </h2>
        <div className="flex items-center gap-4 text-[10px] font-mono">
          <div className="flex items-center gap-1.5 text-emerald-500">
            <span className="w-1.5 h-1.5 bg-emerald-500 rounded-full animate-pulse shadow-[0_0_8px_rgba(16,185,129,0.8)]"></span>
            SYSTEM {SYSTEM_CONFIG.status}
          </div>
          <div className="hidden sm:flex items-center gap-1 text-slate-500">
            <span className="opacity-50">[</span>
            <span className="text-emerald-500/60">LATENCY: {SYSTEM_CONFIG.latency}</span>
            <span className="opacity-50">]</span>
          </div>
          <div className="hidden sm:flex items-center gap-1 text-slate-500">
            <span className="opacity-50">[</span>
            <span className="text-emerald-500/60">NODE_V: {SYSTEM_CONFIG.version}</span>
            <span className="opacity-50">]</span>
          </div>
        </div>
      </div>
      <div className="flex items-end gap-6">
        <div className="hidden md:block text-right">
          <a
            href={CV_LINK}
            target="_blank"
            rel="noreferrer"
            className="flex items-center gap-2 text-xs font-mono text-emerald-500 border border-emerald-500/30 bg-emerald-500/5 px-3 py-2 rounded hover:bg-emerald-500/10 transition-all cursor-pointer"
          >
            <Download size={14} /> DOWNLOAD CV
          </a>
        </div>
        <div className="hidden md:block text-right">
          <div className="text-xs text-slate-500 font-mono">
            CURRENT LOCATION
          </div>
          <div className="text-sm font-mono">{profile.location}</div>
        </div>
      </div>
    </header>
  );
}
