import React from "react";
import { useLocation } from "react-router-dom";
import { Download, Volume2, VolumeX, Zap, ZapOff } from "lucide-react";
import { CV_LINK } from "../../data/constants";
import { NAVIGATION_ITEMS, getActiveLayer } from "../../data/navigation";
import { useUI } from "../../hooks/useUI";

/**
 * Page header with neural layer label and download CV button.
 * Reduce-effects and sound toggles appear here on mobile (hidden on desktop where StatusBar takes over).
 */

export default function Header() {
  const location = useLocation();
  const { soundEnabled, setSoundEnabled, reducedEffects, setReducedEffects } = useUI();
  
  const activeIdx = getActiveLayer(location.pathname);
  const layerInfo = NAVIGATION_ITEMS[activeIdx] || NAVIGATION_ITEMS[0];

  // Don't show header on deep-dive / reader views
  const isDetailView =
    location.pathname.match(/^\/projects\/.+/) ||
    location.pathname.match(/^\/blog\/.+/);
  if (isDetailView) return null;

  return (
    <header className="flex justify-between items-end mb-8 md:mb-12 border-b border-[rgba(0,212,255,0.06)] pb-4 mt-2 md:mt-0">
      <div>
        <div className="flex items-center gap-2 mb-1.5">
          <span className="text-[10px] font-mono text-cyan-500/60 tracking-widest">
            {layerInfo.layerTag}
          </span>
          <span className="w-1.5 h-1.5 rounded-full bg-cyan-500/40 animate-neural-breathe" />
        </div>
        <h2
          className="text-2xl font-bold text-white relative flex items-center gap-3"
          style={{ fontFamily: "var(--font-display)" }}
        >
          {layerInfo.layerTitle}
          
          {/* Mobile-only toggles (desktop uses StatusBar) */}
          <div className="flex md:hidden items-center gap-2 ml-2">
            <button
              onClick={() => setReducedEffects((prev) => !prev)}
              className="text-slate-500 hover:text-cyan-400 transition-colors cursor-pointer p-1"
              aria-label={reducedEffects ? "Enable effects" : "Reduce effects"}
            >
              {reducedEffects ? <ZapOff size={14} /> : <Zap size={14} />}
            </button>
            <button
              onClick={() => setSoundEnabled((prev) => !prev)}
              className="text-slate-500 hover:text-cyan-400 transition-colors cursor-pointer p-1"
              aria-label={soundEnabled ? "Mute sounds" : "Enable sounds"}
            >
              {soundEnabled ? <Volume2 size={14} /> : <VolumeX size={14} />}
            </button>
          </div>
        </h2>
      </div>
      <div className="hidden md:flex items-center gap-3">
        <a
          href={CV_LINK}
          target="_blank"
          rel="noreferrer"
          className="flex items-center gap-2 text-xs font-mono text-cyan-400/80 border border-cyan-500/20 bg-cyan-500/5 px-3 py-2 rounded-lg hover:bg-cyan-500/10 hover:border-cyan-500/30 transition-all cursor-pointer"
        >
          <Download size={14} /> DOWNLOAD CV
        </a>
      </div>
    </header>
  );
}
