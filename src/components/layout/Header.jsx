import React from "react";
import { useLocation } from "react-router-dom";
import { Download } from "lucide-react";
import { CV_LINK } from "../../data/constants";
import { getActiveLayer } from "../../utils/layers";

/**
 * Page header with neural layer label and system status.
 */

const LAYER_LABELS = {
  0: { tag: "INPUT LAYER", title: "System Overview" },
  1: { tag: "HIDDEN LAYER 1", title: "Trained Models" },
  2: { tag: "HIDDEN LAYER 2", title: "Weights & Biases" },
  3: { tag: "HIDDEN LAYER 3", title: "Research Lab" },
  4: { tag: "OUTPUT LAYER", title: "About" },
};

export default function Header() {
  const location = useLocation();
  const activeLayer = getActiveLayer(location.pathname);
  const layerInfo = LAYER_LABELS[activeLayer] || LAYER_LABELS[0];

  // Don't show header on deep-dive / reader views
  const isDetailView =
    location.pathname.match(/^\/projects\/.+/);
  if (isDetailView) return null;

  return (
    <header className="flex justify-between items-end mb-8 md:mb-12 border-b border-[rgba(0,212,255,0.06)] pb-4 mt-2 md:mt-0">
      <div>
        <div className="flex items-center gap-2 mb-1.5">
          <span className="text-[10px] font-mono text-cyan-500/60 tracking-widest">
            {layerInfo.tag}
          </span>
          <span className="w-1.5 h-1.5 rounded-full bg-cyan-500/40 animate-neural-breathe" />
        </div>
        <h2
          className="text-2xl font-bold text-white"
          style={{ fontFamily: "var(--font-display)" }}
        >
          {layerInfo.title}
        </h2>
      </div>
      <div className="hidden md:flex items-center gap-4">
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
