import React from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { getActiveLayer } from "../../utils/layers";
import MagneticElement from "../effects/MagneticElement";

/**
 * Neural layer navigation — vertical chain of connected nodes.
 * Each node represents a section (layer) in the neural network.
 */

const LAYERS = [
  { path: "/", label: "Input", shortLabel: "IN", section: "Overview" },
  { path: "/projects", label: "Hidden 1", shortLabel: "H1", section: "Projects" },
  { path: "/skills", label: "Hidden 2", shortLabel: "H2", section: "Skills" },
  { path: "/research", label: "Hidden 3", shortLabel: "H3", section: "Research" },
  { path: "/about", label: "Output", shortLabel: "OUT", section: "About" },
];

export default function LayerNav() {
  const location = useLocation();
  const navigate = useNavigate();
  const activeIdx = getActiveLayer(location.pathname);

  return (
    <nav
      className="fixed left-0 top-0 h-full z-30 hidden md:flex flex-col items-center justify-center w-20 gap-0"
      aria-label="Network layers"
    >
      {LAYERS.map((layer, idx) => {
        const isActive = activeIdx === idx;

        return (
          <React.Fragment key={layer.path}>
            {/* Connection line above (except first) */}
            {idx > 0 && (
              <div
                className={`w-px h-8 transition-all duration-500 ${
                  (activeIdx >= idx - 1 && activeIdx <= idx)
                    ? "bg-gradient-to-b from-cyan-500/40 to-purple-500/40"
                    : "bg-slate-700/30"
                }`}
              />
            )}

            {/* Node */}
            <MagneticElement strength={0.35}>
            <button
              onClick={() => navigate(layer.path)}
              className="relative group cursor-pointer flex items-center justify-center"
              aria-label={`Navigate to ${layer.section}`}
              aria-current={isActive ? "page" : undefined}
            >
              {/* Glow ring for active */}
              {isActive && (
                <motion.div
                  layoutId="nav-glow"
                  className="absolute w-10 h-10 rounded-full border border-cyan-500/30 animate-neural-pulse"
                  transition={{ type: "spring", stiffness: 300, damping: 25 }}
                />
              )}

              {/* Node circle */}
              <div
                className={`w-4 h-4 rounded-full transition-all duration-300 relative z-10 ${
                  isActive
                    ? "bg-cyan-400 shadow-[0_0_12px_rgba(0,212,255,0.5)]"
                    : "bg-slate-600 group-hover:bg-cyan-500/50 group-hover:shadow-[0_0_8px_rgba(0,212,255,0.2)]"
                }`}
              >
                {/* Inner dot */}
                <div
                  className={`absolute inset-1 rounded-full ${
                    isActive ? "bg-white/80" : "bg-slate-400/50 group-hover:bg-cyan-300/50"
                  }`}
                />
              </div>

              {/* Label tooltip */}
              <div className="absolute left-full ml-4 pointer-events-none opacity-0 group-hover:opacity-100 transition-opacity duration-200 whitespace-nowrap">
                <div className="bg-slate-900/90 backdrop-blur border border-slate-700/50 rounded-lg px-3 py-1.5 text-xs font-mono">
                  <span className="text-cyan-400">{layer.shortLabel}</span>
                  <span className="text-slate-500 mx-1.5">·</span>
                  <span className="text-slate-300">{layer.section}</span>
                </div>
              </div>
            </button>
            </MagneticElement>
          </React.Fragment>
        );
      })}
    </nav>
  );
}
