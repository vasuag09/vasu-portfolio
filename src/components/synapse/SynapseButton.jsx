import React from "react";
import { motion } from "framer-motion";
import MagneticElement from "../effects/MagneticElement";

/**
 * Floating neural node that opens the Synapse AI chat.
 * Pulses gently to indicate it's "alive."
 */
export default function SynapseButton({ onClick }) {
  return (
    <MagneticElement strength={0.3} className="fixed bottom-6 right-6 md:bottom-8 md:right-8 z-40">
    <motion.button
      initial={{ opacity: 0, scale: 0.8 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ delay: 2.5 }}
      onClick={onClick}
      className="cursor-pointer group"
      aria-label="Open Synapse AI assistant"
    >
      {/* Outer glow ring */}
      <div className="absolute inset-0 w-14 h-14 rounded-full bg-cyan-500/10 animate-neural-pulse" />

      {/* Main button */}
      <div className="relative w-14 h-14 rounded-full bg-[rgba(12,16,32,0.9)] border border-cyan-500/30 flex items-center justify-center backdrop-blur-lg shadow-[0_0_20px_rgba(0,212,255,0.15)] group-hover:border-cyan-500/50 group-hover:shadow-[0_0_30px_rgba(0,212,255,0.25)] transition-all">
        {/* Neural node icon */}
        <div className="w-5 h-5 rounded-full bg-gradient-to-br from-cyan-400 to-purple-500 shadow-[0_0_10px_rgba(0,212,255,0.4)]" />

        {/* Connection lines */}
        <div className="absolute w-1.5 h-1.5 rounded-full bg-cyan-400/40 -top-0.5 left-1/2 -translate-x-1/2" />
        <div className="absolute w-1.5 h-1.5 rounded-full bg-purple-400/40 -bottom-0.5 left-1/2 -translate-x-1/2" />
        <div className="absolute w-1.5 h-1.5 rounded-full bg-cyan-400/30 top-1/2 -left-0.5 -translate-y-1/2" />
      </div>

      {/* Tooltip */}
      <div className="absolute bottom-full right-0 mb-3 pointer-events-none opacity-0 group-hover:opacity-100 transition-opacity">
        <div className="bg-slate-900/95 backdrop-blur border border-slate-700/50 rounded-lg px-3 py-1.5 text-xs font-mono text-cyan-400 whitespace-nowrap">
          Synapse AI <span className="text-slate-600">· Cmd+K</span>
        </div>
      </div>
    </motion.button>
    </MagneticElement>
  );
}
