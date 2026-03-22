import React from "react";

/**
 * Neural-themed badge for tier (S/A/B) and status (LIVE/RESEARCH/CODE/BUILDING).
 */

const tierConfig = {
  S: {
    bg: "bg-[rgba(0,212,255,0.1)]",
    border: "border-[rgba(0,212,255,0.3)]",
    text: "text-cyan-300",
    dot: "bg-cyan-400",
    glow: "shadow-[0_0_8px_rgba(0,212,255,0.2)]",
  },
  A: {
    bg: "bg-[rgba(139,92,246,0.1)]",
    border: "border-[rgba(139,92,246,0.3)]",
    text: "text-purple-300",
    dot: "bg-purple-400",
    glow: "shadow-[0_0_8px_rgba(139,92,246,0.2)]",
  },
  B: {
    bg: "bg-[rgba(96,165,250,0.1)]",
    border: "border-[rgba(96,165,250,0.3)]",
    text: "text-blue-300",
    dot: "bg-blue-400",
    glow: "shadow-[0_0_8px_rgba(96,165,250,0.2)]",
  },
};

const statusConfig = {
  LIVE: {
    bg: "bg-[rgba(16,185,129,0.1)]",
    border: "border-[rgba(16,185,129,0.3)]",
    text: "text-emerald-300",
    dot: "bg-emerald-400",
    animate: true,
  },
  RESEARCH: {
    bg: "bg-[rgba(139,92,246,0.1)]",
    border: "border-[rgba(139,92,246,0.3)]",
    text: "text-purple-300",
    dot: "bg-purple-400",
    animate: false,
  },
  CODE: {
    bg: "bg-[rgba(0,212,255,0.1)]",
    border: "border-[rgba(0,212,255,0.3)]",
    text: "text-cyan-300",
    dot: "bg-cyan-400",
    animate: false,
  },
  BUILDING: {
    bg: "bg-[rgba(245,158,11,0.1)]",
    border: "border-[rgba(245,158,11,0.3)]",
    text: "text-amber-300",
    dot: "bg-amber-400",
    animate: true,
  },
};

export function TierBadge({ tier }) {
  const config = tierConfig[tier] || tierConfig.B;
  return (
    <span
      className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-mono font-medium border ${config.bg} ${config.border} ${config.text} ${config.glow}`}
    >
      <span className={`w-1.5 h-1.5 rounded-full ${config.dot}`} />
      TIER {tier}
    </span>
  );
}

export function StatusBadge({ status }) {
  const config = statusConfig[status] || statusConfig.CODE;
  return (
    <span
      className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-mono font-medium border ${config.bg} ${config.border} ${config.text}`}
    >
      <span
        className={`w-1.5 h-1.5 rounded-full ${config.dot} ${config.animate ? "animate-pulse" : ""}`}
      />
      {status}
    </span>
  );
}
