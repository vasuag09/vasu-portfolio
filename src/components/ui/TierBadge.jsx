import React from "react";
import { Sparkles } from "lucide-react";

const tierConfig = {
  S: {
    className:
      "bg-gradient-to-r from-emerald-500/20 to-yellow-500/20 text-yellow-300 border-yellow-500/50",
    glow: "shadow-[0_0_10px_rgba(234,179,8,0.2)]",
    icon: <Sparkles size={10} />,
    label: "TIER S",
  },
  A: {
    className: "bg-blue-500/10 text-blue-300 border-blue-500/50",
    glow: "",
    icon: null,
    label: "TIER A",
  },
  B: {
    className: "bg-slate-700/30 text-slate-400 border-slate-600",
    glow: "",
    icon: null,
    label: "TIER B",
  },
};

export default function TierBadge({ tier, size = "sm" }) {
  const config = tierConfig[tier];
  if (!config) return null;

  const sizeClasses = size === "lg" ? "text-xs px-2 py-1" : "text-[10px] px-2 py-0.5";

  return (
    <span
      className={`font-bold rounded border inline-flex items-center gap-1 ${sizeClasses} ${config.className} ${config.glow}`}
    >
      {config.icon} {config.label}
    </span>
  );
}
