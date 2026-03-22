import React from "react";

/**
 * Tech tag pill with subtle glow on hover.
 * Clickable to filter projects by tech.
 */
export default function TechPill({ label, active = false, onClick }) {
  return (
    <button
      onClick={onClick}
      className={`
        inline-flex items-center px-2.5 py-1 rounded-full text-xs font-mono
        transition-all duration-200 cursor-pointer
        ${
          active
            ? "bg-[rgba(0,212,255,0.15)] border border-[rgba(0,212,255,0.4)] text-cyan-200 shadow-[0_0_12px_rgba(0,212,255,0.15)]"
            : "bg-[rgba(255,255,255,0.04)] border border-[rgba(255,255,255,0.06)] text-slate-400 hover:border-[rgba(0,212,255,0.2)] hover:text-cyan-300 hover:bg-[rgba(0,212,255,0.05)]"
        }
      `}
    >
      {label}
    </button>
  );
}
