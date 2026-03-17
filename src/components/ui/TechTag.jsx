import React from "react";

export default function TechTag({ tech, isActive, onClick }) {
  return (
    <button
      onClick={onClick}
      className={`text-xs font-mono px-2 py-1 rounded transition-colors cursor-pointer ${
        isActive
          ? "bg-emerald-500 text-white"
          : "bg-slate-800 text-slate-300 hover:bg-slate-700"
      }`}
    >
      {tech}
    </button>
  );
}
