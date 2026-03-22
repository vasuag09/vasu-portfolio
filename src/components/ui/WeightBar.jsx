import React from "react";
import { motion } from "framer-motion";

/**
 * Skill proficiency bar styled as a neural connection weight.
 * `weight` is 0-100 representing proficiency percentage.
 */
export default function WeightBar({ label, weight = 75, delay = 0 }) {
  return (
    <div className="flex items-center gap-3 group">
      <div className="w-2 h-2 rounded-full bg-cyan-500/60 group-hover:bg-cyan-400 transition-colors shrink-0 animate-neural-breathe" />
      <span className="text-sm text-slate-300 font-mono w-40 shrink-0 group-hover:text-cyan-200 transition-colors">
        {label}
      </span>
      <div className="flex-1 h-1.5 bg-[rgba(255,255,255,0.04)] rounded-full overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          whileInView={{ width: `${weight}%` }}
          viewport={{ once: true, margin: "-50px" }}
          transition={{ duration: 0.8, delay, ease: "easeOut" }}
          className="h-full rounded-full"
          style={{
            background: `linear-gradient(90deg, var(--accent-cyan), var(--accent-purple))`,
          }}
        />
      </div>
      <span className="text-xs text-slate-500 font-mono w-10 text-right">{weight}%</span>
    </div>
  );
}
