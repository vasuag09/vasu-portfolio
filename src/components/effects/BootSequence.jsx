import React from "react";
import { motion } from "framer-motion";
import { Brain } from "lucide-react";

export default function BootSequence({ bootSequence, isBooted, skipBoot }) {
  return (
    <div className="bg-slate-900/50 border border-slate-800 p-8 rounded-lg relative overflow-hidden group">
      <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
        <Brain size={120} />
      </div>
      <div className="font-mono text-emerald-500 text-sm mb-4">
        {bootSequence.map((log, i) => (
          <div key={i} className="opacity-70">
            <span className="mr-2 text-slate-600">{">"}</span>
            {log}
          </div>
        ))}
        {isBooted && <span className="animate-pulse">_</span>}
      </div>
      {!isBooted && (
        <button
          onClick={skipBoot}
          className="absolute top-4 right-4 text-xs font-mono text-slate-500 hover:text-emerald-400 border border-slate-700 hover:border-emerald-500/50 px-3 py-1 rounded transition-all cursor-pointer z-10"
        >
          SKIP
        </button>
      )}
      <motion.h1
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2, duration: 0.5 }}
        className="text-3xl md:text-5xl font-bold text-white mb-6 max-w-2xl leading-tight"
      >
        Build fast. <br />
        <span className="text-slate-500">Ship models.</span> <br />
        Solve problems.
      </motion.h1>
    </div>
  );
}
