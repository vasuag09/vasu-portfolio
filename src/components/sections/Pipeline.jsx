import React, { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Server, RefreshCw, CheckCircle2 } from "lucide-react";
import { pipelineStages } from "../../data/pipeline";

export default function Pipeline() {
  const [pipelineStep, setPipelineStep] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setPipelineStep((prev) => (prev + 1) % pipelineStages.length);
    }, 1500);
    return () => clearInterval(interval);
  }, []);

  return (
    <div 
      className="bg-slate-900/30 border border-slate-800 p-6 rounded-lg mb-8 overflow-x-auto"
      role="region" 
      aria-label="MLOps Pipeline Architecture"
    >
      <div className="flex items-center gap-3 mb-6">
        <Server className="text-emerald-500" size={24} aria-hidden="true" />
        <h3 className="text-xl font-bold text-white">
          MLOps Pipeline Architecture
        </h3>
      </div>
      
      {/* Desktop: horizontal */}
      <div className="hidden md:flex items-center justify-between min-w-[600px]">
        {pipelineStages.map((stage, idx) => {
          const isActive = idx === pipelineStep;
          const isPast = idx < pipelineStep;
          const Icon = stage.icon;
          
          return (
            <div
              key={idx}
              className="flex items-center flex-1 last:flex-none relative"
            >
              <div className="flex flex-col items-center gap-2 relative z-10">
                <motion.div
                  animate={{
                    scale: isActive ? 1.1 : 1,
                    borderColor: isActive
                      ? "var(--color-emerald-500)"
                      : isPast
                        ? "rgba(16, 185, 129, 0.5)"
                        : "var(--color-slate-700)",
                    backgroundColor: isActive
                      ? "var(--color-emerald-900)"
                      : "var(--color-slate-950)",
                  }}
                  className={`w-12 h-12 rounded-full border-2 flex items-center justify-center transition-colors duration-300 ${
                    isActive ? "shadow-[0_0_15px_rgba(16,185,129,0.5)]" : ""
                  }`}
                >
                  <Icon
                    size={20}
                    className={
                      isActive || isPast ? "text-emerald-400" : "text-slate-500"
                    }
                  />
                </motion.div>
                <span
                  className={`text-xs font-mono font-bold ${
                    isActive ? "text-white" : "text-slate-500"
                  }`}
                >
                  {stage.label}
                </span>
                <div className="h-4">
                  <AnimatePresence>
                    {isActive && (
                      <motion.div
                        initial={{ scale: 0, opacity: 0 }}
                        animate={{ scale: 1, opacity: 1 }}
                        exit={{ scale: 0, opacity: 0 }}
                        className="flex items-center gap-1 text-[10px] text-emerald-400"
                      >
                        <RefreshCw size={10} className="animate-spin" /> PROC
                      </motion.div>
                    )}
                    {isPast && (
                      <motion.div
                        initial={{ scale: 0, opacity: 0 }}
                        animate={{ scale: 1, opacity: 1 }}
                        exit={{ scale: 0, opacity: 0 }}
                        className="text-emerald-500"
                      >
                        <CheckCircle2 size={12} />
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>
              </div>
              
              {idx < pipelineStages.length - 1 && (
                <div className="flex-1 h-[2px] bg-slate-800 mx-2 relative overflow-hidden">
                  <motion.div
                    initial={{ x: "-100%" }}
                    animate={
                      isActive
                        ? { x: "100%" }
                        : { x: isPast ? "100%" : "-100%" }
                    }
                    transition={
                      isActive
                        ? { duration: 1.5, repeat: Infinity, ease: "linear" }
                        : { duration: 0 }
                    }
                    className={`absolute inset-0 bg-gradient-to-r from-transparent via-emerald-500 to-transparent ${
                      isPast ? "opacity-100 w-full bg-emerald-900/50" : "w-1/2"
                    }`}
                  />
                </div>
              )}
            </div>
          );
        })}
      </div>
      
      {/* Mobile: vertical layout */}
      <div className="md:hidden flex flex-col gap-4">
        {pipelineStages.map((stage, idx) => {
          const isActive = idx === pipelineStep;
          const isPast = idx < pipelineStep;
          const Icon = stage.icon;
          
          return (
            <div key={idx} className="flex items-center gap-4">
              <motion.div
                animate={{
                  scale: isActive ? 1.1 : 1,
                  borderColor: isActive
                    ? "var(--color-emerald-500)"
                    : isPast
                      ? "rgba(16, 185, 129, 0.5)"
                      : "var(--color-slate-700)",
                  backgroundColor: isActive
                    ? "var(--color-emerald-900)"
                    : "var(--color-slate-950)",
                }}
                className={`w-10 h-10 shrink-0 rounded-full border-2 flex items-center justify-center transition-colors duration-300 ${
                  isActive ? "shadow-[0_0_15px_rgba(16,185,129,0.5)]" : ""
                }`}
              >
                <Icon
                  size={16}
                  className={
                    isActive || isPast ? "text-emerald-400" : "text-slate-500"
                  }
                />
              </motion.div>
              <div className="flex items-center gap-3 flex-1">
                <span
                  className={`text-xs font-mono font-bold ${
                    isActive ? "text-white" : "text-slate-500"
                  }`}
                >
                  {stage.label}
                </span>
                <AnimatePresence mode="wait">
                  {isActive && (
                    <motion.div
                      key="active"
                      initial={{ scale: 0, opacity: 0 }}
                      animate={{ scale: 1, opacity: 1 }}
                      exit={{ scale: 0, opacity: 0 }}
                      className="flex items-center gap-1 text-[10px] text-emerald-400"
                    >
                      <RefreshCw size={10} className="animate-spin" /> PROC
                    </motion.div>
                  )}
                  {isPast && (
                    <motion.div
                      key="past"
                      initial={{ scale: 0, opacity: 0 }}
                      animate={{ scale: 1, opacity: 1 }}
                      exit={{ scale: 0, opacity: 0 }}
                      className="text-emerald-500"
                    >
                      <CheckCircle2 size={12} />
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
