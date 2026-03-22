import React, { useState, Suspense } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ChevronDown, ChevronUp, Loader2, AlertTriangle } from "lucide-react";
import { ErrorBoundary } from "../ui/ErrorBoundary";

/**
 * Shared wrapper for all Research Lab demos.
 * Provides consistent styling, loading states, error boundaries,
 * domain badge, and expandable "About this demo" section.
 */

const DOMAIN_COLORS = {
  Vision: "from-cyan-500 to-blue-500",
  NLP: "from-emerald-500 to-teal-500",
  Audio: "from-violet-500 to-purple-500",
  Generative: "from-rose-500 to-pink-500",
  Interactive: "from-amber-500 to-orange-500",
};

function DemoFallback() {
  return (
    <div className="flex items-center justify-center py-12">
      <div className="flex items-center gap-3 text-cyan-500/60 font-mono text-xs">
        <Loader2 size={14} className="animate-spin" />
        Loading demo...
      </div>
    </div>
  );
}

function DemoError({ error }) {
  return (
    <div className="flex items-center gap-3 py-8 px-4 text-red-400/80">
      <AlertTriangle size={16} />
      <div>
        <p className="text-sm font-mono">Demo failed to load</p>
        <p className="text-xs text-slate-500 mt-1">{error?.message || "Unknown error"}</p>
      </div>
    </div>
  );
}

export default function DemoCard({
  title,
  subtitle,
  domain,
  modelSize,
  about,
  children,
  defaultExpanded = false,
}) {
  const [expanded, setExpanded] = useState(defaultExpanded);
  const [showAbout, setShowAbout] = useState(false);
  const gradientClass = DOMAIN_COLORS[domain] || DOMAIN_COLORS.Interactive;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="glass-card-static overflow-hidden"
    >
      {/* Header — always visible */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full p-5 flex items-center gap-4 text-left cursor-pointer group hover:bg-[rgba(0,212,255,0.02)] transition-colors"
      >
        {/* Domain badge */}
        <div className={`w-2.5 h-2.5 rounded-full bg-gradient-to-br ${gradientClass} shrink-0`} />

        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <h3
              className="text-sm font-semibold text-white truncate"
              style={{ fontFamily: "var(--font-display)" }}
            >
              {title}
            </h3>
            <span className={`text-[9px] font-mono px-1.5 py-0.5 rounded bg-gradient-to-r ${gradientClass} text-white/90`}>
              {domain}
            </span>
          </div>
          <p className="text-[10px] font-mono text-slate-500 mt-0.5">
            {subtitle}
          </p>
        </div>

        {modelSize && (
          <span className="text-[9px] font-mono text-slate-600 shrink-0">
            ~{modelSize}
          </span>
        )}

        <div className="text-slate-600 group-hover:text-cyan-400 transition-colors shrink-0">
          {expanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
        </div>
      </button>

      {/* Expandable content */}
      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.3 }}
            className="overflow-hidden"
          >
            <div className="px-5 pb-5 space-y-4">
              {/* About toggle */}
              {about && (
                <button
                  onClick={() => setShowAbout(!showAbout)}
                  className="text-[10px] font-mono text-cyan-500/50 hover:text-cyan-400 transition-colors cursor-pointer flex items-center gap-1"
                >
                  {showAbout ? "Hide" : "About this demo"}
                  {showAbout ? <ChevronUp size={10} /> : <ChevronDown size={10} />}
                </button>
              )}

              <AnimatePresence>
                {showAbout && about && (
                  <motion.p
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: "auto", opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    className="text-xs text-slate-500 leading-relaxed border-l-2 border-cyan-500/10 pl-3"
                  >
                    {about}
                  </motion.p>
                )}
              </AnimatePresence>

              {/* Demo content with error boundary */}
              <ErrorBoundary fallback={<DemoError />}>
                <Suspense fallback={<DemoFallback />}>
                  {children}
                </Suspense>
              </ErrorBoundary>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}
