import React, { Suspense, lazy } from "react";
import { motion } from "framer-motion";
import { ArrowUpRight, Play, ExternalLink } from "lucide-react";
import { TierBadge, StatusBadge } from "../ui/NeuralBadge";
import TechPill from "../ui/TechPill";
import TiltCard from "../effects/TiltCard";
import TextScramble from "../effects/TextScramble";

const FloatingGeometry = lazy(() => import("../canvas/FloatingGeometry"));

/**
 * Bento-style project card with 3D floating geometry.
 * S-tier projects get larger cards with full geometry.
 * A-tier gets medium, B-tier gets compact.
 */
export default function ProjectCard({
  project,
  index,
  filterTech,
  onFilterTech,
  onExpand,
  onPreview,
}) {
  const isLarge = project.tier === "S";

  return (
    <TiltCard maxTilt={isLarge ? 5 : 7} className={isLarge ? "md:col-span-2 md:row-span-2" : ""}>
      <motion.div
        initial={{ opacity: 0, y: 40, rotateX: 4 }}
        animate={{ opacity: 1, y: 0, rotateX: 0 }}
        transition={{ duration: 0.6, delay: index * 0.08, ease: [0.23, 1, 0.32, 1] }}
        className="glass-card p-0 overflow-hidden group cursor-pointer h-full flex flex-col"
        onClick={() => onExpand(project)}
      >
        {/* Top section: 3D geometry + tier info */}
        <div className="relative flex items-center justify-between p-5 pb-0">
          <div className="flex-1 min-w-0">
            {/* Badges */}
            <div className="flex items-center gap-2 mb-3 flex-wrap">
              <TierBadge tier={project.tier} />
              <StatusBadge status={project.status} />
              {project.gif && (
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onPreview(project);
                  }}
                  className="inline-flex items-center gap-1 px-2 py-1 rounded-full text-[10px] font-mono text-slate-400 border border-slate-700/50 hover:border-cyan-500/30 hover:text-cyan-300 transition-colors cursor-pointer"
                  aria-label="Preview demo"
                >
                  <Play size={8} /> Demo
                </button>
              )}
            </div>

            {/* Title */}
            <h3
              className="text-lg font-bold text-white group-hover:text-cyan-200 transition-colors mb-2 leading-tight"
              style={{ fontFamily: "var(--font-display)" }}
            >
              {project.title}
            </h3>
          </div>

          {/* 3D floating shape */}
          <div className="shrink-0 ml-3 opacity-70 group-hover:opacity-100 transition-opacity">
            <Suspense fallback={<div className="w-16 h-16" />}>
              <FloatingGeometry tier={project.tier} size={isLarge ? 96 : 64} />
            </Suspense>
          </div>
        </div>

        {/* Description */}
        <div className="px-5 pt-2 flex-1">
          <p className="text-sm text-slate-400 leading-relaxed line-clamp-3"
             style={{ fontFamily: "var(--font-sans)" }}>
            {project.description}
          </p>
        </div>

        {/* Tech Tags */}
        <div className="px-5 pt-3">
          <div className="flex flex-wrap gap-1.5">
            {project.tech.slice(0, isLarge ? 6 : 4).map((t) => (
              <TechPill
                key={t}
                label={t}
                active={filterTech === t}
                onClick={(e) => {
                  e.stopPropagation();
                  onFilterTech(filterTech === t ? null : t);
                }}
              />
            ))}
            {project.tech.length > (isLarge ? 6 : 4) && (
              <span className="text-[10px] text-slate-600 font-mono self-center">
                +{project.tech.length - (isLarge ? 6 : 4)}
              </span>
            )}
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between mt-auto p-5 pt-4 border-t border-[rgba(0,212,255,0.04)] mt-3">
          {project.details?.metrics && (
            <span className="text-[10px] font-mono text-slate-500 truncate max-w-[65%]">
              {project.details.metrics.substring(0, 50)}
            </span>
          )}
          <div className="flex items-center gap-1 shrink-0">
            <TextScramble
              text="EXPLORE"
              className="text-[10px] font-mono text-cyan-500/70 group-hover:text-cyan-400 transition-colors tracking-wider cursor-pointer"
              duration={300}
            />
            <ArrowUpRight size={12} className="text-cyan-500/70 group-hover:text-cyan-400 transition-colors" />
          </div>
        </div>

        {/* Hover gradient border effect */}
        <div className="absolute inset-0 rounded-xl opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none"
          style={{
            background: "linear-gradient(135deg, rgba(0,240,255,0.06), transparent 50%, rgba(168,85,247,0.06))",
          }}
        />
      </motion.div>
    </TiltCard>
  );
}
