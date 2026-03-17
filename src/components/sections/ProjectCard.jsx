import React from "react";
import { motion } from "framer-motion";
import { Briefcase, BookOpen, Eye, ExternalLink } from "lucide-react";
import TierBadge from "../ui/TierBadge";
import StatusBadge from "../ui/StatusBadge";
import TechTag from "../ui/TechTag";

export default function ProjectCard({
  project,
  filterTech,
  onFilterTech,
  onExpand,
  onPreview,
  index,
}) {
  const tierGlows = {
    S: "hover:shadow-[0_0_20px_rgba(16,185,129,0.15)] hover:border-emerald-500/50",
    A: "hover:shadow-[0_0_20px_rgba(59,130,246,0.15)] hover:border-blue-500/50",
    B: "hover:shadow-[0_0_20px_rgba(100,116,139,0.15)] hover:border-slate-500/50",
  };

  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.05, duration: 0.3 }}
      whileHover={{ y: -4 }}
      className={`group bg-slate-900/40 border border-slate-800/60 p-6 rounded-xl transition-all duration-300 hover:bg-slate-900/60 flex flex-col relative overflow-hidden ${tierGlows[project.tier] || tierGlows.B}`}
    >
      {/* Tier Badge */}
      <div className="absolute top-0 right-0 p-2 z-10">
        <TierBadge tier={project.tier} />
      </div>

      <div className="flex justify-between items-start mb-4 gap-4 mt-2">
        <div className="flex items-start gap-2 min-w-0">
          <Briefcase size={16} className="text-emerald-500 mt-1 shrink-0" />
          <h3 className="text-lg font-bold text-white group-hover:text-emerald-400 transition-colors leading-tight break-words">
            {project.title}
          </h3>
        </div>
        <StatusBadge status={project.status} />
      </div>

      <p className="text-slate-400 text-sm leading-relaxed mb-6 flex-grow">
        {project.description}
      </p>

      <div className="flex flex-wrap gap-2 mb-6">
        {project.tech.map((t, i) => (
          <TechTag
            key={i}
            tech={t}
            isActive={filterTech === t}
            onClick={(e) => {
              e.stopPropagation();
              onFilterTech(t);
            }}
          />
        ))}
      </div>

      <div className="mt-auto pt-4 border-t border-slate-800/50 flex items-center gap-4">
        <button
          onClick={() => onExpand(project)}
          aria-label={`View analysis for ${project.title}`}
          className="flex items-center gap-2 text-sm font-medium text-emerald-500 hover:text-emerald-400 transition-colors cursor-pointer"
        >
          <BookOpen size={14} /> View Analysis
        </button>
        {project.gif && (
          <button
            onClick={() => onPreview(project)}
            aria-label={`Watch demo for ${project.title}`}
            className="flex items-center gap-2 text-sm font-medium text-slate-300 hover:text-white transition-colors ml-auto cursor-pointer"
          >
            <Eye size={14} /> Watch Demo
          </button>
        )}
        {!project.gif && project.link !== "#" && (
          <a
            href={project.link}
            target="_blank"
            rel="noreferrer"
            className="flex items-center gap-2 text-sm font-medium text-slate-300 hover:text-white transition-colors ml-auto"
          >
            <ExternalLink size={14} /> Source
          </a>
        )}
      </div>
    </motion.div>
  );
}
