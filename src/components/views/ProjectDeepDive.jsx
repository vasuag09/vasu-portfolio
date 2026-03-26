import React, { Suspense, lazy } from "react";
import { motion } from "framer-motion";
import { useParams, useNavigate } from "react-router-dom";
import {
  ArrowLeft,
  ExternalLink,
  AlertTriangle,
  Target,
  ArrowRight,
} from "lucide-react";
import { projects } from "../../data/projects";
import { useDocumentTitle } from "../../hooks/useDocumentTitle";
import { TierBadge, StatusBadge } from "../ui/NeuralBadge";
import TechPill from "../ui/TechPill";

const FloatingGeometry = lazy(() => import("../canvas/FloatingGeometry"));

export default function ProjectDeepDive() {
  const { alias } = useParams();
  const navigate = useNavigate();
  const project = projects.find((p) => p.alias === alias);

  useDocumentTitle(project?.title || "Node Not Found");

  if (!project) {
    return (
      <div className="flex items-center justify-center min-h-[50vh]">
        <div className="text-center">
          <h2
            className="text-2xl font-bold text-white mb-4"
            style={{ fontFamily: "var(--font-display)" }}
          >
            NODE NOT FOUND
          </h2>
          <button
            onClick={() => navigate("/projects")}
            className="text-cyan-500 hover:text-cyan-400 font-mono cursor-pointer text-sm"
          >
            Back to Hidden Layer 1
          </button>
        </div>
      </div>
    );
  }

  const sections = [
    { key: "problem", label: "PROBLEM STATEMENT", content: project.details?.problem },
    { key: "architecture", label: "ARCHITECTURE", content: project.details?.architecture },
    { key: "decisions", label: "KEY DECISIONS", content: project.details?.decisions },
  ];

  // Parse pipeline string into steps
  const pipelineSteps = project.details?.pipeline
    ? project.details.pipeline.split("→").map((s) => s.trim())
    : [];

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="max-w-5xl pb-20 md:pb-8"
    >
      {/* Back navigation */}
      <button
        onClick={() => navigate("/projects")}
        className="flex items-center gap-2 text-slate-500 hover:text-cyan-400 transition-colors font-mono text-sm cursor-pointer mb-8"
      >
        <ArrowLeft size={16} /> Back to Network
      </button>

      {/* Header */}
      <div className="flex flex-col md:flex-row items-start gap-6 mb-8">
        <div className="flex-1">
          <div className="flex items-center gap-3 mb-3 flex-wrap">
            <TierBadge tier={project.tier} />
            <StatusBadge status={project.status} />
            {project.link !== "#" && (
              <a
                href={project.link}
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-1.5 text-xs font-mono text-cyan-500/70 hover:text-cyan-400 transition-colors"
              >
                View Live <ExternalLink size={12} />
              </a>
            )}
          </div>
          <h1
            className="text-3xl md:text-4xl font-bold text-white mb-4"
            style={{ fontFamily: "var(--font-display)" }}
          >
            {project.title}
          </h1>
          <p className="text-sm text-slate-400 leading-relaxed mb-4">
            {project.description}
          </p>
          <div className="flex flex-wrap gap-1.5">
            {project.tech.map((t) => (
              <TechPill key={t} label={t} onClick={() => {}} />
            ))}
          </div>
        </div>

        {/* 3D floating geometry */}
        <div className="hidden md:block shrink-0">
          <Suspense fallback={<div className="w-[120px] h-[120px]" />}>
            <FloatingGeometry tier={project.tier} size={120} />
          </Suspense>
        </div>
      </div>

      {/* Pipeline visualization */}
      {pipelineSteps.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 15 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="glass-card-static p-5 mb-5"
        >
          <div className="flex items-center gap-2 mb-4">
            <div className="w-1.5 h-1.5 rounded-full bg-cyan-500/40" />
            <h3 className="text-[10px] font-mono text-cyan-500/50 tracking-widest">
              PIPELINE
            </h3>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            {pipelineSteps.map((step, idx) => (
              <React.Fragment key={idx}>
                <div className="bg-[rgba(0,240,255,0.04)] border border-[rgba(0,240,255,0.1)] rounded-lg px-3 py-1.5 text-xs font-mono text-cyan-300/80 whitespace-nowrap">
                  {step}
                </div>
                {idx < pipelineSteps.length - 1 && (
                  <ArrowRight size={12} className="text-cyan-500/30 shrink-0" />
                )}
              </React.Fragment>
            ))}
          </div>
        </motion.div>
      )}

      {/* Blueprint-style sections */}
      <div className="grid lg:grid-cols-3 gap-5">
        <div className="lg:col-span-2 space-y-5">
          {sections.map((section, idx) => (
            section.content && (
              <motion.div
                key={section.key}
                initial={{ opacity: 0, y: 15 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: idx * 0.1 + 0.2 }}
                className="glass-card-static p-5"
              >
                <div className="flex items-center gap-2 mb-3">
                  <div className="w-1.5 h-1.5 rounded-full bg-cyan-500/40" />
                  <h3 className="text-[10px] font-mono text-cyan-500/50 tracking-widest">
                    {section.label}
                  </h3>
                </div>
                <p className="text-sm text-slate-400 leading-relaxed">
                  {section.content}
                </p>
              </motion.div>
            )
          ))}
        </div>

        <div className="space-y-5">
          {/* Failure Modes */}
          {project.details?.failures && (
            <motion.div
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 }}
              className="glass-card-static p-5 border-red-500/10"
            >
              <div className="flex items-center gap-2 mb-3">
                <AlertTriangle size={12} className="text-red-400/60" />
                <h3 className="text-[10px] font-mono text-red-400/50 tracking-widest">
                  FAILURE MODES
                </h3>
              </div>
              <p className="text-sm text-slate-400 leading-relaxed">
                {project.details.failures}
              </p>
            </motion.div>
          )}

          {/* Metrics */}
          {project.details?.metrics && (
            <motion.div
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5 }}
              className="glass-card-static p-5 border-cyan-500/10"
            >
              <div className="flex items-center gap-2 mb-3">
                <Target size={12} className="text-cyan-400/60" />
                <h3 className="text-[10px] font-mono text-cyan-400/50 tracking-widest">
                  METRICS
                </h3>
              </div>
              <p className="text-sm text-cyan-200/80 font-mono leading-relaxed">
                {project.details.metrics}
              </p>
            </motion.div>
          )}
        </div>
      </div>
    </motion.div>
  );
}
