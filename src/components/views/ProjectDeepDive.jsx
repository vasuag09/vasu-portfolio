import React from "react";
import { motion } from "framer-motion";
import { useParams, useNavigate } from "react-router-dom";
import {
  ArrowLeft,
  ExternalLink,
  Layout,
  GitBranch,
  Brain,
  AlertTriangle,
  Target,
} from "lucide-react";
import { projects } from "../../data/projects";
import { useDocumentTitle } from "../../hooks/useDocumentTitle";
import TierBadge from "../ui/TierBadge";

export default function ProjectDeepDive() {
  const { alias } = useParams();
  const navigate = useNavigate();
  const project = projects.find((p) => p.alias === alias);

  useDocumentTitle(project?.title || "Project Not Found");

  if (!project) {
    return (
      <div className="min-h-screen bg-slate-950 flex items-center justify-center">
        <div className="text-center">
          <h2 className="text-2xl font-bold text-white mb-4 font-mono">
            PROJECT NOT FOUND
          </h2>
          <button
            onClick={() => navigate("/projects")}
            className="text-emerald-500 hover:underline font-mono cursor-pointer"
          >
            Back to Deployments
          </button>
        </div>
      </div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="max-w-7xl mx-auto flex flex-col"
    >
      {/* Navigation bar */}
      <div className="flex items-center justify-between mb-8">
        <button
          onClick={() => navigate("/projects")}
          className="flex items-center gap-2 text-slate-400 hover:text-emerald-400 transition-colors font-mono text-sm cursor-pointer"
        >
          <ArrowLeft size={16} /> BACK TO DEPLOYMENTS
        </button>
        <div className="flex gap-4">
          {project.link !== "#" && (
            <a
              href={project.link}
              target="_blank"
              rel="noreferrer"
              className="bg-emerald-600 hover:bg-emerald-500 text-white px-4 py-2 rounded text-sm font-mono flex items-center gap-2 transition-colors cursor-pointer"
            >
              View Live <ExternalLink size={14} />
            </a>
          )}
        </div>
      </div>

      {/* Content */}
      <motion.div
        initial={{ y: 20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ delay: 0.1 }}
        className="grid lg:grid-cols-3 gap-12"
      >
        <div className="lg:col-span-2 space-y-12">
          <div>
            <div className="flex items-center gap-4 mb-4">
              <h1 className="text-3xl md:text-4xl font-bold text-white">
                {project.title}
              </h1>
              <TierBadge tier={project.tier} size="lg" />
            </div>
            <div className="flex flex-wrap gap-2 mb-8">
              {project.tech.map((t, i) => (
                <span
                  key={i}
                  className="text-xs font-mono text-emerald-400 bg-emerald-900/20 border border-emerald-500/20 px-2 py-1 rounded"
                >
                  {t}
                </span>
              ))}
            </div>
            <p className="text-slate-300 leading-relaxed text-lg border-l-4 border-emerald-500 pl-6">
              {project.details?.problem}
            </p>
          </div>

          <div>
            <h3 className="text-xl font-bold text-white flex items-center gap-2 mb-4">
              <Layout className="text-emerald-500" /> System Architecture
            </h3>
            <div className="bg-slate-900/50 border border-slate-800 p-6 rounded-lg font-mono text-sm text-slate-400">
              {project.details?.architecture}
            </div>
          </div>

          <div>
            <h3 className="text-xl font-bold text-white flex items-center gap-2 mb-4">
              <GitBranch className="text-emerald-500" /> Engineering Pipeline
            </h3>
            <p className="text-slate-400 leading-relaxed">
              {project.details?.pipeline}
            </p>
          </div>
        </div>

        <div className="space-y-8">
          <div className="bg-slate-900/30 border border-slate-800 p-6 rounded-lg">
            <h3 className="text-sm font-bold text-white flex items-center gap-2 mb-4 uppercase tracking-wider">
              <Brain size={16} className="text-emerald-500" /> Key Decisions
            </h3>
            <p className="text-slate-400 text-sm leading-relaxed">
              {project.details?.decisions}
            </p>
          </div>

          <div className="bg-red-900/10 border border-red-500/20 p-6 rounded-lg">
            <h3 className="text-sm font-bold text-red-400 flex items-center gap-2 mb-4 uppercase tracking-wider">
              <AlertTriangle size={16} /> Failure Modes
            </h3>
            <p className="text-slate-400 text-sm leading-relaxed">
              {project.details?.failures}
            </p>
          </div>

          <div className="bg-emerald-900/10 border border-emerald-500/20 p-6 rounded-lg">
            <h3 className="text-sm font-bold text-emerald-400 flex items-center gap-2 mb-4 uppercase tracking-wider">
              <Target size={16} /> Performance Metrics
            </h3>
            <div className="text-slate-300 font-mono text-sm">
              {project.details?.metrics}
            </div>
          </div>
        </div>
      </motion.div>
    </motion.div>
  );
}
