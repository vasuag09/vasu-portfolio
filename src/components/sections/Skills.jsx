import React from "react";
import { motion } from "framer-motion";
import { useNavigate } from "react-router-dom";
import { Activity, Award, ChevronRight } from "lucide-react";
import { skills, certifications } from "../../data/skills";
import { careerTrajectory } from "../../data/profile";
import { useDocumentTitle } from "../../hooks/useDocumentTitle";

import SectionWrapper from "../layout/SectionWrapper";

export default function Skills() {
  useDocumentTitle("Tech Stack");
  const navigate = useNavigate();

  const handleTechClick = (tech) => {
    navigate(`/projects?filter=${encodeURIComponent(tech)}`);
  };

  return (
    <SectionWrapper id="skills" className="space-y-6">
      {/* Career Trajectory */}
      <div className="bg-slate-900/30 border border-slate-800 rounded-lg p-6 mb-8">
        <div className="flex items-center gap-3 mb-6">
          <Activity className="text-emerald-500" size={24} />
          <h3 className="text-xl font-bold text-white">
            Competence Trajectory
          </h3>
        </div>
        <div className="relative border-l-2 border-slate-800 ml-4 space-y-8 pb-2">
          {careerTrajectory.map((milestone, idx) => (
            <motion.div
              key={idx}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: idx * 0.1, duration: 0.4 }}
              className="relative pl-8"
            >
              <div className="absolute -left-[9px] top-0 w-4 h-4 rounded-full bg-slate-900 border-2 border-emerald-500"></div>
              <div className="flex flex-col sm:flex-row sm:items-center justify-between mb-1">
                <span className="text-emerald-400 font-bold font-mono text-sm">
                  {milestone.year}
                </span>
                <div className="hidden sm:block flex-1 mx-4 h-px bg-slate-800"></div>
                <div className="flex items-center gap-2 text-xs text-slate-500 font-mono">
                  <span>LEVEL: {milestone.level}%</span>
                </div>
              </div>
              <h4 className="text-lg font-bold text-white mb-1">
                {milestone.title}
              </h4>
              <p className="text-slate-400 text-sm">{milestone.desc}</p>
            </motion.div>
          ))}
        </div>
      </div>

      {/* Skill Categories */}
      <div className="grid md:grid-cols-2 gap-6">
        {skills.map((skillGroup, idx) => (
          <motion.div
            key={idx}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: idx * 0.08, duration: 0.4 }}
            className="bg-slate-900/30 border border-slate-800 rounded-lg p-6"
          >
            <h3 className="text-emerald-500 font-mono text-sm mb-6 border-b border-slate-800 pb-2">
              &#47;&#47; {skillGroup.category}
            </h3>
            <div className="grid grid-cols-2 gap-4">
              {skillGroup.items.map((skill, sIdx) => (
                <button
                  key={sIdx}
                  onClick={() => handleTechClick(skill)}
                  className="flex items-center gap-3 group text-left hover:bg-slate-800/50 p-2 rounded -ml-2 transition-all cursor-pointer"
                >
                  <div className="w-2 h-2 bg-slate-700 rounded-sm group-hover:bg-emerald-500 transition-colors"></div>
                  <span className="text-slate-300 group-hover:text-white font-mono text-sm">
                    {skill}
                  </span>
                </button>
              ))}
            </div>
          </motion.div>
        ))}
      </div>

      {/* Certifications */}
      <div className="bg-gradient-to-r from-slate-900 to-slate-900/50 border border-slate-800 rounded-lg p-8">
        <div className="flex items-center gap-3 mb-6">
          <Award className="text-emerald-500" size={24} />
          <h3 className="text-xl font-bold text-white">
            Professional Certifications
          </h3>
        </div>
        <div className="grid md:grid-cols-2 gap-4">
          {certifications.map((cert, idx) => (
            <motion.div
              key={idx}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: idx * 0.05, duration: 0.3 }}
              className="flex items-start gap-3 text-slate-300 hover:text-white transition-colors"
            >
              <ChevronRight
                className="text-emerald-500 mt-0.5 shrink-0"
                size={16}
              />
              <span className="font-mono text-sm">{cert}</span>
            </motion.div>
          ))}
        </div>
      </div>
    </SectionWrapper>
  );
}
