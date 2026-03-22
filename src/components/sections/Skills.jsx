import React, { Suspense, lazy } from "react";
import { motion } from "framer-motion";
import { useNavigate, Link } from "react-router-dom";
import { Award, FlaskConical, ArrowRight } from "lucide-react";
import { skills, certifications } from "../../data/skills";
import { careerTrajectory } from "../../data/profile";
import { useDocumentTitle } from "../../hooks/useDocumentTitle";
import ScrollReveal from "../effects/ScrollReveal";
import TextScramble from "../effects/TextScramble";
import TiltCard from "../effects/TiltCard";
import LiveInference from "../ml/LiveInference";
import SentimentHeatmap from "../ml/SentimentHeatmap";

const SkillOrb = lazy(() => import("../canvas/SkillOrb"));

// Flatten all skills for the orb
const allSkills = skills.flatMap((group) => group.items);

export default function Skills() {
  useDocumentTitle("Weights & Biases");
  const navigate = useNavigate();

  const handleTechClick = (tech) => {
    navigate(`/projects?tech=${encodeURIComponent(tech)}`);
  };

  return (
    <section id="skills" className="space-y-8 pb-20 md:pb-8">
      {/* 3D Skill Orb — Interactive */}
      <div className="glass-card-static overflow-hidden">
        <div className="flex items-center gap-3 px-6 pt-6 mb-2">
          <div className="w-2.5 h-2.5 rounded-full bg-gradient-to-br from-cyan-400 to-violet-500 animate-neural-breathe" />
          <h3
            className="text-sm font-semibold text-white tracking-wide"
            style={{ fontFamily: "var(--font-display)" }}
          >
            Neural Weight Space
          </h3>
          <span className="text-[10px] font-mono text-slate-600 ml-auto">
            DRAG TO EXPLORE
          </span>
        </div>
        <p className="text-xs text-slate-500 px-6 mb-4">
          Interactive 3D visualization of technical competencies. Each node represents a skill — hover to highlight.
        </p>
        <div className="h-[350px] md:h-[420px]">
          <Suspense fallback={
            <div className="flex items-center justify-center h-full text-cyan-500/40 font-mono text-xs">
              Initializing weight space...
            </div>
          }>
            <SkillOrb skills={allSkills} className="w-full h-full" />
          </Suspense>
        </div>
      </div>

      {/* Training Epochs — Career Timeline */}
      <ScrollReveal animation="fadeUp">
        <div className="glass-card-static p-6">
          <h3 className="text-sm font-mono text-cyan-500/70 tracking-wider mb-6">
            TRAINING EPOCHS
          </h3>
          <div className="relative">
            {/* Horizontal connection line */}
            <div className="absolute top-4 left-4 right-4 h-px bg-gradient-to-r from-cyan-500/20 via-purple-500/20 to-purple-500/10 hidden md:block" />

            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {careerTrajectory.map((milestone, idx) => (
                <TiltCard key={idx} maxTilt={5}>
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: idx * 0.12 }}
                    className="relative p-3"
                  >
                    {/* Epoch node */}
                    <div className="flex items-center gap-2 mb-3">
                      <div
                        className={`w-3 h-3 rounded-full shrink-0 ${
                          idx === careerTrajectory.length - 1
                            ? "bg-cyan-400 shadow-[0_0_10px_rgba(0,212,255,0.4)] animate-neural-breathe"
                            : "bg-slate-500"
                        }`}
                      />
                      <span className="text-cyan-400 font-mono text-sm font-bold">
                        {milestone.year}
                      </span>
                    </div>

                    <h4
                      className="text-sm font-semibold text-white mb-1"
                      style={{ fontFamily: "var(--font-display)" }}
                    >
                      {milestone.title}
                    </h4>
                    <p className="text-xs text-slate-500">{milestone.desc}</p>

                    {/* Weight bar */}
                    <div className="mt-3 h-1.5 bg-[rgba(255,255,255,0.04)] rounded-full overflow-hidden">
                      <motion.div
                        initial={{ width: 0 }}
                        whileInView={{ width: `${milestone.level}%` }}
                        viewport={{ once: true }}
                        transition={{ duration: 1, delay: idx * 0.15, ease: [0.23, 1, 0.32, 1] }}
                        className="h-full rounded-full"
                        style={{
                          background: `linear-gradient(90deg, var(--accent-electric), var(--accent-violet))`,
                        }}
                      />
                    </div>
                  </motion.div>
                </TiltCard>
              ))}
            </div>
          </div>
        </div>
      </ScrollReveal>

      {/* Skill Categories — Clickable Grid */}
      <div className="grid md:grid-cols-2 gap-5">
        {skills.map((skillGroup, idx) => (
          <ScrollReveal key={idx} animation={idx % 2 === 0 ? "slideLeft" : "slideRight"}>
            <div className="glass-card-static p-5">
              <h3 className="text-xs font-mono text-cyan-500/60 tracking-wider mb-5 pb-2 border-b border-[rgba(0,212,255,0.06)]">
                <TextScramble text={skillGroup.category} duration={400} />
              </h3>
              <div className="space-y-2">
                {skillGroup.items.map((skill, sIdx) => (
                  <button
                    key={sIdx}
                    onClick={() => handleTechClick(skill)}
                    className="flex items-center gap-3 group w-full text-left hover:bg-[rgba(0,212,255,0.03)] p-2 rounded-lg -ml-1 transition-all cursor-pointer"
                  >
                    <div className="w-2 h-2 rounded-full bg-slate-600 group-hover:bg-cyan-400 group-hover:shadow-[0_0_8px_rgba(0,212,255,0.4)] transition-all shrink-0" />
                    <span className="text-sm text-slate-400 group-hover:text-cyan-200 font-mono transition-colors">
                      {skill}
                    </span>
                  </button>
                ))}
              </div>
            </div>
          </ScrollReveal>
        ))}
      </div>

      {/* Live ML Demos */}
      <ScrollReveal animation="scaleIn">
        <div className="space-y-5">
          <LiveInference />
          <SentimentHeatmap />

          {/* Research Lab CTA */}
          <Link
            to="/research"
            className="group flex items-center justify-between p-4 rounded-xl border border-[rgba(0,212,255,0.06)] hover:border-cyan-500/20 bg-[rgba(0,212,255,0.02)] hover:bg-[rgba(0,212,255,0.04)] transition-all"
          >
            <div className="flex items-center gap-3">
              <FlaskConical size={16} className="text-cyan-500/60" />
              <div>
                <p className="text-xs font-semibold text-slate-300 group-hover:text-white transition-colors">
                  Want more? Visit the Research Lab
                </p>
                <p className="text-[10px] font-mono text-slate-600">
                  5 MORE INTERACTIVE EXPERIMENTS · VISION · NLP · AUDIO · GENERATIVE
                </p>
              </div>
            </div>
            <ArrowRight size={14} className="text-slate-600 group-hover:text-cyan-400 group-hover:translate-x-1 transition-all" />
          </Link>
        </div>
      </ScrollReveal>

      {/* Certifications — Validated Weights */}
      <ScrollReveal animation="fadeUp">
        <div className="glass-card-static p-6">
          <div className="flex items-center gap-3 mb-5">
            <Award size={18} className="text-purple-400" />
            <h3 className="text-sm font-mono text-purple-400/80 tracking-wider">
              VALIDATED WEIGHTS
            </h3>
          </div>
          <div className="grid md:grid-cols-2 gap-3">
            {certifications.map((cert, idx) => (
              <motion.div
                key={idx}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: idx * 0.06 }}
                className="flex items-start gap-3 text-slate-400 hover:text-slate-200 transition-colors p-2"
              >
                <div className="w-1.5 h-1.5 rounded-full bg-purple-500/50 mt-1.5 shrink-0" />
                <span className="text-sm">{cert}</span>
              </motion.div>
            ))}
          </div>
        </div>
      </ScrollReveal>
    </section>
  );
}
