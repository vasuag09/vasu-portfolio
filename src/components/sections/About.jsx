import React, { Suspense, lazy } from "react";
import { motion } from "framer-motion";
import { Github, Linkedin, Mail, CheckCircle2, ArrowUpRight, Camera, Briefcase, Quote } from "lucide-react";
import { useDocumentTitle } from "../../hooks/useDocumentTitle";
import { SOCIAL_LINKS } from "../../data/constants";
import { experience, testimonials } from "../../data/profile";
import MagneticElement from "../effects/MagneticElement";
import ScrollReveal from "../effects/ScrollReveal";
import TiltCard from "../effects/TiltCard";
import TextScramble from "../effects/TextScramble";
import ContactForm from "../ui/ContactForm";

const FloatingGeometry = lazy(() => import("../canvas/FloatingGeometry"));
const PoseDetection = lazy(() => import("../ml/PoseDetection"));

const CONTACT_NODES = [
  { label: "GitHub", href: SOCIAL_LINKS.github, icon: Github, desc: "Open source & projects" },
  { label: "LinkedIn", href: SOCIAL_LINKS.linkedin, icon: Linkedin, desc: "Professional network" },
  { label: "Email", href: SOCIAL_LINKS.email, icon: Mail, desc: "Direct contact" },
];

export default function About() {
  useDocumentTitle("Output Layer");

  return (
    <section id="about" className="max-w-4xl space-y-6 pb-20 md:pb-8">
      {/* Bio — Large, editorial style */}
      <motion.div
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        className="glass-card-static overflow-hidden"
      >
        <div className="flex flex-col md:flex-row items-start gap-6 p-6 md:p-8">
          <div className="flex-1">
            <div className="flex items-center gap-2 mb-5">
              <div className="w-2.5 h-2.5 rounded-full bg-gradient-to-br from-purple-400 to-cyan-400 animate-neural-breathe" />
              <span className="text-[10px] font-mono text-purple-400/60 tracking-[0.2em]">
                OUTPUT LAYER
              </span>
            </div>

            <h2
              className="text-3xl md:text-4xl font-extrabold text-white mb-6 tracking-tight"
              style={{ fontFamily: "var(--font-display)" }}
            >
              Vasu Agrawal
            </h2>

            <div className="space-y-4 text-slate-400 leading-relaxed">
              <p>
                AI/ML engineer and full-stack developer focused on building real,
                deployable systems. I specialize in deep learning, model debugging,
                computer vision, classical ML pipelines, and building AI-powered web
                applications.
              </p>
              <p>
                I work across the stack — Python, ML, TensorFlow, React/Next.js,
                Node.js — and prioritize shipping end-to-end projects that
                demonstrate clear engineering depth and practical value.
              </p>
            </div>
          </div>

          {/* 3D element */}
          <div className="hidden md:block shrink-0">
            <Suspense fallback={<div className="w-[140px] h-[140px]" />}>
              <FloatingGeometry tier="A" size={140} />
            </Suspense>
          </div>
        </div>
      </motion.div>

      {/* Experience */}
      {experience.length > 0 && (
        <ScrollReveal animation="fadeUp">
          <div className="glass-card-static p-6">
            <div className="flex items-center gap-3 mb-5">
              <Briefcase size={16} className="text-cyan-400/70" />
              <h3 className="text-[10px] font-mono text-cyan-500/60 tracking-[0.2em]">
                EXPERIENCE
              </h3>
            </div>
            <div className="space-y-5">
              {experience.map((exp, idx) => (
                <div key={idx} className="border-l-2 border-cyan-500/20 pl-4">
                  <h4
                    className="text-base font-bold text-white"
                    style={{ fontFamily: "var(--font-display)" }}
                  >
                    {exp.role}
                  </h4>
                  <p className="text-sm text-slate-400 mt-0.5">{exp.company}</p>
                  <p className="text-xs font-mono text-slate-600 mt-1">{exp.period}</p>
                  <p className="text-sm text-slate-400 mt-3 leading-relaxed">
                    {exp.description}
                  </p>
                  {exp.highlights && (
                    <div className="flex flex-wrap gap-2 mt-3">
                      {exp.highlights.map((h, i) => (
                        <span
                          key={i}
                          className="text-[10px] bg-[rgba(0,212,255,0.04)] border border-[rgba(0,212,255,0.08)] text-cyan-400/60 px-2 py-0.5 rounded-full font-mono"
                        >
                          {h}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        </ScrollReveal>
      )}

      {/* Open for Collaboration — Bento cards */}
      <ScrollReveal animation="fadeUp" stagger={0.1}>
        <TiltCard maxTilt={4}>
          <div className="glass-card-static p-6 md:p-8">
            <div className="flex items-center gap-3 mb-8">
              <div className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
              <span
                className="text-base font-bold text-white"
                style={{ fontFamily: "var(--font-display)" }}
              >
                Open for Collaboration
              </span>
            </div>

            <div className="grid md:grid-cols-2 gap-8">
              <div>
                <h4 className="text-[10px] font-mono text-cyan-500/60 tracking-[0.2em] mb-4">
                  ROLES
                </h4>
                <ul className="space-y-3">
                  {["AI/ML Engineer", "Full-Stack Developer", "Technical Co-Founder"].map(
                    (role) => (
                      <li
                        key={role}
                        className="flex items-center gap-3 text-slate-300 text-sm group"
                      >
                        <CheckCircle2 size={14} className="text-cyan-500/60 group-hover:text-cyan-400 transition-colors shrink-0" />
                        <TextScramble text={role} duration={300} className="cursor-default" />
                      </li>
                    ),
                  )}
                </ul>
              </div>
              <div>
                <h4 className="text-[10px] font-mono text-cyan-500/60 tracking-[0.2em] mb-4">
                  EXPERTISE
                </h4>
                <ul className="space-y-3">
                  {[
                    "MVP Development (0 to 1)",
                    "RAG Pipeline Design",
                    "Model Fine-Tuning & Deployment",
                  ].map((service) => (
                    <li
                      key={service}
                      className="flex items-center gap-3 text-slate-300 text-sm group"
                    >
                      <CheckCircle2 size={14} className="text-cyan-500/60 group-hover:text-cyan-400 transition-colors shrink-0" />
                      <TextScramble text={service} duration={300} className="cursor-default" />
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        </TiltCard>
      </ScrollReveal>

      {/* Education */}
      <ScrollReveal animation="slideLeft">
        <TiltCard maxTilt={5}>
          <div className="glass-card-static p-6">
            <h4 className="text-[10px] font-mono text-cyan-500/60 tracking-[0.2em] mb-4">
              EDUCATION
            </h4>
            <div>
              <span
                className="text-white font-bold text-lg"
                style={{ fontFamily: "var(--font-display)" }}
              >
                MBA Tech - Computer Engineering
              </span>
              <p className="text-sm text-slate-400 mt-1">
                SVKM&apos;s NMIMS (Mukesh Patel School)
              </p>
              <p className="text-xs text-slate-600 font-mono mt-2">
                2023 – Present · CGPA: 3.82/4
              </p>
            </div>
          </div>
        </TiltCard>
      </ScrollReveal>

      {/* Testimonials */}
      {testimonials.length > 0 && testimonials[0].name !== "Your Name Here" && (
        <ScrollReveal animation="fadeUp">
          <div className="glass-card-static p-6">
            <div className="flex items-center gap-3 mb-5">
              <Quote size={16} className="text-purple-400/70" />
              <h3 className="text-[10px] font-mono text-purple-400/60 tracking-[0.2em]">
                PEER VALIDATION
              </h3>
            </div>
            <div className="space-y-4">
              {testimonials.map((t, idx) => (
                <div key={idx} className="border-l-2 border-purple-500/20 pl-4">
                  <p className="text-sm text-slate-400 italic leading-relaxed mb-2">
                    &ldquo;{t.text}&rdquo;
                  </p>
                  <p className="text-xs font-mono text-slate-500">
                    — {t.name}, {t.role}
                  </p>
                </div>
              ))}
            </div>
          </div>
        </ScrollReveal>
      )}

      {/* Neural Vision — Fun interactive webcam effect */}
      <ScrollReveal animation="scaleIn">
        <div className="glass-card-static overflow-hidden">
          <div className="flex items-center gap-3 px-6 pt-6 mb-2">
            <Camera size={16} className="text-cyan-400/70" />
            <h3
              className="text-sm font-semibold text-white"
              style={{ fontFamily: "var(--font-display)" }}
            >
              Neural Vision
            </h3>
            <span className="text-[10px] font-mono text-slate-600 ml-auto">
              TRY IT OUT
            </span>
          </div>
          <p className="text-xs text-slate-500 px-6 mb-4">
            See the world through a neural network&apos;s eyes. Enable your camera for a real-time
            edge detection overlay — all processing happens locally, nothing is recorded.
          </p>
          <div className="px-6 pb-6">
            <Suspense
              fallback={
                <div className="flex items-center justify-center h-[200px] text-cyan-500/40 font-mono text-xs">
                  Loading neural vision...
                </div>
              }
            >
              <PoseDetection />
            </Suspense>
          </div>
        </div>
      </ScrollReveal>

      {/* Contact Form */}
      <ScrollReveal animation="fadeUp">
        <ContactForm />
      </ScrollReveal>

      {/* Establish Connection — Large interactive cards */}
      <ScrollReveal animation="fadeUp" stagger={0.12}>
        <h4 className="text-[10px] font-mono text-cyan-500/60 tracking-[0.2em] mb-4 px-1">
          ESTABLISH CONNECTION
        </h4>
        <div className="grid md:grid-cols-3 gap-4">
          {CONTACT_NODES.map((node, idx) => {
            const Icon = node.icon;
            return (
              <MagneticElement key={idx} strength={0.2}>
                <a
                  href={node.href}
                  target={node.href.startsWith("mailto:") ? undefined : "_blank"}
                  rel="noreferrer"
                  className="glass-card p-5 flex flex-col gap-3 group"
                >
                  <div className="flex items-center justify-between">
                    <div className="w-10 h-10 rounded-xl bg-[rgba(0,240,255,0.06)] border border-[rgba(0,240,255,0.1)] flex items-center justify-center group-hover:bg-[rgba(0,240,255,0.1)] group-hover:border-cyan-500/25 transition-all">
                      <Icon size={18} className="text-cyan-400/70 group-hover:text-cyan-300 transition-colors" />
                    </div>
                    <ArrowUpRight size={14} className="text-slate-600 group-hover:text-cyan-400 transition-colors" />
                  </div>
                  <div>
                    <span className="text-sm font-semibold text-white group-hover:text-cyan-200 transition-colors block"
                      style={{ fontFamily: "var(--font-display)" }}>
                      {node.label}
                    </span>
                    <span className="text-[11px] text-slate-500 font-mono">
                      {node.desc}
                    </span>
                  </div>
                </a>
              </MagneticElement>
            );
          })}
        </div>
      </ScrollReveal>
    </section>
  );
}
