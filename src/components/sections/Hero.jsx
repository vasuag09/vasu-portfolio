import React, { Suspense, lazy } from "react";
import { motion, useScroll, useTransform } from "framer-motion";
import { ChevronRight, Download, Sparkles } from "lucide-react";
import { Link } from "react-router-dom";
import { profile, stats } from "../../data/profile";
import { useDocumentTitle } from "../../hooks/useDocumentTitle";
import { CV_LINK } from "../../data/constants";
import DiffusionReveal from "../ml/DiffusionReveal";
import TextScramble from "../effects/TextScramble";
import MagneticElement from "../effects/MagneticElement";
import TiltCard from "../effects/TiltCard";
import ScrollReveal from "../effects/ScrollReveal";
import AnimatedCounter from "../effects/AnimatedCounter";

const FloatingGeometry = lazy(() => import("../canvas/FloatingGeometry"));

export default function Hero() {
  useDocumentTitle("Neural Network");

  // Scroll-driven parallax
  const { scrollY } = useScroll();
  const titleY = useTransform(scrollY, [0, 500], [0, -40]);
  const bioY = useTransform(scrollY, [0, 500], [0, -20]);
  const ctaY = useTransform(scrollY, [0, 500], [0, 10]);
  const geoY = useTransform(scrollY, [0, 500], [0, -60]);
  const geoRotate = useTransform(scrollY, [0, 500], [0, 15]);
  const statsY = useTransform(scrollY, [0, 500], [0, 30]);

  return (
    <section id="hero" className="min-h-[85vh] flex flex-col justify-center relative">
      <div className="flex flex-col md:flex-row items-start md:items-center gap-8 md:gap-16">
        {/* Text content */}
        <div className="relative z-10 flex-1 max-w-2xl">
          {/* Network status */}
          <motion.div
            initial={{ opacity: 0, x: -30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2, ease: [0.23, 1, 0.32, 1] }}
            className="flex items-center gap-2 mb-8"
          >
            <span className="w-2 h-2 rounded-full bg-[var(--accent-electric)] animate-neural-breathe" />
            <span className="text-[10px] font-mono text-cyan-500/60 tracking-[0.25em]">
              NETWORK ACTIVE · INPUT LAYER
            </span>
          </motion.div>

          {/* Name — Diffusion reveal with parallax */}
          <motion.div style={{ y: titleY }}>
            <DiffusionReveal
              firstName="VASU"
              lastName="AGRAWAL"
              className="h-[180px] md:h-[280px] lg:h-[320px] w-full mb-6"
            />
          </motion.div>

          {/* Title with accent line */}
          <motion.div
            initial={{ opacity: 0, width: 0 }}
            animate={{ opacity: 1, width: "100%" }}
            transition={{ delay: 1.0, duration: 0.8, ease: [0.23, 1, 0.32, 1] }}
            className="overflow-hidden mb-6"
            style={{ y: bioY }}
          >
            <div className="flex items-center gap-4">
              <div className="h-px flex-1 bg-gradient-to-r from-cyan-500/40 to-transparent" />
              <p className="text-sm md:text-base font-mono text-cyan-400/80 whitespace-nowrap">
                {profile.title}
              </p>
              <div className="h-px flex-1 bg-gradient-to-l from-purple-500/40 to-transparent" />
            </div>
          </motion.div>

          {/* Bio with parallax */}
          <motion.p
            initial={{ opacity: 0, y: 15 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 1.3, ease: [0.23, 1, 0.32, 1] }}
            className="text-slate-400 max-w-lg leading-relaxed mb-10 text-base"
            style={{ fontFamily: "var(--font-sans)", y: bioY }}
          >
            {profile.bio}
          </motion.p>

          {/* CTAs with parallax */}
          <motion.div
            initial={{ opacity: 0, y: 15 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 1.6, ease: [0.23, 1, 0.32, 1] }}
            className="flex flex-wrap gap-4"
            style={{ y: ctaY }}
          >
            <MagneticElement strength={0.15}>
              <Link
                to="/projects"
                className="btn-neural group bg-gradient-to-r from-cyan-600 to-cyan-500 hover:from-cyan-500 hover:to-cyan-400 text-white font-mono px-7 py-3.5 rounded-xl flex items-center gap-2 transition-all active:scale-95 cursor-pointer shadow-[0_0_30px_rgba(0,212,255,0.2)] hover:shadow-[0_0_50px_rgba(0,212,255,0.35)] text-sm"
              >
                <Sparkles size={14} />
                EXPLORE NETWORK
                <ChevronRight
                  size={14}
                  className="group-hover:translate-x-1 transition-transform"
                />
              </Link>
            </MagneticElement>
            <MagneticElement strength={0.15}>
              <a
                href={CV_LINK}
                target="_blank"
                rel="noreferrer"
                className="btn-neural border border-slate-700 hover:border-cyan-500/40 text-slate-300 hover:text-white font-mono px-7 py-3.5 rounded-xl flex items-center gap-2 transition-all cursor-pointer text-sm"
              >
                <Download size={14} /> DOWNLOAD CV
              </a>
            </MagneticElement>
          </motion.div>
        </div>

        {/* 3D Hero Element with parallax rotation */}
        <motion.div
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.8, duration: 1.2, ease: [0.23, 1, 0.32, 1] }}
          className="hidden md:block shrink-0"
          style={{ y: geoY, rotate: geoRotate }}
        >
          <Suspense fallback={<div className="w-[200px] h-[200px]" />}>
            <FloatingGeometry tier="S" size={220} />
          </Suspense>
        </motion.div>
      </div>

      {/* Stats — Bento grid with animated counters and gradient borders */}
      <ScrollReveal animation="fadeUp" stagger={0.1} className="mt-16 md:mt-24">
        <motion.div
          className="grid grid-cols-2 md:grid-cols-4 gap-3 md:gap-4"
          style={{ y: statsY }}
        >
          {stats.map((stat, idx) => (
            <TiltCard key={idx} maxTilt={8}>
              <div
                className="glass-card-glow p-5 md:p-6 group"
                style={{
                  background: idx === 0
                    ? "linear-gradient(135deg, rgba(0,240,255,0.04), rgba(17,24,39,0.6))"
                    : undefined,
                }}
              >
                <div className="flex items-center gap-2 mb-3">
                  <div className={`w-1.5 h-1.5 rounded-full ${
                    idx === 0 ? "bg-cyan-400 animate-neural-breathe" : "bg-slate-600"
                  }`} />
                  <TextScramble
                    text={stat.label}
                    className="text-[10px] text-slate-500 font-mono tracking-wider cursor-default"
                    duration={300}
                  />
                </div>
                <div
                  className="text-3xl md:text-4xl font-extrabold text-white"
                  style={{ fontFamily: "var(--font-display)" }}
                >
                  <AnimatedCounter value={stat.value} duration={2000} />
                </div>
              </div>
            </TiltCard>
          ))}
        </motion.div>
      </ScrollReveal>
    </section>
  );
}
