import React, { lazy, useState } from "react";
import { motion } from "framer-motion";
import { FlaskConical, Filter } from "lucide-react";
import { Link } from "react-router-dom";
import { useDocumentTitle } from "../../hooks/useDocumentTitle";
import DemoCard from "../ml/DemoCard";
import ScrollReveal from "../effects/ScrollReveal";

// Lazy-load all demos — only fetched when user expands a card
const PoseDetection = lazy(() => import("../ml/PoseDetection"));
const ImageSegmentation = lazy(() => import("../ml/ImageSegmentation"));
const AttentionVisualizer = lazy(() => import("../ml/AttentionVisualizer"));
const AudioClassifier = lazy(() => import("../ml/AudioClassifier"));
const HandwritingSynthesis = lazy(() => import("../ml/HandwritingSynthesis"));

const DEMOS = [
  {
    id: "pose",
    title: "Neural Vision",
    subtitle: "WEBCAM · EDGE DETECTION · REAL-TIME",
    domain: "Vision",
    modelSize: "0KB (client-side)",
    about:
      "Uses your webcam to apply real-time edge detection and motion tracking with a neural network-inspired visual overlay. All processing happens locally in your browser — no data is sent anywhere.",
    Component: PoseDetection,
  },
  {
    id: "segmentation",
    title: "Image Segmentation",
    subtitle: "COLOR CLUSTERING · CANVAS 2D · INTERACTIVE",
    domain: "Vision",
    modelSize: "0KB (algorithmic)",
    about:
      "Upload an image and watch it get segmented into distinct regions based on color similarity. Uses threshold-based clustering to identify and highlight different visual segments.",
    Component: ImageSegmentation,
  },
  {
    id: "attention",
    title: "Attention Visualizer",
    subtitle: "SELF-ATTENTION · MULTI-HEAD · TRANSFORMER",
    domain: "NLP",
    modelSize: "0KB (computed)",
    about:
      "Visualizes how transformer self-attention works. Enter any sentence and see the attention weight heatmap — which words attend to which other words across multiple attention heads.",
    Component: AttentionVisualizer,
  },
  {
    id: "audio",
    title: "Audio Classifier",
    subtitle: "WEB AUDIO API · FREQUENCY ANALYSIS · LIVE",
    domain: "Audio",
    modelSize: "0KB (Web Audio)",
    about:
      "Captures microphone audio and performs real-time frequency analysis to classify sounds. Visualizes waveforms and frequency spectra with live classification predictions.",
    Component: AudioClassifier,
  },
  {
    id: "handwriting",
    title: "Handwriting Synthesis",
    subtitle: "PROCEDURAL GENERATION · STROKE ANIMATION · CANVAS",
    domain: "Generative",
    modelSize: "0KB (procedural)",
    about:
      "Type any text and watch it rendered as realistic handwriting, stroke by stroke. Uses procedural generation with natural variation in stroke width, baseline wobble, and character spacing.",
    Component: HandwritingSynthesis,
  },
];

const DOMAINS = ["All", "Vision", "NLP", "Audio", "Generative"];

export default function Research() {
  useDocumentTitle("Research Lab");
  const [activeFilter, setActiveFilter] = useState("All");

  const filteredDemos =
    activeFilter === "All"
      ? DEMOS
      : DEMOS.filter((d) => d.domain === activeFilter);

  return (
    <section id="research" className="space-y-8 pb-20 md:pb-8">
      {/* Hero banner */}
      <ScrollReveal animation="fadeUp">
        <div className="glass-card-static p-6 md:p-8">
          <div className="flex items-start gap-4">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-500/20 to-violet-500/20 flex items-center justify-center shrink-0">
              <FlaskConical size={20} className="text-cyan-400" />
            </div>
            <div>
              <h2
                className="text-xl md:text-2xl font-bold text-white mb-2"
                style={{ fontFamily: "var(--font-display)" }}
              >
                Research Lab
              </h2>
              <p className="text-sm text-slate-400 leading-relaxed max-w-xl">
                Interactive ML experiments running entirely in your browser. No data leaves
                your device — everything is computed client-side using Web APIs and TensorFlow.js.
              </p>
              <div className="flex items-center gap-3 mt-3">
                <span className="text-[10px] font-mono text-cyan-500/50">
                  {DEMOS.length} EXPERIMENTS
                </span>
                <span className="text-slate-700">·</span>
                <Link
                  to="/skills"
                  className="text-[10px] font-mono text-cyan-500/50 hover:text-cyan-400 transition-colors"
                >
                  ← BACK TO SKILLS
                </Link>
              </div>
            </div>
          </div>
        </div>
      </ScrollReveal>

      {/* Domain filter */}
      <div className="flex items-center gap-2 overflow-x-auto pb-1">
        <Filter size={12} className="text-slate-600 shrink-0" />
        {DOMAINS.map((domain) => (
          <button
            key={domain}
            onClick={() => setActiveFilter(domain)}
            className={`text-[10px] font-mono px-3 py-1.5 rounded-lg border transition-all cursor-pointer whitespace-nowrap ${
              activeFilter === domain
                ? "border-cyan-500/30 bg-cyan-500/10 text-cyan-400"
                : "border-[rgba(255,255,255,0.04)] text-slate-500 hover:text-slate-300 hover:border-slate-600"
            }`}
          >
            {domain}
          </button>
        ))}
      </div>

      {/* Demo cards */}
      <div className="space-y-4">
        {filteredDemos.map((demo, idx) => (
          <ScrollReveal key={demo.id} animation="fadeUp" delay={idx * 0.08}>
            <DemoCard
              title={demo.title}
              subtitle={demo.subtitle}
              domain={demo.domain}
              modelSize={demo.modelSize}
              about={demo.about}
            >
              <demo.Component />
            </DemoCard>
          </ScrollReveal>
        ))}
      </div>

      {/* Footer note */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.5 }}
        className="text-center py-4"
      >
        <p className="text-[10px] font-mono text-slate-700">
          ALL EXPERIMENTS RUN CLIENT-SIDE · NO DATA LEAVES YOUR BROWSER · PRIVACY FIRST
        </p>
      </motion.div>
    </section>
  );
}
