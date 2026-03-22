import React, { useState, useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { MessageSquareText, RotateCcw } from "lucide-react";

/**
 * Sentiment Heatmap — type text and see each word colored by sentiment.
 * Uses a subset of the AFINN-165 lexicon for instant, zero-model analysis.
 * Green = positive, Red = negative, Gray = neutral.
 */

// AFINN-165 subset — most common English sentiment words (scores: -5 to +5)
const AFINN = {
  // Strong positive
  love: 3, loved: 3, loves: 3, loving: 3, excellent: 3, amazing: 4, awesome: 4,
  fantastic: 4, wonderful: 4, brilliant: 4, outstanding: 5, superb: 5,
  beautiful: 3, perfect: 3, great: 3, best: 3, happy: 3, happiness: 3,
  joy: 3, joyful: 3, excited: 3, exciting: 3, delightful: 3, incredible: 4,
  magnificent: 3, extraordinary: 3, exceptional: 3, remarkable: 3,
  // Moderate positive
  good: 2, nice: 2, like: 2, liked: 2, enjoy: 2, enjoyed: 2, pleasant: 2,
  pleased: 2, fun: 2, glad: 2, cool: 1, fine: 1, helpful: 2, kind: 2,
  smart: 2, clever: 2, impressive: 3, inspire: 2, inspired: 2, inspiring: 3,
  innovative: 2, creative: 2, positive: 2, success: 2, successful: 2,
  win: 2, winning: 2, won: 2, hope: 2, hopeful: 2, thank: 2, thanks: 2,
  grateful: 2, appreciate: 2, recommend: 2, worth: 1, valuable: 2,
  // Mild positive
  ok: 1, okay: 1, agree: 1, easy: 1, interesting: 1, useful: 1,
  // Strong negative
  hate: -3, hated: -3, hates: -3, terrible: -3, horrible: -3, awful: -3,
  worst: -3, disgusting: -3, pathetic: -3, disaster: -3, catastrophe: -3,
  devastating: -3, atrocious: -3, appalling: -3, dreadful: -3,
  // Moderate negative
  bad: -2, ugly: -2, poor: -2, sad: -2, angry: -2, anger: -2, annoying: -2,
  annoyed: -2, boring: -2, bored: -2, fail: -2, failed: -2, failure: -2,
  wrong: -2, broken: -2, pain: -2, painful: -2, hurt: -2, disappointed: -2,
  disappointing: -2, frustrating: -2, frustrated: -2, difficult: -1,
  problem: -2, problems: -2, mistake: -2, stupid: -2, useless: -2,
  waste: -2, wasted: -2, fear: -2, fearful: -2, worried: -2,
  worry: -2, stress: -2, stressed: -2, anxious: -2, negative: -2,
  // Mild negative
  hard: -1, slow: -1, miss: -1, missed: -1, lost: -1, lose: -1,
  confusing: -1, confused: -1, complex: -1, weird: -1, strange: -1,
  // Negation
  not: -1, no: -1, never: -1, neither: -1, nobody: -1, nothing: -1,
  nowhere: -1, hardly: -1,
};

function analyzeSentiment(text) {
  if (!text.trim()) return { words: [], overall: 0, label: "neutral" };

  const words = text.split(/\s+/).filter(Boolean);
  let totalScore = 0;
  let scoredCount = 0;

  const analyzed = words.map((word) => {
    const clean = word.toLowerCase().replace(/[^a-z']/g, "");
    const score = AFINN[clean] ?? 0;
    if (score !== 0) scoredCount++;
    totalScore += score;
    return {
      original: word,
      clean,
      score,
    };
  });

  const avgScore = scoredCount > 0 ? totalScore / scoredCount : 0;
  const normalizedScore = Math.max(-1, Math.min(1, totalScore / Math.max(words.length * 0.5, 1)));

  let label = "Neutral";
  if (normalizedScore > 0.15) label = "Positive";
  if (normalizedScore > 0.5) label = "Very Positive";
  if (normalizedScore < -0.15) label = "Negative";
  if (normalizedScore < -0.5) label = "Very Negative";

  return {
    words: analyzed,
    overall: normalizedScore,
    avgScore,
    totalScore,
    label,
  };
}

function getWordColor(score) {
  if (score === 0) return "text-slate-400";
  if (score >= 3) return "text-emerald-300";
  if (score >= 1) return "text-emerald-400/80";
  if (score <= -3) return "text-red-400";
  if (score <= -1) return "text-red-400/80";
  return "text-slate-400";
}

function getWordBg(score) {
  if (score === 0) return "";
  if (score >= 3) return "bg-emerald-500/15";
  if (score >= 1) return "bg-emerald-500/8";
  if (score <= -3) return "bg-red-500/15";
  if (score <= -1) return "bg-red-500/8";
  return "";
}

function getScoreBarColor(score) {
  if (score > 0) return "from-emerald-500 to-emerald-400";
  if (score < 0) return "from-red-500 to-red-400";
  return "from-slate-500 to-slate-400";
}

const EXAMPLE_SENTENCES = [
  "I love building amazing neural networks that solve incredible problems",
  "This terrible bug is really frustrating and disappointing",
  "The weather today is okay nothing special happening",
  "Machine learning is an exciting and wonderful field of research",
];

export default function SentimentHeatmap() {
  const [text, setText] = useState("");
  const [hoveredIdx, setHoveredIdx] = useState(null);

  const result = useMemo(() => analyzeSentiment(text), [text]);

  return (
    <motion.div
      initial={{ opacity: 0, y: 30 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.3 }}
      className="glass-card-static p-6 space-y-4"
    >
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-3 h-3 rounded-full bg-gradient-to-br from-emerald-400 to-red-400 animate-neural-breathe" />
          <div>
            <h3
              className="text-sm font-semibold text-white"
              style={{ fontFamily: "var(--font-display)" }}
            >
              Sentiment Heatmap
            </h3>
            <p className="text-[10px] font-mono text-cyan-500/50">
              AFINN-165 LEXICON · REAL-TIME NLP
            </p>
          </div>
        </div>
        {text && (
          <button
            onClick={() => setText("")}
            className="text-slate-500 hover:text-white transition-colors cursor-pointer p-1"
            aria-label="Clear text"
          >
            <RotateCcw size={14} />
          </button>
        )}
      </div>

      <p className="text-xs text-slate-500">
        Type any sentence — each word is analyzed for sentiment in real-time.
        Hover words to see their individual scores.
      </p>

      {/* Input */}
      <div className="relative">
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Type anything — I'll analyze the sentiment..."
          className="w-full bg-[rgba(255,255,255,0.03)] border border-[rgba(0,212,255,0.08)] rounded-xl px-4 py-3 text-sm text-slate-200 font-mono placeholder:text-slate-600 focus:outline-none focus:border-cyan-500/30 resize-none transition-colors"
          rows={2}
          style={{ fontFamily: "var(--font-mono)" }}
        />
        <MessageSquareText
          size={14}
          className="absolute top-3 right-3 text-slate-600"
        />
      </div>

      {/* Example sentences */}
      {!text && (
        <div className="flex flex-wrap gap-2">
          <span className="text-[10px] font-mono text-slate-600 self-center">TRY:</span>
          {EXAMPLE_SENTENCES.map((sentence, i) => (
            <button
              key={i}
              onClick={() => setText(sentence)}
              className="text-[10px] font-mono text-cyan-500/50 hover:text-cyan-400 border border-[rgba(0,212,255,0.06)] hover:border-cyan-500/20 rounded-lg px-2.5 py-1 transition-all cursor-pointer truncate max-w-[200px]"
            >
              {sentence.slice(0, 35)}...
            </button>
          ))}
        </div>
      )}

      {/* Analyzed words */}
      <AnimatePresence>
        {result.words.length > 0 && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="space-y-4"
          >
            {/* Word display */}
            <div className="flex flex-wrap gap-1.5 p-3 rounded-xl bg-[rgba(255,255,255,0.02)] border border-[rgba(255,255,255,0.03)] min-h-[48px]">
              {result.words.map((word, idx) => (
                <span
                  key={`${idx}-${word.original}`}
                  className={`relative px-1.5 py-0.5 rounded text-sm font-mono transition-all cursor-default ${getWordColor(word.score)} ${getWordBg(word.score)}`}
                  onMouseEnter={() => setHoveredIdx(idx)}
                  onMouseLeave={() => setHoveredIdx(null)}
                >
                  {word.original}

                  {/* Tooltip */}
                  {hoveredIdx === idx && word.score !== 0 && (
                    <motion.div
                      initial={{ opacity: 0, y: 5 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="absolute -top-8 left-1/2 -translate-x-1/2 bg-slate-900/95 border border-slate-700/50 rounded-md px-2 py-1 text-[10px] font-mono whitespace-nowrap z-10"
                    >
                      <span className={word.score > 0 ? "text-emerald-400" : "text-red-400"}>
                        {word.score > 0 ? "+" : ""}{word.score}
                      </span>
                    </motion.div>
                  )}
                </span>
              ))}
            </div>

            {/* Overall score */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-[10px] font-mono text-slate-600 tracking-wider">
                  OVERALL SENTIMENT
                </span>
                <span className={`text-xs font-mono font-bold ${
                  result.overall > 0.15 ? "text-emerald-400" :
                  result.overall < -0.15 ? "text-red-400" :
                  "text-slate-400"
                }`}>
                  {result.label}
                </span>
              </div>

              {/* Score bar */}
              <div className="relative h-2 bg-[rgba(255,255,255,0.04)] rounded-full overflow-hidden">
                {/* Center marker */}
                <div className="absolute top-0 bottom-0 left-1/2 w-px bg-slate-600/50 z-10" />
                {/* Score fill */}
                <motion.div
                  initial={{ width: 0 }}
                  animate={{
                    width: `${Math.abs(result.overall) * 50}%`,
                    left: result.overall >= 0 ? "50%" : undefined,
                    right: result.overall < 0 ? "50%" : undefined,
                  }}
                  transition={{ duration: 0.3 }}
                  className={`absolute top-0 bottom-0 rounded-full bg-gradient-to-r ${getScoreBarColor(result.overall)}`}
                  style={{
                    ...(result.overall >= 0
                      ? { left: "50%" }
                      : { right: "50%" }),
                  }}
                />
              </div>

              {/* Scale labels */}
              <div className="flex justify-between text-[9px] font-mono text-slate-700">
                <span>Negative</span>
                <span>Neutral</span>
                <span>Positive</span>
              </div>
            </div>

            {/* Stats row */}
            <div className="flex gap-4 text-[10px] font-mono text-slate-600">
              <span>Words: {result.words.length}</span>
              <span>Score: {result.totalScore > 0 ? "+" : ""}{result.totalScore}</span>
              <span>Avg: {result.avgScore.toFixed(1)}</span>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}
