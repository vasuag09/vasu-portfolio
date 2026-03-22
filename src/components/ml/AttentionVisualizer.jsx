import React, { useState, useMemo, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Brain, Grid3x3, Type, RotateCcw } from "lucide-react";

/**
 * Transformer Attention Visualizer — computes self-attention from scratch.
 * Tokenizes text, generates deterministic embeddings via hashing,
 * builds Q/K/V projections for 4 heads, and renders an interactive heatmap.
 * No external ML model required.
 */

const NUM_HEADS = 4;
const D_MODEL = 32;
const D_K = D_MODEL / NUM_HEADS; // 8

const DEMO_SENTENCES = [
  "The cat sat on the mat",
  "Machine learning models process natural language",
  "Attention is all you need",
  "Neural networks learn hierarchical representations",
  "Transformers revolutionized natural language processing",
  "The quick brown fox jumps over the lazy dog",
];

const HEAD_LABELS = [
  { name: "Head 1", desc: "positional" },
  { name: "Head 2", desc: "syntactic" },
  { name: "Head 3", desc: "semantic" },
  { name: "Head 4", desc: "global" },
];

// --- Deterministic math utilities ---

function hashString(str, seed = 0) {
  let h = seed | 0;
  for (let i = 0; i < str.length; i++) {
    h = Math.imul(h ^ str.charCodeAt(i), 2654435761);
    h = (h >>> 0) ^ (h >>> 16);
  }
  return h >>> 0;
}

function seededRandom(seed) {
  let s = seed | 0;
  return function next() {
    s = Math.imul(s ^ (s >>> 15), 1 | s);
    s = (s + Math.imul(s ^ (s >>> 7), 61 | s)) ^ s;
    return ((s ^ (s >>> 14)) >>> 0) / 4294967296;
  };
}

function createEmbedding(word, dim) {
  const rng = seededRandom(hashString(word.toLowerCase(), 42));
  const emb = new Float32Array(dim);
  for (let i = 0; i < dim; i++) {
    emb[i] = (rng() - 0.5) * 2;
  }
  return emb;
}

function createProjectionMatrix(rows, cols, seed) {
  const rng = seededRandom(seed);
  const mat = [];
  const scale = Math.sqrt(2.0 / (rows + cols));
  for (let r = 0; r < rows; r++) {
    const row = new Float32Array(cols);
    for (let c = 0; c < cols; c++) {
      row[c] = (rng() - 0.5) * 2 * scale;
    }
    mat.push(row);
  }
  return mat;
}

function matVecMul(matrix, vec) {
  const out = new Float32Array(matrix.length);
  for (let r = 0; r < matrix.length; r++) {
    let sum = 0;
    for (let c = 0; c < vec.length; c++) {
      sum += matrix[r][c] * vec[c];
    }
    out[r] = sum;
  }
  return out;
}

function dotProduct(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
  return sum;
}

function softmax(arr) {
  const max = Math.max(...arr);
  const exps = arr.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

// --- Attention computation ---

function computeAttention(tokens) {
  if (tokens.length === 0) return [];

  const embeddings = tokens.map((t) => createEmbedding(t, D_MODEL));

  const heads = [];
  for (let h = 0; h < NUM_HEADS; h++) {
    const seedBase = (h + 1) * 1337;
    const Wq = createProjectionMatrix(D_K, D_MODEL, seedBase);
    const Wk = createProjectionMatrix(D_K, D_MODEL, seedBase + 100);

    const queries = embeddings.map((e) => matVecMul(Wq, e));
    const keys = embeddings.map((e) => matVecMul(Wk, e));

    const scaleFactor = Math.sqrt(D_K);
    const attentionMatrix = [];

    for (let i = 0; i < tokens.length; i++) {
      const scores = [];
      for (let j = 0; j < tokens.length; j++) {
        scores.push(dotProduct(queries[i], keys[j]) / scaleFactor);
      }
      attentionMatrix.push(softmax(scores));
    }

    heads.push(attentionMatrix);
  }

  return heads;
}

// --- Color helpers ---

function attentionToColor(weight) {
  const t = Math.min(1, Math.max(0, weight));
  const r = Math.round(40 + t * 99);
  const g = Math.round(40 + t * 150);
  const b = Math.round(60 + t * 195);
  return `rgb(${r}, ${g}, ${b})`;
}

function attentionToHighlightOpacity(weight) {
  return Math.min(1, Math.max(0.05, weight));
}

export default function AttentionVisualizer() {
  const [text, setText] = useState("");
  const [activeHead, setActiveHead] = useState(0);
  const [hoveredCell, setHoveredCell] = useState(null);
  const [selectedRow, setSelectedRow] = useState(null);
  const [hoveredWord, setHoveredWord] = useState(null);

  const tokens = useMemo(() => {
    const t = text.trim();
    return t ? t.split(/\s+/).filter(Boolean) : [];
  }, [text]);

  const attentionHeads = useMemo(() => computeAttention(tokens), [tokens]);

  const currentAttention = attentionHeads[activeHead] || [];

  const handleReset = useCallback(() => {
    setText("");
    setActiveHead(0);
    setHoveredCell(null);
    setSelectedRow(null);
    setHoveredWord(null);
  }, []);

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
          <div className="w-3 h-3 rounded-full bg-gradient-to-br from-cyan-400 to-violet-500 animate-neural-breathe" />
          <div>
            <h3
              className="text-sm font-semibold text-white"
              style={{ fontFamily: "var(--font-display)" }}
            >
              Attention Visualizer
            </h3>
            <p className="text-[10px] font-mono text-cyan-500/50">
              SELF-ATTENTION · {NUM_HEADS} HEADS · d_k={D_K}
            </p>
          </div>
        </div>
        {text && (
          <button
            onClick={handleReset}
            className="text-slate-500 hover:text-white transition-colors cursor-pointer p-1"
            aria-label="Clear text"
          >
            <RotateCcw size={14} />
          </button>
        )}
      </div>

      <p className="text-xs text-slate-500">
        Enter a sentence to visualize transformer self-attention. Each cell
        shows how much one token attends to another.
      </p>

      {/* Input */}
      <div className="relative">
        <input
          type="text"
          value={text}
          onChange={(e) => {
            setText(e.target.value);
            setSelectedRow(null);
            setHoveredWord(null);
          }}
          placeholder="Type a sentence or pick an example below..."
          className="w-full bg-[rgba(255,255,255,0.03)] border border-[rgba(0,212,255,0.08)] rounded-xl px-4 py-3 pr-10 text-sm text-slate-200 placeholder:text-slate-600 focus:outline-none focus:border-cyan-500/30 transition-colors"
          style={{ fontFamily: "var(--font-mono)" }}
        />
        <Type
          size={14}
          className="absolute top-1/2 right-3 -translate-y-1/2 text-slate-600"
        />
      </div>

      {/* Example sentences */}
      {!text && (
        <div className="flex flex-wrap gap-2">
          <span className="text-[10px] font-mono text-slate-600 self-center">
            TRY:
          </span>
          {DEMO_SENTENCES.map((sentence, i) => (
            <button
              key={i}
              onClick={() => setText(sentence)}
              className="text-[10px] font-mono text-cyan-500/50 hover:text-cyan-400 border border-[rgba(0,212,255,0.06)] hover:border-cyan-500/20 rounded-lg px-2.5 py-1 transition-all cursor-pointer truncate max-w-[220px]"
            >
              {sentence}
            </button>
          ))}
        </div>
      )}

      {/* Attention visualization */}
      <AnimatePresence>
        {tokens.length > 0 && currentAttention.length > 0 && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="space-y-4"
          >
            {/* Head selector tabs */}
            <div className="flex items-center gap-2">
              <Brain size={12} className="text-cyan-500/50 shrink-0" />
              <span className="text-[10px] font-mono text-slate-600 tracking-wider shrink-0">
                HEAD:
              </span>
              <div className="flex gap-1.5 flex-wrap">
                {HEAD_LABELS.map((head, i) => (
                  <button
                    key={i}
                    onClick={() => {
                      setActiveHead(i);
                      setHoveredCell(null);
                      setSelectedRow(null);
                    }}
                    className={`text-[10px] font-mono px-2.5 py-1 rounded-lg transition-all cursor-pointer border ${
                      activeHead === i
                        ? "text-cyan-300 bg-cyan-500/10 border-cyan-500/30"
                        : "text-slate-500 border-[rgba(255,255,255,0.05)] hover:text-slate-300 hover:border-slate-600"
                    }`}
                  >
                    {head.name}
                    <span className="text-[8px] ml-1 opacity-50">
                      {head.desc}
                    </span>
                  </button>
                ))}
              </div>
            </div>

            {/* Heatmap */}
            <div className="space-y-2">
              <div className="flex items-center gap-2 text-[10px] font-mono text-slate-600">
                <Grid3x3 size={11} />
                <span className="tracking-wider">
                  ATTENTION MATRIX (Q x K)
                </span>
              </div>

              <div className="overflow-x-auto -mx-2 px-2 pb-2">
                <div
                  className="inline-block min-w-fit"
                  style={{ fontFamily: "var(--font-mono)" }}
                >
                  {/* Column headers (keys) */}
                  <div className="flex">
                    <div className="w-20 shrink-0" />
                    {tokens.map((token, j) => (
                      <div
                        key={j}
                        className="w-14 shrink-0 text-center text-[9px] text-slate-500 truncate px-0.5 pb-1"
                        title={token}
                      >
                        {token.length > 6
                          ? token.slice(0, 5) + "\u2026"
                          : token}
                      </div>
                    ))}
                  </div>

                  {/* Rows (queries) */}
                  {tokens.map((queryToken, i) => (
                    <div
                      key={i}
                      className={`flex items-center group transition-colors rounded ${
                        selectedRow === i
                          ? "bg-cyan-500/5"
                          : ""
                      }`}
                    >
                      {/* Row label */}
                      <button
                        onClick={() =>
                          setSelectedRow(selectedRow === i ? null : i)
                        }
                        className={`w-20 shrink-0 text-right pr-2 text-[10px] truncate cursor-pointer transition-colors ${
                          selectedRow === i
                            ? "text-cyan-400"
                            : "text-slate-500 hover:text-slate-300"
                        }`}
                        title={`Click to highlight attention for "${queryToken}"`}
                      >
                        {queryToken}
                      </button>

                      {/* Attention cells */}
                      {currentAttention[i].map((weight, j) => {
                        const isHovered =
                          hoveredCell?.row === i && hoveredCell?.col === j;
                        const isRowSelected = selectedRow === i;
                        const dimmed =
                          selectedRow !== null && selectedRow !== i;

                        return (
                          <div
                            key={j}
                            className="w-14 h-10 shrink-0 p-0.5"
                          >
                            <motion.div
                              className={`w-full h-full rounded-sm cursor-crosshair relative transition-opacity ${
                                dimmed ? "opacity-25" : "opacity-100"
                              }`}
                              style={{
                                backgroundColor: attentionToColor(weight),
                              }}
                              whileHover={{ scale: 1.15 }}
                              onMouseEnter={() =>
                                setHoveredCell({ row: i, col: j, weight })
                              }
                              onMouseLeave={() => setHoveredCell(null)}
                            >
                              {/* Weight label on selected row */}
                              {isRowSelected && (
                                <span className="absolute inset-0 flex items-center justify-center text-[8px] text-white/80 font-mono">
                                  {(weight * 100).toFixed(0)}%
                                </span>
                              )}

                              {/* Tooltip on hover */}
                              {isHovered && (
                                <motion.div
                                  initial={{ opacity: 0, y: 4 }}
                                  animate={{ opacity: 1, y: 0 }}
                                  className="absolute -top-10 left-1/2 -translate-x-1/2 bg-slate-900/95 border border-slate-700/50 rounded-md px-2 py-1 text-[9px] font-mono whitespace-nowrap z-20 pointer-events-none"
                                >
                                  <span className="text-cyan-400">
                                    {tokens[i]}
                                  </span>
                                  <span className="text-slate-600">
                                    {" \u2192 "}
                                  </span>
                                  <span className="text-violet-400">
                                    {tokens[j]}
                                  </span>
                                  <span className="text-slate-400 ml-1">
                                    {(weight * 100).toFixed(1)}%
                                  </span>
                                </motion.div>
                              )}
                            </motion.div>
                          </div>
                        );
                      })}
                    </div>
                  ))}
                </div>
              </div>

              {/* Color scale legend */}
              <div className="flex items-center gap-2 text-[9px] font-mono text-slate-600">
                <span>Low</span>
                <div className="flex h-2 rounded-full overflow-hidden flex-1 max-w-[120px]">
                  {Array.from({ length: 20 }, (_, i) => (
                    <div
                      key={i}
                      className="flex-1"
                      style={{
                        backgroundColor: attentionToColor(i / 19),
                      }}
                    />
                  ))}
                </div>
                <span>High</span>
              </div>
            </div>

            {/* Text attention view */}
            <div className="space-y-2">
              <span className="text-[10px] font-mono text-slate-600 tracking-wider flex items-center gap-2">
                <Type size={11} />
                TOKEN ATTENTION VIEW
              </span>
              <p className="text-[10px] text-slate-600">
                Hover a word to see its attention distribution across the
                sentence.
              </p>
              <div
                className="flex flex-wrap gap-1.5 p-3 rounded-xl bg-[rgba(255,255,255,0.02)] border border-[rgba(255,255,255,0.03)] min-h-[48px]"
                style={{ fontFamily: "var(--font-mono)" }}
              >
                {tokens.map((token, idx) => {
                  const isSource = hoveredWord === idx;
                  let bgOpacity = 0;
                  let borderHighlight = false;

                  if (hoveredWord !== null && hoveredWord !== idx) {
                    const weight = currentAttention[hoveredWord]?.[idx] ?? 0;
                    bgOpacity = attentionToHighlightOpacity(weight);
                  }
                  if (isSource) {
                    borderHighlight = true;
                  }

                  return (
                    <motion.span
                      key={`${idx}-${token}`}
                      className={`relative px-2 py-1 rounded text-sm transition-all cursor-pointer select-none ${
                        borderHighlight
                          ? "border border-cyan-400/60 text-cyan-300"
                          : hoveredWord !== null
                          ? "text-slate-300"
                          : "text-slate-400 hover:text-slate-200"
                      }`}
                      style={
                        hoveredWord !== null && !isSource
                          ? {
                              backgroundColor: `rgba(139, 92, 246, ${bgOpacity * 0.4})`,
                              borderColor: `rgba(139, 92, 246, ${bgOpacity * 0.5})`,
                              borderWidth: "1px",
                              borderStyle: "solid",
                            }
                          : {}
                      }
                      onMouseEnter={() => setHoveredWord(idx)}
                      onMouseLeave={() => setHoveredWord(null)}
                      whileHover={{ scale: 1.05 }}
                    >
                      {token}

                      {/* Attention weight label when another word is hovered */}
                      {hoveredWord !== null &&
                        hoveredWord !== idx &&
                        currentAttention[hoveredWord] && (
                          <span className="absolute -top-4 left-1/2 -translate-x-1/2 text-[8px] text-violet-400/80 font-mono whitespace-nowrap">
                            {(
                              currentAttention[hoveredWord][idx] * 100
                            ).toFixed(0)}
                            %
                          </span>
                        )}
                    </motion.span>
                  );
                })}
              </div>
            </div>

            {/* Stats row */}
            <div className="flex flex-wrap gap-4 text-[10px] font-mono text-slate-600">
              <span>Tokens: {tokens.length}</span>
              <span>Heads: {NUM_HEADS}</span>
              <span>d_model: {D_MODEL}</span>
              <span>d_k: {D_K}</span>
              <span>
                Matrix: {tokens.length}x{tokens.length}
              </span>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}
