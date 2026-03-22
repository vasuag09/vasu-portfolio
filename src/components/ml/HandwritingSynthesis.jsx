import React, { useRef, useState, useCallback, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { PenTool, Type, RotateCcw, Gauge } from "lucide-react";

/**
 * Handwriting Synthesis Demo — type text and watch it rendered as
 * realistic handwriting on a canvas, stroke by stroke.
 *
 * Procedural approach: each letter is defined as an array of strokes
 * (arrays of control points). Quadratic bezier curves produce smooth
 * rendering, and per-character randomization adds natural variation.
 */

// ──────────────────────────────────────────────
// Style presets — control randomness magnitude
// ──────────────────────────────────────────────
const STYLE_PRESETS = {
  Neat: {
    posJitter: 0.4,
    angleJitter: 0.02,
    widthBase: 1.8,
    widthVar: 0.2,
    baselineWobble: 0.3,
    slant: 0.0,
    letterSpacingMul: 1.0,
  },
  Casual: {
    posJitter: 1.0,
    angleJitter: 0.06,
    widthBase: 2.0,
    widthVar: 0.5,
    baselineWobble: 0.8,
    slant: 0.08,
    letterSpacingMul: 0.95,
  },
  Messy: {
    posJitter: 2.2,
    angleJitter: 0.12,
    widthBase: 2.2,
    widthVar: 1.0,
    baselineWobble: 1.8,
    slant: 0.15,
    letterSpacingMul: 0.9,
  },
};

// ──────────────────────────────────────────────
// Letter definitions: each letter is an array of strokes.
// Each stroke is an array of {x, y} points in a 10x14 cell
// (origin top-left, baseline at y=12).
// Points are connected with quadratic bezier curves.
// ──────────────────────────────────────────────
const LETTER_WIDTH = 10;
const LETTER_HEIGHT = 14;
const BASELINE = 12;

const LETTERS = {
  a: [
    [{ x: 7, y: 5 }, { x: 5, y: 4 }, { x: 3, y: 5 }, { x: 2, y: 8 }, { x: 3, y: 11 }, { x: 5, y: 12 }, { x: 7, y: 11 }],
    [{ x: 7, y: 5 }, { x: 7, y: 12 }],
  ],
  b: [
    [{ x: 2, y: 1 }, { x: 2, y: 12 }],
    [{ x: 2, y: 7 }, { x: 3, y: 5 }, { x: 5, y: 4 }, { x: 7, y: 5 }, { x: 8, y: 8 }, { x: 7, y: 11 }, { x: 5, y: 12 }, { x: 2, y: 11 }],
  ],
  c: [
    [{ x: 8, y: 5 }, { x: 6, y: 4 }, { x: 4, y: 4 }, { x: 2, y: 6 }, { x: 2, y: 10 }, { x: 4, y: 12 }, { x: 6, y: 12 }, { x: 8, y: 11 }],
  ],
  d: [
    [{ x: 8, y: 1 }, { x: 8, y: 12 }],
    [{ x: 8, y: 11 }, { x: 5, y: 12 }, { x: 3, y: 11 }, { x: 2, y: 8 }, { x: 3, y: 5 }, { x: 5, y: 4 }, { x: 8, y: 5 }],
  ],
  e: [
    [{ x: 2, y: 8 }, { x: 8, y: 8 }, { x: 8, y: 6 }, { x: 6, y: 4 }, { x: 4, y: 4 }, { x: 2, y: 6 }, { x: 2, y: 10 }, { x: 4, y: 12 }, { x: 6, y: 12 }, { x: 8, y: 11 }],
  ],
  f: [
    [{ x: 7, y: 2 }, { x: 6, y: 1 }, { x: 5, y: 1 }, { x: 4, y: 2 }, { x: 4, y: 12 }],
    [{ x: 2, y: 6 }, { x: 6, y: 6 }],
  ],
  g: [
    [{ x: 7, y: 5 }, { x: 5, y: 4 }, { x: 3, y: 5 }, { x: 2, y: 8 }, { x: 3, y: 11 }, { x: 5, y: 12 }, { x: 7, y: 11 }],
    [{ x: 7, y: 5 }, { x: 7, y: 14 }, { x: 5, y: 15 }, { x: 3, y: 14 }],
  ],
  h: [
    [{ x: 2, y: 1 }, { x: 2, y: 12 }],
    [{ x: 2, y: 7 }, { x: 3, y: 5 }, { x: 5, y: 4 }, { x: 7, y: 5 }, { x: 7, y: 12 }],
  ],
  i: [
    [{ x: 4, y: 3 }, { x: 4, y: 3.2 }],
    [{ x: 4, y: 5 }, { x: 4, y: 12 }],
  ],
  j: [
    [{ x: 5, y: 3 }, { x: 5, y: 3.2 }],
    [{ x: 5, y: 5 }, { x: 5, y: 14 }, { x: 4, y: 15 }, { x: 3, y: 14 }],
  ],
  k: [
    [{ x: 2, y: 1 }, { x: 2, y: 12 }],
    [{ x: 7, y: 4 }, { x: 2, y: 8 }],
    [{ x: 3, y: 7 }, { x: 7, y: 12 }],
  ],
  l: [
    [{ x: 4, y: 1 }, { x: 4, y: 12 }],
  ],
  m: [
    [{ x: 1, y: 5 }, { x: 1, y: 12 }],
    [{ x: 1, y: 6 }, { x: 2, y: 5 }, { x: 4, y: 4 }, { x: 5, y: 6 }, { x: 5, y: 12 }],
    [{ x: 5, y: 6 }, { x: 6, y: 5 }, { x: 8, y: 4 }, { x: 9, y: 6 }, { x: 9, y: 12 }],
  ],
  n: [
    [{ x: 2, y: 5 }, { x: 2, y: 12 }],
    [{ x: 2, y: 7 }, { x: 3, y: 5 }, { x: 5, y: 4 }, { x: 7, y: 5 }, { x: 7, y: 12 }],
  ],
  o: [
    [{ x: 5, y: 4 }, { x: 3, y: 4 }, { x: 2, y: 6 }, { x: 2, y: 10 }, { x: 3, y: 12 }, { x: 5, y: 12 }, { x: 7, y: 10 }, { x: 7, y: 6 }, { x: 5, y: 4 }],
  ],
  p: [
    [{ x: 2, y: 5 }, { x: 2, y: 15 }],
    [{ x: 2, y: 7 }, { x: 3, y: 5 }, { x: 5, y: 4 }, { x: 7, y: 5 }, { x: 8, y: 8 }, { x: 7, y: 11 }, { x: 5, y: 12 }, { x: 2, y: 11 }],
  ],
  q: [
    [{ x: 7, y: 5 }, { x: 5, y: 4 }, { x: 3, y: 5 }, { x: 2, y: 8 }, { x: 3, y: 11 }, { x: 5, y: 12 }, { x: 7, y: 11 }],
    [{ x: 7, y: 5 }, { x: 7, y: 15 }],
  ],
  r: [
    [{ x: 2, y: 5 }, { x: 2, y: 12 }],
    [{ x: 2, y: 8 }, { x: 3, y: 5 }, { x: 5, y: 4 }, { x: 7, y: 5 }],
  ],
  s: [
    [{ x: 7, y: 5 }, { x: 5, y: 4 }, { x: 3, y: 4 }, { x: 2, y: 5.5 }, { x: 3, y: 7.5 }, { x: 6, y: 9 }, { x: 7, y: 10.5 }, { x: 6, y: 12 }, { x: 4, y: 12 }, { x: 2, y: 11 }],
  ],
  t: [
    [{ x: 4, y: 2 }, { x: 4, y: 12 }, { x: 5, y: 12 }, { x: 6, y: 11 }],
    [{ x: 2, y: 5 }, { x: 6, y: 5 }],
  ],
  u: [
    [{ x: 2, y: 5 }, { x: 2, y: 10 }, { x: 3, y: 12 }, { x: 5, y: 12 }, { x: 7, y: 10 }],
    [{ x: 7, y: 5 }, { x: 7, y: 12 }],
  ],
  v: [
    [{ x: 1, y: 5 }, { x: 4.5, y: 12 }],
    [{ x: 4.5, y: 12 }, { x: 8, y: 5 }],
  ],
  w: [
    [{ x: 1, y: 5 }, { x: 2.5, y: 12 }],
    [{ x: 2.5, y: 12 }, { x: 4.5, y: 7 }],
    [{ x: 4.5, y: 7 }, { x: 6.5, y: 12 }],
    [{ x: 6.5, y: 12 }, { x: 8, y: 5 }],
  ],
  x: [
    [{ x: 2, y: 5 }, { x: 7, y: 12 }],
    [{ x: 7, y: 5 }, { x: 2, y: 12 }],
  ],
  y: [
    [{ x: 2, y: 5 }, { x: 4.5, y: 9 }],
    [{ x: 7, y: 5 }, { x: 3, y: 14 }, { x: 2, y: 15 }],
  ],
  z: [
    [{ x: 2, y: 5 }, { x: 7, y: 5 }, { x: 2, y: 12 }, { x: 7, y: 12 }],
  ],
  " ": [],
};

// ──────────────────────────────────────────────
// Utility helpers
// ──────────────────────────────────────────────
function jitter(magnitude) {
  return (Math.random() - 0.5) * 2 * magnitude;
}

function getLetterWidth(ch) {
  if (ch === " ") return 6;
  if (ch === "m" || ch === "w") return 12;
  if (ch === "i" || ch === "l" || ch === "j") return 6;
  return LETTER_WIDTH;
}

/**
 * Build the full list of animated segments from the input text.
 * Returns an array of { strokes, offsetX, offsetY, charWidth } per character.
 */
function layoutText(text, canvasWidth, scale, style) {
  const preset = STYLE_PRESETS[style];
  const lineHeight = (LETTER_HEIGHT + 4) * scale;
  const margin = 16;
  const maxWidth = canvasWidth - margin * 2;

  const chars = [];
  let cursorX = margin;
  let cursorY = margin + BASELINE * scale;

  // Simple word-wrapping
  const words = text.split(/( )/);

  for (const word of words) {
    // Measure word width
    let wordWidth = 0;
    for (const ch of word) {
      wordWidth += getLetterWidth(ch) * scale * preset.letterSpacingMul;
    }

    // Wrap if needed (don't wrap spaces at line start)
    if (cursorX + wordWidth > maxWidth && word.trim().length > 0 && cursorX > margin + 1) {
      cursorX = margin;
      cursorY += lineHeight;
    }

    for (const ch of word) {
      const lower = ch.toLowerCase();
      const strokes = LETTERS[lower];
      const charWidth = getLetterWidth(lower) * scale * preset.letterSpacingMul;

      if (strokes !== undefined) {
        // Per-character randomization
        const baselineOff = jitter(preset.baselineWobble) * scale;
        const slantOff = preset.slant * scale * LETTER_HEIGHT;

        const mapped = strokes.map((stroke) =>
          stroke.map((pt) => ({
            x: cursorX + (pt.x * scale) + jitter(preset.posJitter) + (BASELINE - pt.y) / BASELINE * slantOff,
            y: cursorY + ((pt.y - BASELINE) * scale) + jitter(preset.posJitter) + baselineOff,
          }))
        );

        chars.push({
          ch: lower,
          strokes: mapped,
          offsetX: cursorX,
          offsetY: cursorY,
          charWidth,
          strokeWidth: preset.widthBase + jitter(preset.widthVar),
        });
      }

      cursorX += charWidth;
    }
  }

  return chars;
}

/**
 * Compute required canvas height based on text layout.
 */
function computeCanvasHeight(text, canvasWidth, scale, style) {
  const preset = STYLE_PRESETS[style];
  const lineHeight = (LETTER_HEIGHT + 4) * scale;
  const margin = 16;
  const maxWidth = canvasWidth - margin * 2;

  let cursorX = margin;
  let maxY = margin + BASELINE * scale;

  const words = text.split(/( )/);
  for (const word of words) {
    let wordWidth = 0;
    for (const ch of word) {
      wordWidth += getLetterWidth(ch.toLowerCase()) * scale * preset.letterSpacingMul;
    }
    if (cursorX + wordWidth > maxWidth && word.trim().length > 0 && cursorX > margin + 1) {
      cursorX = margin;
      maxY += lineHeight;
    }
    for (const ch of word) {
      cursorX += getLetterWidth(ch.toLowerCase()) * scale * preset.letterSpacingMul;
    }
  }

  return maxY + lineHeight * 0.5;
}

// ──────────────────────────────────────────────
// Drawing helpers
// ──────────────────────────────────────────────
function drawPaperBackground(ctx, w, h) {
  // Off-white base
  ctx.fillStyle = "#0d1117";
  ctx.fillRect(0, 0, w, h);

  // Faint grid
  ctx.strokeStyle = "rgba(0, 212, 255, 0.04)";
  ctx.lineWidth = 0.5;
  const gridSize = 20;
  for (let x = gridSize; x < w; x += gridSize) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, h);
    ctx.stroke();
  }
  for (let y = gridSize; y < h; y += gridSize) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
  }
}

/**
 * Draw a single stroke (array of points) onto the canvas using
 * quadratic bezier curves for smoothness.
 */
function drawStrokeSegment(ctx, points, progress, strokeWidth) {
  if (points.length < 2) return;

  const totalSegments = points.length - 1;
  const segsToDraw = Math.min(totalSegments, Math.floor(progress * totalSegments) + 1);
  const lastSegProgress = (progress * totalSegments) - Math.floor(progress * totalSegments);
  const isPartial = segsToDraw <= totalSegments && progress < 1;

  ctx.lineWidth = strokeWidth;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.strokeStyle = "rgba(220, 235, 255, 0.88)";

  ctx.beginPath();
  ctx.moveTo(points[0].x, points[0].y);

  for (let i = 1; i < segsToDraw; i++) {
    if (i < segsToDraw - 1 || !isPartial) {
      // Full segment — use midpoints as control points for smooth curves
      if (i < points.length - 1) {
        const midX = (points[i].x + points[i + 1].x) / 2;
        const midY = (points[i].y + points[i + 1].y) / 2;
        ctx.quadraticCurveTo(points[i].x, points[i].y, midX, midY);
      } else {
        ctx.lineTo(points[i].x, points[i].y);
      }
    } else {
      // Partial last segment
      const prev = points[i - 1];
      const curr = points[i];
      const partialX = prev.x + (curr.x - prev.x) * lastSegProgress;
      const partialY = prev.y + (curr.y - prev.y) * lastSegProgress;
      ctx.lineTo(partialX, partialY);
    }
  }

  // Handle the very last partial segment if we haven't drawn it
  if (isPartial && segsToDraw <= totalSegments && segsToDraw > 0) {
    const prev = points[segsToDraw - 1];
    const curr = points[Math.min(segsToDraw, points.length - 1)];
    if (segsToDraw < points.length) {
      const partialX = prev.x + (curr.x - prev.x) * lastSegProgress;
      const partialY = prev.y + (curr.y - prev.y) * lastSegProgress;
      ctx.lineTo(partialX, partialY);
    }
  }

  ctx.stroke();
}

// ──────────────────────────────────────────────
// Component
// ──────────────────────────────────────────────
export default function HandwritingSynthesis() {
  const canvasRef = useRef(null);
  const animFrameRef = useRef(null);
  const layoutRef = useRef([]);

  const [text, setText] = useState("hello world");
  const [style, setStyle] = useState("Casual");
  const [speed, setSpeed] = useState(50);
  const [isWriting, setIsWriting] = useState(false);

  const CANVAS_WIDTH = 500;
  const SCALE = 3.2;

  // Compute dynamic canvas height
  const canvasHeight = Math.max(
    80,
    computeCanvasHeight(text || " ", CANVAS_WIDTH, SCALE, style)
  );

  // ── Animate writing ──
  const animate = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");

    const inputText = text.trim() || " ";
    const chars = layoutText(inputText, CANVAS_WIDTH, SCALE, style);
    layoutRef.current = chars;

    // Build a flat list of all strokes across all chars
    const allStrokes = [];
    for (const ch of chars) {
      for (const stroke of ch.strokes) {
        allStrokes.push({ points: stroke, strokeWidth: ch.strokeWidth });
      }
    }

    if (allStrokes.length === 0) {
      drawPaperBackground(ctx, canvas.width, canvas.height);
      setIsWriting(false);
      return;
    }

    // Speed: map slider 1-100 to duration multiplier
    const durationPerStroke = Math.max(8, 200 - speed * 1.8); // ms per stroke
    const totalDuration = allStrokes.length * durationPerStroke;

    let startTime = null;
    setIsWriting(true);

    // Cancel any running animation
    if (animFrameRef.current) {
      cancelAnimationFrame(animFrameRef.current);
    }

    function frame(timestamp) {
      if (!startTime) startTime = timestamp;
      const elapsed = timestamp - startTime;
      const globalProgress = Math.min(1, elapsed / totalDuration);

      // Redraw background
      drawPaperBackground(ctx, canvas.width, canvas.height);

      // Draw each stroke based on progress
      for (let i = 0; i < allStrokes.length; i++) {
        const strokeStart = i / allStrokes.length;
        const strokeEnd = (i + 1) / allStrokes.length;

        if (globalProgress <= strokeStart) continue;

        const strokeProgress = Math.min(
          1,
          (globalProgress - strokeStart) / (strokeEnd - strokeStart)
        );

        drawStrokeSegment(
          ctx,
          allStrokes[i].points,
          strokeProgress,
          allStrokes[i].strokeWidth
        );
      }

      if (globalProgress < 1) {
        animFrameRef.current = requestAnimationFrame(frame);
      } else {
        setIsWriting(false);
        animFrameRef.current = null;
      }
    }

    animFrameRef.current = requestAnimationFrame(frame);
  }, [text, style, speed]);

  // Auto-animate on mount and when text/style changes
  useEffect(() => {
    const timer = setTimeout(animate, 150);
    return () => {
      clearTimeout(timer);
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
    };
  }, [animate]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
    };
  }, []);

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
      className="space-y-4"
    >
      {/* Text input */}
      <div className="space-y-2">
        <label className="flex items-center gap-2 text-[10px] font-mono text-cyan-500/50 tracking-wider uppercase">
          <Type size={10} />
          Input Text
        </label>
        <input
          type="text"
          value={text}
          onChange={(e) => setText(e.target.value)}
          maxLength={120}
          placeholder="Type something..."
          className="w-full bg-[rgba(0,212,255,0.04)] border border-[rgba(0,212,255,0.1)] rounded-lg px-3 py-2 text-sm text-white font-mono placeholder:text-slate-600 focus:outline-none focus:border-cyan-500/30 transition-colors"
          style={{ fontFamily: "var(--font-mono)" }}
        />
      </div>

      {/* Canvas */}
      <div className="relative rounded-xl overflow-hidden border border-[rgba(0,212,255,0.12)]">
        <canvas
          ref={canvasRef}
          width={CANVAS_WIDTH}
          height={canvasHeight}
          className="w-full"
          style={{ maxWidth: CANVAS_WIDTH, display: "block" }}
        />

        {/* Writing indicator */}
        <AnimatePresence>
          {isWriting && (
            <motion.div
              initial={{ opacity: 0, y: 4 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -4 }}
              className="absolute top-2 right-2 flex items-center gap-1.5 bg-[rgba(0,0,0,0.7)] backdrop-blur-sm rounded-md px-2 py-1"
            >
              <PenTool size={10} className="text-cyan-400 animate-pulse" />
              <span className="text-[10px] font-mono text-cyan-400/80">
                writing...
              </span>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Controls row */}
      <div className="flex flex-col sm:flex-row gap-3 items-start sm:items-end">
        {/* Style selector */}
        <div className="space-y-1.5 flex-1 min-w-0">
          <label className="flex items-center gap-1.5 text-[10px] font-mono text-cyan-500/50 tracking-wider uppercase">
            <PenTool size={10} />
            Style
          </label>
          <div className="flex gap-1.5">
            {Object.keys(STYLE_PRESETS).map((preset) => (
              <button
                key={preset}
                onClick={() => setStyle(preset)}
                className={`px-3 py-1.5 rounded-md text-xs font-mono transition-all cursor-pointer ${
                  style === preset
                    ? "bg-cyan-500/15 text-cyan-400 border border-cyan-500/30"
                    : "bg-[rgba(255,255,255,0.03)] text-slate-500 border border-transparent hover:text-slate-300 hover:bg-[rgba(255,255,255,0.05)]"
                }`}
              >
                {preset}
              </button>
            ))}
          </div>
        </div>

        {/* Speed slider */}
        <div className="space-y-1.5 w-full sm:w-36">
          <label className="flex items-center gap-1.5 text-[10px] font-mono text-cyan-500/50 tracking-wider uppercase">
            <Gauge size={10} />
            Speed
          </label>
          <input
            type="range"
            min={1}
            max={100}
            value={speed}
            onChange={(e) => setSpeed(Number(e.target.value))}
            className="w-full h-1.5 bg-[rgba(0,212,255,0.08)] rounded-full appearance-none cursor-pointer
              [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3
              [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-cyan-400
              [&::-webkit-slider-thumb]:shadow-[0_0_6px_rgba(0,212,255,0.4)]"
          />
        </div>

        {/* Replay button */}
        <button
          onClick={animate}
          disabled={isWriting}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-mono transition-all cursor-pointer
            bg-[rgba(0,212,255,0.06)] text-cyan-500/70 border border-[rgba(0,212,255,0.12)]
            hover:bg-[rgba(0,212,255,0.12)] hover:text-cyan-400
            disabled:opacity-40 disabled:cursor-not-allowed"
        >
          <RotateCcw size={12} className={isWriting ? "animate-spin" : ""} />
          Replay
        </button>
      </div>

      {/* Info footer */}
      <p className="text-[10px] font-mono text-slate-600 leading-relaxed">
        Procedural handwriting synthesis — stroke paths with quadratic bezier
        curves, per-character jitter, and baseline wobble. No ML model required.
      </p>
    </motion.div>
  );
}
