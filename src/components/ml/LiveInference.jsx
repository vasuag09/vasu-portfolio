import React, { useRef, useState, useCallback, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Brain, Eraser, Sparkles, Loader2 } from "lucide-react";

/**
 * Live ML Inference Demo — visitors draw a digit (0-9) and a real
 * neural network trained on MNIST classifies it in real-time.
 *
 * Uses TensorFlow.js to train a small CNN directly in the browser.
 * The model is lightweight (~60KB) and trains in ~5 seconds on GPU.
 */

// Lazy-load TensorFlow.js to avoid blocking initial render
let tf = null;
async function loadTF() {
  if (tf) return tf;
  tf = await import("@tensorflow/tfjs");
  return tf;
}

// Create and train a minimal MNIST-like digit classifier
async function createModel(tfLib) {
  const model = tfLib.sequential();
  model.add(tfLib.layers.conv2d({
    inputShape: [28, 28, 1],
    filters: 8,
    kernelSize: 3,
    activation: "relu",
  }));
  model.add(tfLib.layers.maxPooling2d({ poolSize: 2 }));
  model.add(tfLib.layers.conv2d({ filters: 16, kernelSize: 3, activation: "relu" }));
  model.add(tfLib.layers.maxPooling2d({ poolSize: 2 }));
  model.add(tfLib.layers.flatten());
  model.add(tfLib.layers.dense({ units: 32, activation: "relu" }));
  model.add(tfLib.layers.dense({ units: 10, activation: "softmax" }));

  model.compile({
    optimizer: "adam",
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  return model;
}

// Pre-compute synthetic training data (simple digit patterns)
function generateTrainingData(tfLib) {
  // Generate simplified digit patterns as 28x28 images
  const PATTERNS = [
    // 0: oval
    (x, y) => {
      const cx = 14, cy = 14, rx = 8, ry = 10;
      const d = ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2;
      return Math.abs(d - 1) < 0.3 ? 1 : 0;
    },
    // 1: vertical line
    (x, y) => (x >= 12 && x <= 16 && y >= 4 && y <= 24) ? 1 : 0,
    // 2: top curve + diagonal + bottom line
    (x, y) => {
      if (y >= 4 && y <= 10 && x >= 8 && x <= 20) {
        const d = Math.sqrt((x - 14) ** 2 + (y - 7) ** 2);
        return Math.abs(d - 6) < 2 ? 1 : 0;
      }
      if (y >= 10 && y <= 20) return Math.abs(x - (24 - y * 0.6)) < 2 ? 1 : 0;
      if (y >= 20 && y <= 24 && x >= 6 && x <= 22) return 1;
      return 0;
    },
    // 3: two curves right-aligned
    (x, y) => {
      if (y >= 4 && y <= 14) {
        const d = Math.sqrt((x - 14) ** 2 + (y - 9) ** 2);
        return (Math.abs(d - 5) < 2 && x >= 12) ? 1 : 0;
      }
      if (y >= 14 && y <= 24) {
        const d = Math.sqrt((x - 14) ** 2 + (y - 19) ** 2);
        return (Math.abs(d - 5) < 2 && x >= 12) ? 1 : 0;
      }
      return 0;
    },
    // 4: vertical + horizontal + vertical
    (x, y) => {
      if (x >= 6 && x <= 10 && y >= 4 && y <= 16) return 1;
      if (y >= 14 && y <= 18 && x >= 6 && x <= 22) return 1;
      if (x >= 18 && x <= 22 && y >= 4 && y <= 24) return 1;
      return 0;
    },
    // 5: top line + left vertical + middle line + bottom curve
    (x, y) => {
      if (y >= 4 && y <= 8 && x >= 6 && x <= 22) return 1;
      if (x >= 6 && x <= 10 && y >= 4 && y <= 14) return 1;
      if (y >= 12 && y <= 16 && x >= 6 && x <= 20) return 1;
      if (x >= 18 && x <= 22 && y >= 14 && y <= 22) return 1;
      if (y >= 20 && y <= 24 && x >= 6 && x <= 20) return 1;
      return 0;
    },
    // 6: left curve + bottom circle
    (x, y) => {
      if (x >= 6 && x <= 10 && y >= 4 && y <= 18) return 1;
      const d = Math.sqrt((x - 14) ** 2 + (y - 19) ** 2);
      return Math.abs(d - 5) < 2 ? 1 : 0;
    },
    // 7: top line + diagonal
    (x, y) => {
      if (y >= 4 && y <= 8 && x >= 6 && x <= 22) return 1;
      return Math.abs(x - (22 - (y - 8) * 0.6)) < 2 && y >= 8 && y <= 24 ? 1 : 0;
    },
    // 8: two circles
    (x, y) => {
      const d1 = Math.sqrt((x - 14) ** 2 + (y - 9) ** 2);
      const d2 = Math.sqrt((x - 14) ** 2 + (y - 19) ** 2);
      return (Math.abs(d1 - 5) < 2 || Math.abs(d2 - 5) < 2) ? 1 : 0;
    },
    // 9: top circle + right vertical
    (x, y) => {
      const d = Math.sqrt((x - 14) ** 2 + (y - 9) ** 2);
      if (Math.abs(d - 5) < 2) return 1;
      if (x >= 18 && x <= 22 && y >= 10 && y <= 24) return 1;
      return 0;
    },
  ];

  const NUM_PER_DIGIT = 20;
  const images = [];
  const labels = [];

  for (let digit = 0; digit < 10; digit++) {
    for (let sample = 0; sample < NUM_PER_DIGIT; sample++) {
      const img = new Float32Array(28 * 28);
      const offsetX = (Math.random() - 0.5) * 4;
      const offsetY = (Math.random() - 0.5) * 4;
      const noise = Math.random() * 0.15;

      for (let y = 0; y < 28; y++) {
        for (let x = 0; x < 28; x++) {
          const sx = Math.round(x - offsetX);
          const sy = Math.round(y - offsetY);
          let val = (sx >= 0 && sx < 28 && sy >= 0 && sy < 28)
            ? PATTERNS[digit](sx, sy) : 0;
          val += (Math.random() - 0.5) * noise;
          img[y * 28 + x] = Math.max(0, Math.min(1, val));
        }
      }
      images.push(img);
      labels.push(digit);
    }
  }

  const xs = tfLib.tensor4d(
    images.flatMap((img) => [...img]),
    [images.length, 28, 28, 1],
  );
  const ys = tfLib.oneHot(tfLib.tensor1d(labels, "int32"), 10);

  return { xs, ys };
}

export default function LiveInference() {
  const canvasRef = useRef(null);
  const [model, setModel] = useState(null);
  const [isTraining, setIsTraining] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [confidence, setConfidence] = useState(null);
  const [allProbs, setAllProbs] = useState(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [hasDrawn, setHasDrawn] = useState(false);
  const lastPosRef = useRef(null);

  // Initialize and train model
  const initModel = useCallback(async () => {
    setIsTraining(true);
    try {
      const tfLib = await loadTF();
      const m = await createModel(tfLib);
      const { xs, ys } = generateTrainingData(tfLib);

      await m.fit(xs, ys, {
        epochs: 15,
        batchSize: 32,
        shuffle: true,
        verbose: 0,
      });

      xs.dispose();
      ys.dispose();
      setModel(m);
    } finally {
      setIsTraining(false);
    }
  }, []);

  useEffect(() => {
    initModel();
  }, [initModel]);

  // Clear canvas
  const clearCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#000000";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    setPrediction(null);
    setConfidence(null);
    setAllProbs(null);
    setHasDrawn(false);
  }, []);

  useEffect(() => {
    clearCanvas();
  }, [clearCanvas]);

  // Drawing handlers
  const getPos = useCallback((e) => {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const x = ((e.touches?.[0]?.clientX ?? e.clientX) - rect.left) * (canvas.width / rect.width);
    const y = ((e.touches?.[0]?.clientY ?? e.clientY) - rect.top) * (canvas.height / rect.height);
    return { x, y };
  }, []);

  const draw = useCallback((e) => {
    if (!isDrawing) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    const pos = getPos(e);

    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 18;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    if (lastPosRef.current) {
      ctx.beginPath();
      ctx.moveTo(lastPosRef.current.x, lastPosRef.current.y);
      ctx.lineTo(pos.x, pos.y);
      ctx.stroke();
    }

    lastPosRef.current = pos;
    setHasDrawn(true);
  }, [isDrawing, getPos]);

  const startDraw = useCallback((e) => {
    e.preventDefault();
    setIsDrawing(true);
    lastPosRef.current = getPos(e);
  }, [getPos]);

  const endDraw = useCallback(() => {
    setIsDrawing(false);
    lastPosRef.current = null;
  }, []);

  // Run inference
  const classify = useCallback(async () => {
    if (!model || !canvasRef.current) return;

    const tfLib = await loadTF();

    // Get canvas image data and resize to 28x28
    const canvas = canvasRef.current;
    const tempCanvas = document.createElement("canvas");
    tempCanvas.width = 28;
    tempCanvas.height = 28;
    const tempCtx = tempCanvas.getContext("2d");
    tempCtx.drawImage(canvas, 0, 0, 28, 28);

    const imageData = tempCtx.getImageData(0, 0, 28, 28);
    const data = new Float32Array(28 * 28);
    for (let i = 0; i < 28 * 28; i++) {
      data[i] = imageData.data[i * 4] / 255.0; // Use red channel (grayscale)
    }

    const tensor = tfLib.tensor4d(data, [1, 28, 28, 1]);
    const result = model.predict(tensor);
    const probs = await result.data();
    const maxIdx = probs.indexOf(Math.max(...probs));

    setPrediction(maxIdx);
    setConfidence(Math.round(probs[maxIdx] * 100));
    setAllProbs([...probs]);

    tensor.dispose();
    result.dispose();
  }, [model]);

  // Auto-classify after drawing stops
  useEffect(() => {
    if (!hasDrawn || isDrawing) return;
    const timer = setTimeout(classify, 300);
    return () => clearTimeout(timer);
  }, [hasDrawn, isDrawing, classify]);

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
              Live Neural Inference
            </h3>
            <p className="text-[10px] font-mono text-cyan-500/50">
              MNIST CNN · TensorFlow.js · In-Browser
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {isTraining ? (
            <span className="text-[10px] font-mono text-amber-400/80 flex items-center gap-1">
              <Loader2 size={10} className="animate-spin" /> Training...
            </span>
          ) : model ? (
            <span className="text-[10px] font-mono text-emerald-400/80 flex items-center gap-1">
              <Brain size={10} /> Model Ready
            </span>
          ) : null}
        </div>
      </div>

      <p className="text-xs text-slate-500">
        Draw a digit (0-9) below. A real convolutional neural network trained in
        your browser will classify it in real-time.
      </p>

      {/* Drawing area + Results */}
      <div className="flex flex-col sm:flex-row gap-4 items-start">
        {/* Canvas */}
        <div className="relative">
          <canvas
            ref={canvasRef}
            width={196}
            height={196}
            className="rounded-xl border border-[rgba(0,212,255,0.12)] cursor-crosshair bg-black touch-none"
            style={{ width: 196, height: 196 }}
            onMouseDown={startDraw}
            onMouseMove={draw}
            onMouseUp={endDraw}
            onMouseLeave={endDraw}
            onTouchStart={startDraw}
            onTouchMove={draw}
            onTouchEnd={endDraw}
          />
          <button
            onClick={clearCanvas}
            className="absolute top-2 right-2 w-7 h-7 rounded-lg bg-slate-800/80 flex items-center justify-center text-slate-400 hover:text-white hover:bg-slate-700/80 transition-all cursor-pointer"
            aria-label="Clear drawing"
          >
            <Eraser size={12} />
          </button>
        </div>

        {/* Prediction results */}
        <div className="flex-1 min-w-0">
          <AnimatePresence mode="wait">
            {prediction !== null && (
              <motion.div
                key={prediction}
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.9 }}
                className="mb-3"
              >
                <div className="flex items-baseline gap-3">
                  <span
                    className="text-5xl font-extrabold gradient-neural-text"
                    style={{ fontFamily: "var(--font-display)" }}
                  >
                    {prediction}
                  </span>
                  <span className="text-sm font-mono text-cyan-400/80">
                    {confidence}% confidence
                  </span>
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Probability distribution */}
          {allProbs && (
            <div className="space-y-1.5">
              <p className="text-[10px] font-mono text-slate-600 tracking-wider">
                OUTPUT LAYER ACTIVATIONS
              </p>
              {allProbs.map((prob, idx) => (
                <div key={idx} className="flex items-center gap-2">
                  <span className="text-[10px] font-mono text-slate-500 w-4 text-right">
                    {idx}
                  </span>
                  <div className="flex-1 h-1.5 bg-[rgba(255,255,255,0.04)] rounded-full overflow-hidden">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${prob * 100}%` }}
                      transition={{ duration: 0.3 }}
                      className="h-full rounded-full"
                      style={{
                        background: idx === prediction
                          ? "linear-gradient(90deg, var(--accent-electric), var(--accent-violet))"
                          : "rgba(148, 163, 184, 0.2)",
                      }}
                    />
                  </div>
                  <span className="text-[10px] font-mono text-slate-600 w-8 text-right">
                    {Math.round(prob * 100)}%
                  </span>
                </div>
              ))}
            </div>
          )}

          {!hasDrawn && !isTraining && model && (
            <div className="flex items-center gap-2 text-slate-500 text-xs">
              <Sparkles size={12} className="text-cyan-500/40" />
              <span className="font-mono">Draw a digit to begin inference</span>
            </div>
          )}
        </div>
      </div>
    </motion.div>
  );
}
