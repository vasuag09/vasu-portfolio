import React, { useRef, useState, useCallback, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Upload, Image, Sliders, Layers } from "lucide-react";

/**
 * Image Segmentation Demo — upload an image and see it segmented
 * into color-based regions using HSL clustering with flood-fill.
 *
 * Pure Canvas 2D — no ML model needed, just color-space math
 * that mimics semantic segmentation output.
 */

const MAX_CANVAS_WIDTH = 400;
const MAX_PROCESS_DIM = 200; // scale down for processing speed
const OVERLAY_ALPHA = 0.4;

// Distinct overlay colors for segment regions (HSL)
const SEGMENT_COLORS = [
  [0, 100, 60],    // red
  [120, 80, 50],   // green
  [210, 90, 55],   // blue
  [45, 95, 55],    // orange
  [280, 80, 60],   // purple
  [170, 80, 45],   // teal
  [330, 85, 60],   // pink
  [60, 90, 50],    // yellow
  [195, 85, 50],   // cyan
  [30, 90, 50],    // amber
  [240, 70, 55],   // indigo
  [150, 70, 45],   // emerald
];

// Convert RGB to HSL
function rgbToHsl(r, g, b) {
  r /= 255;
  g /= 255;
  b /= 255;
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const l = (max + min) / 2;
  let h = 0;
  let s = 0;

  if (max !== min) {
    const d = max - min;
    s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
    switch (max) {
      case r:
        h = ((g - b) / d + (g < b ? 6 : 0)) / 6;
        break;
      case g:
        h = ((b - r) / d + 2) / 6;
        break;
      case b:
        h = ((r - g) / d + 4) / 6;
        break;
    }
  }

  return [h * 360, s * 100, l * 100];
}

// Check if two HSL colors are similar within a threshold
function hslSimilar(a, b, threshold) {
  // Circular hue distance
  let hueDiff = Math.abs(a[0] - b[0]);
  if (hueDiff > 180) hueDiff = 360 - hueDiff;
  const satDiff = Math.abs(a[1] - b[1]);
  const lightDiff = Math.abs(a[2] - b[2]);

  // Weight hue more when saturation is high
  const satWeight = Math.max(a[1], b[1]) / 100;
  const hueWeight = satWeight * 2.0;

  const distance = hueDiff * hueWeight + satDiff * 0.5 + lightDiff * 0.8;
  return distance < threshold;
}

// Flood-fill segmentation on HSL image data
function segmentImage(imageData, width, height, threshold) {
  const pixels = imageData.data;
  const totalPixels = width * height;
  const labels = new Int32Array(totalPixels).fill(-1);
  const hslData = new Array(totalPixels);

  // Pre-compute HSL for all pixels
  for (let i = 0; i < totalPixels; i++) {
    const idx = i * 4;
    hslData[i] = rgbToHsl(pixels[idx], pixels[idx + 1], pixels[idx + 2]);
  }

  let currentLabel = 0;
  const segmentSizes = [];

  // Flood-fill based segmentation
  for (let i = 0; i < totalPixels; i++) {
    if (labels[i] !== -1) continue;

    // BFS flood fill
    const queue = [i];
    labels[i] = currentLabel;
    let size = 0;

    while (queue.length > 0) {
      const pos = queue.pop();
      size++;
      const x = pos % width;
      const y = Math.floor(pos / width);
      const refHsl = hslData[pos];

      // 4-connected neighbors
      const neighbors = [];
      if (x > 0) neighbors.push(pos - 1);
      if (x < width - 1) neighbors.push(pos + 1);
      if (y > 0) neighbors.push(pos - width);
      if (y < height - 1) neighbors.push(pos + width);

      for (const n of neighbors) {
        if (labels[n] === -1 && hslSimilar(refHsl, hslData[n], threshold)) {
          labels[n] = currentLabel;
          queue.push(n);
        }
      }
    }

    segmentSizes.push(size);
    currentLabel++;
  }

  // Merge tiny segments (< 1% of image) into nearest neighbor
  const minSize = totalPixels * 0.01;
  const mergedLabels = new Int32Array(labels);

  for (let seg = 0; seg < segmentSizes.length; seg++) {
    if (segmentSizes[seg] >= minSize) continue;

    for (let i = 0; i < totalPixels; i++) {
      if (mergedLabels[i] !== seg) continue;

      const x = i % width;
      const y = Math.floor(i / width);

      // Find nearest non-tiny segment neighbor
      const neighbors = [];
      if (x > 0) neighbors.push(i - 1);
      if (x < width - 1) neighbors.push(i + 1);
      if (y > 0) neighbors.push(i - width);
      if (y < height - 1) neighbors.push(i + width);

      for (const n of neighbors) {
        const nLabel = mergedLabels[n];
        if (nLabel !== seg && segmentSizes[nLabel] >= minSize) {
          mergedLabels[i] = nLabel;
          break;
        }
      }
    }
  }

  // Count unique segments remaining
  const uniqueLabels = new Set(mergedLabels);
  return { labels: mergedLabels, segmentCount: uniqueLabels.size };
}

// Detect boundaries between segments
function detectBoundaries(labels, width, height) {
  const boundaries = new Uint8Array(width * height);

  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const idx = y * width + x;
      const label = labels[idx];
      if (
        labels[idx - 1] !== label ||
        labels[idx + 1] !== label ||
        labels[idx - width] !== label ||
        labels[idx + width] !== label
      ) {
        boundaries[idx] = 1;
      }
    }
  }

  return boundaries;
}

// Generate a sample image on canvas (colored rectangles composition)
function generateSampleImage(type) {
  const canvas = document.createElement("canvas");
  canvas.width = 200;
  canvas.height = 200;
  const ctx = canvas.getContext("2d");

  if (type === "geometric") {
    ctx.fillStyle = "#1a1a2e";
    ctx.fillRect(0, 0, 200, 200);
    ctx.fillStyle = "#e94560";
    ctx.fillRect(20, 30, 70, 80);
    ctx.fillStyle = "#0f3460";
    ctx.fillRect(100, 20, 80, 60);
    ctx.fillStyle = "#16213e";
    ctx.fillRect(60, 120, 100, 60);
    ctx.fillStyle = "#533483";
    ctx.beginPath();
    ctx.arc(50, 160, 30, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "#e94560";
    ctx.beginPath();
    ctx.arc(160, 130, 25, 0, Math.PI * 2);
    ctx.fill();
  } else if (type === "gradient") {
    const grd1 = ctx.createLinearGradient(0, 0, 200, 0);
    grd1.addColorStop(0, "#00d4ff");
    grd1.addColorStop(1, "#7c3aed");
    ctx.fillStyle = grd1;
    ctx.fillRect(0, 0, 200, 100);
    const grd2 = ctx.createLinearGradient(0, 100, 200, 200);
    grd2.addColorStop(0, "#f59e0b");
    grd2.addColorStop(1, "#ef4444");
    ctx.fillStyle = grd2;
    ctx.fillRect(0, 100, 200, 100);
    ctx.fillStyle = "#10b981";
    ctx.beginPath();
    ctx.arc(100, 100, 40, 0, Math.PI * 2);
    ctx.fill();
  } else if (type === "landscape") {
    // Sky
    const sky = ctx.createLinearGradient(0, 0, 0, 120);
    sky.addColorStop(0, "#1e3a5f");
    sky.addColorStop(1, "#e07b39");
    ctx.fillStyle = sky;
    ctx.fillRect(0, 0, 200, 120);
    // Sun
    ctx.fillStyle = "#fbbf24";
    ctx.beginPath();
    ctx.arc(150, 60, 25, 0, Math.PI * 2);
    ctx.fill();
    // Mountains
    ctx.fillStyle = "#374151";
    ctx.beginPath();
    ctx.moveTo(0, 120);
    ctx.lineTo(60, 60);
    ctx.lineTo(120, 120);
    ctx.fill();
    ctx.fillStyle = "#4b5563";
    ctx.beginPath();
    ctx.moveTo(80, 120);
    ctx.lineTo(160, 50);
    ctx.lineTo(200, 100);
    ctx.lineTo(200, 120);
    ctx.fill();
    // Ground
    ctx.fillStyle = "#065f46";
    ctx.fillRect(0, 120, 200, 80);
  }

  return canvas;
}

const SAMPLE_IMAGES = [
  { type: "geometric", label: "Shapes" },
  { type: "gradient", label: "Gradient" },
  { type: "landscape", label: "Scene" },
];

export default function ImageSegmentation() {
  const canvasRef = useRef(null);
  const displayCanvasRef = useRef(null);
  const fileInputRef = useRef(null);
  const [imageLoaded, setImageLoaded] = useState(false);
  const [processing, setProcessing] = useState(false);
  const [threshold, setThreshold] = useState(30);
  const [showBoundaries, setShowBoundaries] = useState(true);
  const [segmentCount, setSegmentCount] = useState(0);
  const [isDragOver, setIsDragOver] = useState(false);
  const [sourceImageData, setSourceImageData] = useState(null);
  const [canvasDims, setCanvasDims] = useState({ width: 0, height: 0 });

  // Load image onto hidden processing canvas and store raw data
  const loadImage = useCallback((source) => {
    setProcessing(true);

    const processSource = (img, imgWidth, imgHeight) => {
      // Scale down for display
      let displayW = imgWidth;
      let displayH = imgHeight;
      if (displayW > MAX_CANVAS_WIDTH) {
        const ratio = MAX_CANVAS_WIDTH / displayW;
        displayW = MAX_CANVAS_WIDTH;
        displayH = Math.round(imgHeight * ratio);
      }

      // Scale down for processing
      let procW = displayW;
      let procH = displayH;
      if (procW > MAX_PROCESS_DIM) {
        const ratio = MAX_PROCESS_DIM / procW;
        procW = Math.round(displayW * ratio);
        procH = Math.round(displayH * ratio);
      }

      const procCanvas = document.createElement("canvas");
      procCanvas.width = procW;
      procCanvas.height = procH;
      const procCtx = procCanvas.getContext("2d");

      if (source instanceof HTMLCanvasElement) {
        procCtx.drawImage(source, 0, 0, procW, procH);
      } else {
        procCtx.drawImage(img, 0, 0, procW, procH);
      }

      const procData = procCtx.getImageData(0, 0, procW, procH);

      setCanvasDims({ width: displayW, height: displayH });
      setSourceImageData({
        imageData: procData,
        procWidth: procW,
        procHeight: procH,
        displayWidth: displayW,
        displayHeight: displayH,
        sourceElement: source instanceof HTMLCanvasElement ? source : img,
      });
      setImageLoaded(true);
    };

    if (source instanceof HTMLCanvasElement) {
      processSource(source, source.width, source.height);
    } else {
      // File or URL
      const img = new window.Image();
      img.onload = () => {
        processSource(img, img.naturalWidth, img.naturalHeight);
      };
      img.onerror = () => {
        setProcessing(false);
      };
      if (source instanceof File) {
        img.src = URL.createObjectURL(source);
      } else {
        img.src = source;
      }
    }
  }, []);

  // Run segmentation whenever source data or threshold changes
  useEffect(() => {
    if (!sourceImageData) return;

    // Use requestAnimationFrame to avoid blocking UI
    const frameId = requestAnimationFrame(() => {
      setProcessing(true);
      const { imageData, procWidth, procHeight, displayWidth, displayHeight, sourceElement } =
        sourceImageData;

      const { labels, segmentCount: count } = segmentImage(
        imageData,
        procWidth,
        procHeight,
        threshold
      );

      const boundaries = showBoundaries
        ? detectBoundaries(labels, procWidth, procHeight)
        : null;

      // Draw result on display canvas
      const displayCanvas = displayCanvasRef.current;
      if (!displayCanvas) return;

      displayCanvas.width = displayWidth;
      displayCanvas.height = displayHeight;
      const ctx = displayCanvas.getContext("2d");

      // Draw original image at display size
      ctx.drawImage(sourceElement, 0, 0, displayWidth, displayHeight);

      // Draw overlay at processing resolution, then scale up
      const overlayCanvas = document.createElement("canvas");
      overlayCanvas.width = procWidth;
      overlayCanvas.height = procHeight;
      const oCtx = overlayCanvas.getContext("2d");
      const overlayData = oCtx.createImageData(procWidth, procHeight);

      for (let i = 0; i < procWidth * procHeight; i++) {
        const label = labels[i];
        const colorIdx = label % SEGMENT_COLORS.length;
        const [h, s, l] = SEGMENT_COLORS[colorIdx];

        // Convert HSL to RGB for overlay
        const c = (1 - Math.abs((2 * l) / 100 - 1)) * (s / 100);
        const x = c * (1 - Math.abs(((h / 60) % 2) - 1));
        const m = l / 100 - c / 2;
        let r1, g1, b1;
        if (h < 60) [r1, g1, b1] = [c, x, 0];
        else if (h < 120) [r1, g1, b1] = [x, c, 0];
        else if (h < 180) [r1, g1, b1] = [0, c, x];
        else if (h < 240) [r1, g1, b1] = [0, x, c];
        else if (h < 300) [r1, g1, b1] = [x, 0, c];
        else [r1, g1, b1] = [c, 0, x];

        const idx = i * 4;
        overlayData.data[idx] = Math.round((r1 + m) * 255);
        overlayData.data[idx + 1] = Math.round((g1 + m) * 255);
        overlayData.data[idx + 2] = Math.round((b1 + m) * 255);
        overlayData.data[idx + 3] = Math.round(OVERLAY_ALPHA * 255);

        // Draw boundaries as bright white
        if (boundaries && boundaries[i]) {
          overlayData.data[idx] = 255;
          overlayData.data[idx + 1] = 255;
          overlayData.data[idx + 2] = 255;
          overlayData.data[idx + 3] = 200;
        }
      }

      oCtx.putImageData(overlayData, 0, 0);

      // Scale overlay to display size
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(overlayCanvas, 0, 0, displayWidth, displayHeight);

      setSegmentCount(count);
      setProcessing(false);
    });

    return () => cancelAnimationFrame(frameId);
  }, [sourceImageData, threshold, showBoundaries]);

  const handleFile = useCallback(
    (file) => {
      if (!file || !file.type.startsWith("image/")) return;
      loadImage(file);
    },
    [loadImage]
  );

  const handleDrop = useCallback(
    (e) => {
      e.preventDefault();
      setIsDragOver(false);
      const file = e.dataTransfer?.files?.[0];
      handleFile(file);
    },
    [handleFile]
  );

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback(() => {
    setIsDragOver(false);
  }, []);

  const handleSample = useCallback(
    (type) => {
      const sampleCanvas = generateSampleImage(type);
      loadImage(sampleCanvas);
    },
    [loadImage]
  );

  const handleReset = useCallback(() => {
    setImageLoaded(false);
    setSourceImageData(null);
    setSegmentCount(0);
    setProcessing(false);
    const canvas = displayCanvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
  }, []);

  return (
    <div className="space-y-4">
      {/* Upload area / Canvas display */}
      <AnimatePresence mode="wait">
        {!imageLoaded ? (
          <motion.div
            key="upload"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="space-y-3"
          >
            {/* Drop zone */}
            <div
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onClick={() => fileInputRef.current?.click()}
              className={`
                relative border-2 border-dashed rounded-lg p-8 text-center cursor-pointer
                transition-all duration-200
                ${
                  isDragOver
                    ? "border-cyan-400 bg-cyan-500/5"
                    : "border-slate-700/50 hover:border-cyan-500/30 hover:bg-cyan-500/[0.02]"
                }
              `}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={(e) => handleFile(e.target.files?.[0])}
              />

              <motion.div
                animate={isDragOver ? { scale: 1.05, y: -2 } : { scale: 1, y: 0 }}
                className="flex flex-col items-center gap-3"
              >
                <div className="w-10 h-10 rounded-lg bg-cyan-500/10 flex items-center justify-center">
                  <Upload size={18} className="text-cyan-500/60" />
                </div>
                <div>
                  <p
                    className="text-xs text-slate-300"
                    style={{ fontFamily: "var(--font-display)" }}
                  >
                    Drop an image or click to upload
                  </p>
                  <p className="text-[10px] font-mono text-slate-600 mt-1">
                    PNG, JPG, WebP supported
                  </p>
                </div>
              </motion.div>
            </div>

            {/* Sample images */}
            <div>
              <p className="text-[10px] font-mono text-slate-600 mb-2">
                Or try a sample:
              </p>
              <div className="flex gap-2">
                {SAMPLE_IMAGES.map(({ type, label }) => (
                  <SampleButton key={type} type={type} label={label} onClick={handleSample} />
                ))}
              </div>
            </div>
          </motion.div>
        ) : (
          <motion.div
            key="canvas"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="space-y-3"
          >
            {/* Canvas with segmented image */}
            <div className="relative flex justify-center">
              <canvas
                ref={displayCanvasRef}
                className="rounded-lg border border-slate-700/30 max-w-full"
                style={{
                  maxWidth: MAX_CANVAS_WIDTH,
                  width: canvasDims.width || "auto",
                  height: "auto",
                }}
              />

              {/* Processing overlay */}
              <AnimatePresence>
                {processing && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="absolute inset-0 flex items-center justify-center bg-black/40 rounded-lg"
                  >
                    <div className="flex items-center gap-2 text-cyan-400 font-mono text-xs">
                      <motion.div
                        animate={{ rotate: 360 }}
                        transition={{ repeat: Infinity, duration: 1, ease: "linear" }}
                      >
                        <Layers size={14} />
                      </motion.div>
                      Segmenting...
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>

            {/* Segment count badge */}
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Image size={12} className="text-cyan-500/50" />
                <span className="text-[10px] font-mono text-slate-500">
                  <span className="text-cyan-400">{segmentCount}</span> segments detected
                </span>
              </div>
              <button
                onClick={handleReset}
                className="text-[10px] font-mono text-slate-600 hover:text-cyan-400 transition-colors cursor-pointer"
              >
                Upload new image
              </button>
            </div>

            {/* Controls */}
            <div className="space-y-3 pt-1">
              {/* Threshold slider */}
              <div className="space-y-1.5">
                <div className="flex items-center justify-between">
                  <label className="flex items-center gap-1.5 text-[10px] font-mono text-slate-500">
                    <Sliders size={10} className="text-cyan-500/50" />
                    Segmentation Threshold
                  </label>
                  <span
                    className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-cyan-500/10"
                    style={{ color: "var(--accent-electric)" }}
                  >
                    {threshold}
                  </span>
                </div>
                <input
                  type="range"
                  min={5}
                  max={80}
                  step={1}
                  value={threshold}
                  onChange={(e) => setThreshold(Number(e.target.value))}
                  className="w-full h-1 appearance-none bg-slate-700/50 rounded-full cursor-pointer
                    [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3
                    [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:rounded-full
                    [&::-webkit-slider-thumb]:bg-cyan-400 [&::-webkit-slider-thumb]:cursor-pointer
                    [&::-moz-range-thumb]:w-3 [&::-moz-range-thumb]:h-3
                    [&::-moz-range-thumb]:rounded-full [&::-moz-range-thumb]:bg-cyan-400
                    [&::-moz-range-thumb]:border-0 [&::-moz-range-thumb]:cursor-pointer"
                />
                <div className="flex justify-between text-[9px] font-mono text-slate-700">
                  <span>Fine</span>
                  <span>Coarse</span>
                </div>
              </div>

              {/* Boundary toggle */}
              <div className="flex items-center justify-between">
                <label className="flex items-center gap-1.5 text-[10px] font-mono text-slate-500">
                  <Layers size={10} className="text-cyan-500/50" />
                  Show Boundaries
                </label>
                <button
                  onClick={() => setShowBoundaries(!showBoundaries)}
                  className={`
                    relative w-8 h-4 rounded-full transition-colors cursor-pointer
                    ${showBoundaries ? "bg-cyan-500/40" : "bg-slate-700/50"}
                  `}
                >
                  <motion.div
                    animate={{ x: showBoundaries ? 16 : 2 }}
                    transition={{ type: "spring", stiffness: 500, damping: 30 }}
                    className={`
                      absolute top-0.5 w-3 h-3 rounded-full
                      ${showBoundaries ? "bg-cyan-400" : "bg-slate-500"}
                    `}
                  />
                </button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Hidden processing canvas */}
      <canvas ref={canvasRef} className="hidden" />
    </div>
  );
}

// Sample image button with preview thumbnail
function SampleButton({ type, label, onClick }) {
  const previewRef = useRef(null);

  useEffect(() => {
    const canvas = previewRef.current;
    if (!canvas) return;
    const source = generateSampleImage(type);
    canvas.width = 48;
    canvas.height = 48;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(source, 0, 0, 48, 48);
  }, [type]);

  return (
    <motion.button
      whileHover={{ scale: 1.05, y: -1 }}
      whileTap={{ scale: 0.97 }}
      onClick={() => onClick(type)}
      className="flex flex-col items-center gap-1 cursor-pointer group"
    >
      <canvas
        ref={previewRef}
        className="w-12 h-12 rounded border border-slate-700/40 group-hover:border-cyan-500/30 transition-colors"
      />
      <span className="text-[9px] font-mono text-slate-600 group-hover:text-cyan-400 transition-colors">
        {label}
      </span>
    </motion.button>
  );
}
