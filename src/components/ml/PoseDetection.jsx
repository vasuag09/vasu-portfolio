import React, { useRef, useState, useCallback, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Camera, CameraOff, Eye, EyeOff, Activity } from "lucide-react";

/**
 * Neural Vision Demo — webcam-based motion/edge detection overlay.
 *
 * Privacy-first: camera is only activated on explicit user opt-in.
 * Uses a Sobel edge detection kernel combined with frame differencing
 * to produce a stylized "neural vision" overlay in real-time.
 * No ML model required — pure canvas pixel manipulation.
 */

const CANVAS_WIDTH = 640;
const CANVAS_HEIGHT = 480;

/** Check if user prefers reduced motion */
function prefersReducedMotion() {
  if (typeof window === "undefined") return false;
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

/** Sobel edge detection on grayscale image data */
function sobelEdgeDetect(src, dst, width, height) {
  const gx = [-1, 0, 1, -2, 0, 2, -1, 0, 1];
  const gy = [-1, -2, -1, 0, 0, 0, 1, 2, 1];

  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      let sumX = 0;
      let sumY = 0;
      for (let ky = -1; ky <= 1; ky++) {
        for (let kx = -1; kx <= 1; kx++) {
          const idx = ((y + ky) * width + (x + kx)) * 4;
          const lum = src[idx] * 0.299 + src[idx + 1] * 0.587 + src[idx + 2] * 0.114;
          const ki = (ky + 1) * 3 + (kx + 1);
          sumX += lum * gx[ki];
          sumY += lum * gy[ki];
        }
      }
      const mag = Math.min(255, Math.sqrt(sumX * sumX + sumY * sumY));
      const i = (y * width + x) * 4;
      dst[i] = mag;
      dst[i + 1] = mag;
      dst[i + 2] = mag;
      dst[i + 3] = 255;
    }
  }
}

/** Colorize edge map with cyan/violet gradient based on intensity and position */
function colorizeEdges(edgeData, motionData, width, height) {
  for (let y = 0; y < height; y++) {
    const yRatio = y / height;
    for (let x = 0; x < width; x++) {
      const i = (y * width + x) * 4;
      const edge = edgeData[i];
      const motion = motionData ? motionData[i] : 0;
      const combined = Math.min(255, edge + motion * 1.5);

      if (combined > 30) {
        const xRatio = x / width;
        const blend = (xRatio + yRatio) * 0.5;
        // Cyan (0, 212, 255) -> Violet (139, 92, 246)
        const r = Math.round(0 + blend * 139);
        const g = Math.round(212 - blend * 120);
        const b = Math.round(255 - blend * 9);
        const alpha = Math.min(255, combined * 1.2);
        edgeData[i] = r;
        edgeData[i + 1] = g;
        edgeData[i + 2] = b;
        edgeData[i + 3] = alpha;
      } else {
        edgeData[i] = 0;
        edgeData[i + 1] = 0;
        edgeData[i + 2] = 0;
        edgeData[i + 3] = 255;
      }
    }
  }
}

export default function PoseDetection() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const offscreenRef = useRef(null);
  const prevFrameRef = useRef(null);
  const animFrameRef = useRef(null);
  const streamRef = useRef(null);
  const fpsTimesRef = useRef([]);

  const [cameraActive, setCameraActive] = useState(false);
  const [neuralVision, setNeuralVision] = useState(true);
  const [fps, setFps] = useState(0);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const reducedMotion = prefersReducedMotion();

  // Stop camera and cleanup
  const stopCamera = useCallback(() => {
    if (animFrameRef.current) {
      cancelAnimationFrame(animFrameRef.current);
      animFrameRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    prevFrameRef.current = null;
    fpsTimesRef.current = [];
    setCameraActive(false);
    setFps(0);
  }, []);

  // Start camera
  const startCamera = useCallback(async () => {
    setError(null);
    setLoading(true);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: CANVAS_WIDTH },
          height: { ideal: CANVAS_HEIGHT },
          facingMode: "user",
        },
        audio: false,
      });

      streamRef.current = stream;

      const video = videoRef.current;
      if (!video) {
        stream.getTracks().forEach((t) => t.stop());
        return;
      }

      video.srcObject = stream;
      await video.play();
      setCameraActive(true);
    } catch (err) {
      if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
        setError("Camera permission denied. Please allow camera access and try again.");
      } else if (err.name === "NotFoundError" || err.name === "DevicesNotFoundError") {
        setError("No camera found. Please connect a camera and try again.");
      } else {
        setError(`Camera error: ${err.message}`);
      }
    } finally {
      setLoading(false);
    }
  }, []);

  // Toggle camera
  const toggleCamera = useCallback(() => {
    if (cameraActive) {
      stopCamera();
    } else {
      startCamera();
    }
  }, [cameraActive, stopCamera, startCamera]);

  // Rendering loop
  useEffect(() => {
    if (!cameraActive) return;

    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;

    const ctx = canvas.getContext("2d", { willReadFrequently: true });

    // Create offscreen canvas for edge detection
    if (!offscreenRef.current) {
      offscreenRef.current = document.createElement("canvas");
      offscreenRef.current.width = CANVAS_WIDTH;
      offscreenRef.current.height = CANVAS_HEIGHT;
    }
    const offCtx = offscreenRef.current.getContext("2d", { willReadFrequently: true });

    function renderFrame() {
      if (!streamRef.current) return;

      // FPS tracking
      const now = performance.now();
      fpsTimesRef.current.push(now);
      // Keep only the last second of timestamps
      fpsTimesRef.current = fpsTimesRef.current.filter((t) => now - t < 1000);
      if (fpsTimesRef.current.length > 1) {
        setFps(fpsTimesRef.current.length);
      }

      // Draw video frame to offscreen canvas
      offCtx.drawImage(video, 0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
      const frameData = offCtx.getImageData(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);

      if (neuralVision) {
        // Compute motion data from frame differencing
        let motionData = null;
        if (prevFrameRef.current) {
          motionData = new Uint8ClampedArray(frameData.data.length);
          for (let i = 0; i < frameData.data.length; i += 4) {
            const diff = Math.abs(frameData.data[i] - prevFrameRef.current[i])
              + Math.abs(frameData.data[i + 1] - prevFrameRef.current[i + 1])
              + Math.abs(frameData.data[i + 2] - prevFrameRef.current[i + 2]);
            const val = Math.min(255, diff);
            motionData[i] = val;
            motionData[i + 1] = val;
            motionData[i + 2] = val;
            motionData[i + 3] = 255;
          }
        }

        // Store current frame for next diff
        prevFrameRef.current = new Uint8ClampedArray(frameData.data);

        // Edge detection
        const edgePixels = new Uint8ClampedArray(frameData.data.length);
        sobelEdgeDetect(frameData.data, edgePixels, CANVAS_WIDTH, CANVAS_HEIGHT);

        // Colorize with cyan/violet
        colorizeEdges(edgePixels, motionData, CANVAS_WIDTH, CANVAS_HEIGHT);

        const edgeImage = new ImageData(edgePixels, CANVAS_WIDTH, CANVAS_HEIGHT);
        ctx.putImageData(edgeImage, 0, 0);

        // Draw scanline effect (subtle)
        if (!reducedMotion) {
          ctx.fillStyle = "rgba(0, 0, 0, 0.03)";
          const scanlineOffset = (now * 0.05) % CANVAS_HEIGHT;
          for (let y = 0; y < CANVAS_HEIGHT; y += 4) {
            const adjustedY = (y + scanlineOffset) % CANVAS_HEIGHT;
            ctx.fillRect(0, adjustedY, CANVAS_WIDTH, 1);
          }
        }
      } else {
        // Raw feed — just mirror and draw
        ctx.save();
        ctx.scale(-1, 1);
        ctx.drawImage(video, -CANVAS_WIDTH, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
        ctx.restore();
        prevFrameRef.current = null;
      }

      animFrameRef.current = requestAnimationFrame(renderFrame);
    }

    animFrameRef.current = requestAnimationFrame(renderFrame);

    return () => {
      if (animFrameRef.current) {
        cancelAnimationFrame(animFrameRef.current);
        animFrameRef.current = null;
      }
    };
  }, [cameraActive, neuralVision, reducedMotion]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopCamera();
      offscreenRef.current = null;
    };
  }, [stopCamera]);

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
              Neural Vision
            </h3>
            <p
              className="text-[10px] text-cyan-500/50"
              style={{ fontFamily: "var(--font-mono)" }}
            >
              Edge Detection &middot; Motion Tracking &middot; Real-Time
            </p>
          </div>
        </div>

        {/* FPS Counter */}
        {cameraActive && (
          <motion.div
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            className="flex items-center gap-1.5"
          >
            <Activity size={10} className="text-emerald-400/80" />
            <span
              className="text-[10px] text-emerald-400/80"
              style={{ fontFamily: "var(--font-mono)" }}
            >
              {fps} FPS
            </span>
          </motion.div>
        )}
      </div>

      <p className="text-xs text-slate-500">
        Real-time edge detection and motion tracking using Sobel operators and
        frame differencing. All processing runs locally in your browser.
      </p>

      {/* Video feed area */}
      <div className="relative w-full aspect-video max-w-[640px] mx-auto">
        {/* Hidden video element for capturing stream */}
        <video
          ref={videoRef}
          className="hidden"
          playsInline
          muted
          width={CANVAS_WIDTH}
          height={CANVAS_HEIGHT}
        />

        {/* Canvas for rendering */}
        <canvas
          ref={canvasRef}
          width={CANVAS_WIDTH}
          height={CANVAS_HEIGHT}
          className="w-full h-full rounded-xl border border-[rgba(0,212,255,0.12)] bg-black object-contain"
        />

        {/* Placeholder when camera is off */}
        {!cameraActive && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-4 rounded-xl bg-[rgba(0,0,0,0.6)]">
            <div className="w-16 h-16 rounded-full border border-[rgba(0,212,255,0.15)] flex items-center justify-center">
              <Camera size={24} className="text-cyan-500/40" />
            </div>
            <p
              className="text-xs text-slate-500 text-center max-w-[240px]"
              style={{ fontFamily: "var(--font-mono)" }}
            >
              Camera feed stays on your device. Nothing is recorded or sent to
              any server.
            </p>
          </div>
        )}
      </div>

      {/* Error message */}
      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            className="rounded-lg bg-red-500/10 border border-red-500/20 px-4 py-3"
          >
            <p
              className="text-xs text-red-400/90"
              style={{ fontFamily: "var(--font-mono)" }}
            >
              {error}
            </p>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Controls */}
      <div className="flex items-center gap-3">
        <button
          onClick={toggleCamera}
          disabled={loading}
          className="flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-medium transition-all cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
          style={{
            fontFamily: "var(--font-mono)",
            background: cameraActive
              ? "rgba(239, 68, 68, 0.15)"
              : "linear-gradient(135deg, rgba(0, 212, 255, 0.15), rgba(139, 92, 246, 0.15))",
            border: cameraActive
              ? "1px solid rgba(239, 68, 68, 0.3)"
              : "1px solid rgba(0, 212, 255, 0.2)",
            color: cameraActive ? "rgb(252, 165, 165)" : "rgb(0, 212, 255)",
          }}
        >
          {loading ? (
            <>
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
              >
                <Camera size={14} />
              </motion.div>
              Starting...
            </>
          ) : cameraActive ? (
            <>
              <CameraOff size={14} />
              Stop Camera
            </>
          ) : (
            <>
              <Camera size={14} />
              Enable Camera
            </>
          )}
        </button>

        {cameraActive && (
          <motion.button
            initial={{ opacity: 0, x: -10 }}
            animate={{ opacity: 1, x: 0 }}
            onClick={() => setNeuralVision((v) => !v)}
            className="flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-medium transition-all cursor-pointer"
            style={{
              fontFamily: "var(--font-mono)",
              background: neuralVision
                ? "rgba(0, 212, 255, 0.12)"
                : "rgba(255, 255, 255, 0.05)",
              border: neuralVision
                ? "1px solid rgba(0, 212, 255, 0.25)"
                : "1px solid rgba(255, 255, 255, 0.08)",
              color: neuralVision
                ? "rgb(0, 212, 255)"
                : "rgb(148, 163, 184)",
            }}
          >
            {neuralVision ? (
              <>
                <Eye size={14} />
                Neural Vision
              </>
            ) : (
              <>
                <EyeOff size={14} />
                Raw Feed
              </>
            )}
          </motion.button>
        )}
      </div>
    </motion.div>
  );
}
