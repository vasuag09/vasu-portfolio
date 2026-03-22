import React, { useRef, useState, useCallback, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Mic, MicOff, Activity, Volume2 } from "lucide-react";

/**
 * Audio Classification Demo — real-time audio analysis using the Web Audio API.
 *
 * Captures microphone input (opt-in, privacy-first) and performs:
 * - Waveform visualization (time domain)
 * - Frequency spectrum visualization (frequency domain)
 * - Heuristic-based audio classification (silence, speech, music, clap, whistle)
 *
 * No ML model required — all analysis done via AnalyserNode and simple heuristics.
 */

const CATEGORIES = [
  { label: "Silence", color: "rgba(100, 116, 139, 0.6)" },
  { label: "Speech", color: "rgba(0, 212, 255, 0.8)" },
  { label: "Music", color: "rgba(139, 92, 246, 0.8)" },
  { label: "Clap/Tap", color: "rgba(251, 191, 36, 0.8)" },
  { label: "Whistle", color: "rgba(52, 211, 153, 0.8)" },
];

const GRADIENT_CYAN = "rgba(0, 212, 255, 0.8)";
const GRADIENT_VIOLET = "rgba(139, 92, 246, 0.8)";

function classifyAudio(analyser, dataArray, freqArray, sampleRate) {
  analyser.getByteTimeDomainData(dataArray);
  analyser.getByteFrequencyData(freqArray);

  // Calculate RMS volume
  let sumSquares = 0;
  for (let i = 0; i < dataArray.length; i++) {
    const normalized = (dataArray[i] - 128) / 128;
    sumSquares += normalized * normalized;
  }
  const rms = Math.sqrt(sumSquares / dataArray.length);

  // Calculate frequency characteristics
  const binWidth = sampleRate / (analyser.fftSize);
  let maxFreqBin = 0;
  let maxFreqValue = 0;
  let activeBins = 0;
  const threshold = 20;

  for (let i = 1; i < freqArray.length; i++) {
    const val = freqArray[i];
    if (val > threshold) {
      activeBins++;
    }
    if (val > maxFreqValue) {
      maxFreqValue = val;
      maxFreqBin = i;
    }
  }

  const dominantFreq = maxFreqBin * binWidth;
  const spectralSpread = activeBins / freqArray.length;

  // Detect sudden volume spikes (clap detection)
  const volumeDb = rms > 0 ? 20 * Math.log10(rms) : -100;

  // Classification probabilities
  const probs = [0, 0, 0, 0, 0]; // silence, speech, music, clap, whistle

  // Silence: low RMS
  if (rms < 0.015) {
    probs[0] = 0.9;
    probs[1] = 0.03;
    probs[2] = 0.03;
    probs[3] = 0.02;
    probs[4] = 0.02;
  } else {
    // Clap/Tap: sudden high energy, wide spectrum
    if (rms > 0.3 && spectralSpread > 0.15) {
      probs[3] = 0.7 + Math.min(rms * 0.3, 0.25);
      probs[1] = 0.1;
      probs[2] = 0.1;
      probs[4] = 0.05;
      probs[0] = 0.05;
    }
    // Whistle: narrow high-frequency peak
    else if (
      dominantFreq > 500 &&
      dominantFreq < 4000 &&
      spectralSpread < 0.08 &&
      maxFreqValue > 100
    ) {
      probs[4] = 0.6 + Math.min(maxFreqValue / 400, 0.35);
      probs[1] = 0.1;
      probs[2] = 0.15;
      probs[3] = 0.05;
      probs[0] = 0.1;
    }
    // Speech: dominant freq 80-300Hz, moderate volume
    else if (
      dominantFreq >= 80 &&
      dominantFreq <= 350 &&
      rms > 0.02 &&
      rms < 0.4 &&
      spectralSpread > 0.03
    ) {
      probs[1] = 0.5 + Math.min(rms * 2, 0.4);
      probs[2] = 0.15;
      probs[0] = 0.15;
      probs[3] = 0.1;
      probs[4] = 0.1;
    }
    // Music: wide frequency spread
    else if (spectralSpread > 0.12 && rms > 0.02) {
      probs[2] = 0.5 + Math.min(spectralSpread * 2, 0.4);
      probs[1] = 0.2;
      probs[0] = 0.1;
      probs[3] = 0.1;
      probs[4] = 0.1;
    }
    // Fallback: distribute based on features
    else {
      probs[0] = Math.max(0.1, 1 - rms * 10);
      probs[1] = rms > 0.02 ? 0.3 : 0.1;
      probs[2] = spectralSpread > 0.05 ? 0.2 : 0.1;
      probs[3] = rms > 0.2 ? 0.15 : 0.05;
      probs[4] = dominantFreq > 500 ? 0.15 : 0.05;
    }
  }

  // Normalize probabilities
  const sum = probs.reduce((a, b) => a + b, 0);
  for (let i = 0; i < probs.length; i++) {
    probs[i] /= sum;
  }

  const topIdx = probs.indexOf(Math.max(...probs));

  return {
    label: CATEGORIES[topIdx].label,
    confidence: Math.round(probs[topIdx] * 100),
    probabilities: probs,
    rms,
    dominantFreq: Math.round(dominantFreq),
    volumeDb: Math.round(volumeDb),
  };
}

export default function AudioClassifier() {
  const waveformCanvasRef = useRef(null);
  const spectrumCanvasRef = useRef(null);
  const audioCtxRef = useRef(null);
  const analyserRef = useRef(null);
  const streamRef = useRef(null);
  const animFrameRef = useRef(null);
  const fpsCounterRef = useRef(null);

  const [isListening, setIsListening] = useState(false);
  const [permissionDenied, setPermissionDenied] = useState(false);
  const [classification, setClassification] = useState(null);
  const [fps, setFps] = useState(0);
  const [sampleRate, setSampleRate] = useState(0);
  const [prefersReducedMotion, setPrefersReducedMotion] = useState(
    () => typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches
  );

  // Listen for reduced motion preference changes
  useEffect(() => {
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    const handler = (e) => setPrefersReducedMotion(e.matches);
    mq.addEventListener("change", handler);
    return () => mq.removeEventListener("change", handler);
  }, []);

  const drawWaveform = useCallback(
    (analyser, dataArray, canvas) => {
      const ctx = canvas.getContext("2d");
      const width = canvas.width;
      const height = canvas.height;

      ctx.clearRect(0, 0, width, height);

      analyser.getByteTimeDomainData(dataArray);

      // Draw waveform
      const gradient = ctx.createLinearGradient(0, 0, width, 0);
      gradient.addColorStop(0, GRADIENT_CYAN);
      gradient.addColorStop(1, GRADIENT_VIOLET);

      ctx.lineWidth = 2;
      ctx.strokeStyle = gradient;
      ctx.beginPath();

      const sliceWidth = width / dataArray.length;
      let x = 0;

      for (let i = 0; i < dataArray.length; i++) {
        const v = dataArray[i] / 128.0;
        const y = (v * height) / 2;

        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
        x += sliceWidth;
      }

      ctx.lineTo(width, height / 2);
      ctx.stroke();

      // Subtle center line
      ctx.strokeStyle = "rgba(100, 116, 139, 0.15)";
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(0, height / 2);
      ctx.lineTo(width, height / 2);
      ctx.stroke();
      ctx.setLineDash([]);
    },
    [],
  );

  const drawSpectrum = useCallback(
    (analyser, freqArray, canvas) => {
      const ctx = canvas.getContext("2d");
      const width = canvas.width;
      const height = canvas.height;

      ctx.clearRect(0, 0, width, height);

      analyser.getByteFrequencyData(freqArray);

      // Only draw the useful frequency range (first ~60% of bins)
      const usefulBins = Math.floor(freqArray.length * 0.6);
      const barCount = Math.min(usefulBins, 64);
      const binsPerBar = Math.floor(usefulBins / barCount);
      const barWidth = (width / barCount) - 1;

      for (let i = 0; i < barCount; i++) {
        // Average bins for this bar
        let sum = 0;
        for (let j = 0; j < binsPerBar; j++) {
          sum += freqArray[i * binsPerBar + j];
        }
        const avg = sum / binsPerBar;
        const barHeight = (avg / 255) * height;

        const t = i / barCount;
        const r = Math.round(0 + t * 139);
        const g = Math.round(212 - t * 120);
        const b = Math.round(255 - t * 9);

        ctx.fillStyle = `rgba(${r}, ${g}, ${b}, 0.75)`;
        ctx.fillRect(
          i * (barWidth + 1),
          height - barHeight,
          barWidth,
          barHeight,
        );
      }
    },
    [],
  );

  const startListening = useCallback(async () => {
    try {
      setPermissionDenied(false);

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      audioCtxRef.current = audioCtx;
      setSampleRate(audioCtx.sampleRate);

      const source = audioCtx.createMediaStreamSource(stream);
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 2048;
      analyser.smoothingTimeConstant = 0.8;
      source.connect(analyser);
      analyserRef.current = analyser;

      const bufferLength = analyser.frequencyBinCount;
      const dataArray = new Uint8Array(analyser.fftSize);
      const freqArray = new Uint8Array(bufferLength);

      setIsListening(true);
      fpsCounterRef.current = { frames: 0, lastTime: performance.now(), fps: 0 };

      const loop = () => {
        // FPS tracking
        const counter = fpsCounterRef.current;
        counter.frames++;
        const now = performance.now();
        if (now - counter.lastTime >= 1000) {
          counter.fps = counter.frames;
          counter.frames = 0;
          counter.lastTime = now;
          setFps(counter.fps);
        }

        // Draw visualizations (skip if reduced motion)
        if (!prefersReducedMotion) {
          if (waveformCanvasRef.current) {
            drawWaveform(analyser, dataArray, waveformCanvasRef.current);
          }
          if (spectrumCanvasRef.current) {
            drawSpectrum(analyser, freqArray, spectrumCanvasRef.current);
          }
        }

        // Classify
        const result = classifyAudio(analyser, dataArray, freqArray, audioCtx.sampleRate);
        setClassification(result);

        animFrameRef.current = requestAnimationFrame(loop);
      };

      animFrameRef.current = requestAnimationFrame(loop);
    } catch (err) {
      if (
        err.name === "NotAllowedError" ||
        err.name === "PermissionDeniedError"
      ) {
        setPermissionDenied(true);
      }
      console.error("Microphone access error:", err);
    }
  }, [prefersReducedMotion, drawWaveform, drawSpectrum]);

  const stopListening = useCallback(() => {
    if (animFrameRef.current) {
      cancelAnimationFrame(animFrameRef.current);
      animFrameRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (audioCtxRef.current) {
      audioCtxRef.current.close();
      audioCtxRef.current = null;
    }
    analyserRef.current = null;
    setIsListening(false);
    setClassification(null);
    setFps(0);
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }
      if (audioCtxRef.current) {
        audioCtxRef.current.close();
      }
    };
  }, []);

  const topCategory =
    classification &&
    classification.probabilities.indexOf(
      Math.max(...classification.probabilities),
    );

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
          <div className="w-3 h-3 rounded-full bg-gradient-to-br from-violet-500 to-purple-500 animate-neural-breathe" />
          <div>
            <h3
              className="text-sm font-semibold text-white"
              style={{ fontFamily: "var(--font-display)" }}
            >
              Audio Classifier
            </h3>
            <p className="text-[10px] font-mono text-cyan-500/50">
              Web Audio API · Real-Time · In-Browser
            </p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          {isListening && (
            <span className="text-[10px] font-mono text-slate-600">
              {fps} FPS · {sampleRate}Hz
            </span>
          )}
          {isListening && (
            <span className="text-[10px] font-mono text-emerald-400/80 flex items-center gap-1">
              <Activity size={10} /> Live
            </span>
          )}
        </div>
      </div>

      <p className="text-xs text-slate-500">
        Analyze ambient audio in real-time. Detects silence, speech, music,
        claps, and whistles using frequency analysis heuristics. No data leaves
        your device.
      </p>

      {/* Microphone toggle */}
      <div className="flex items-center gap-3">
        <button
          onClick={isListening ? stopListening : startListening}
          className={`
            flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-mono
            transition-all cursor-pointer border
            ${
              isListening
                ? "bg-red-500/10 border-red-500/30 text-red-400 hover:bg-red-500/20"
                : "bg-cyan-500/10 border-cyan-500/30 text-cyan-400 hover:bg-cyan-500/20"
            }
          `}
          aria-label={isListening ? "Stop listening" : "Start listening"}
        >
          {isListening ? (
            <>
              <MicOff size={14} /> Stop Listening
            </>
          ) : (
            <>
              <Mic size={14} /> Start Listening
            </>
          )}
        </button>

        {permissionDenied && (
          <span className="text-[10px] font-mono text-red-400/80">
            Microphone permission denied. Please allow access and try again.
          </span>
        )}
      </div>

      {/* Visualizations */}
      <AnimatePresence>
        {isListening && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.3 }}
            className="space-y-3 overflow-hidden"
          >
            {/* Waveform */}
            <div>
              <p className="text-[10px] font-mono text-slate-600 tracking-wider mb-1.5">
                WAVEFORM
              </p>
              <canvas
                ref={waveformCanvasRef}
                width={400}
                height={80}
                className="w-full max-w-[400px] h-[60px] sm:h-[80px] rounded-lg border border-[rgba(0,212,255,0.08)] bg-[rgba(0,0,0,0.3)]"
                aria-label="Audio waveform visualization"
              />
            </div>

            {/* Frequency Spectrum */}
            <div>
              <p className="text-[10px] font-mono text-slate-600 tracking-wider mb-1.5">
                FREQUENCY SPECTRUM
              </p>
              <canvas
                ref={spectrumCanvasRef}
                width={400}
                height={80}
                className="w-full max-w-[400px] h-[60px] sm:h-[80px] rounded-lg border border-[rgba(0,212,255,0.08)] bg-[rgba(0,0,0,0.3)]"
                aria-label="Audio frequency spectrum visualization"
              />
            </div>

            {/* Classification Results */}
            {classification && (
              <div className="space-y-3">
                {/* Current classification */}
                <div className="flex items-center justify-between">
                  <div className="flex items-baseline gap-3">
                    <motion.span
                      key={classification.label}
                      initial={{ opacity: 0, scale: 0.9 }}
                      animate={{ opacity: 1, scale: 1 }}
                      className="text-2xl sm:text-3xl font-extrabold gradient-neural-text"
                      style={{ fontFamily: "var(--font-display)" }}
                    >
                      {classification.label}
                    </motion.span>
                    <span className="text-sm font-mono text-cyan-400/80">
                      {classification.confidence}%
                    </span>
                  </div>
                  <div className="flex items-center gap-3 text-[10px] font-mono text-slate-600">
                    <span className="flex items-center gap-1">
                      <Volume2 size={10} />
                      {classification.volumeDb} dB
                    </span>
                    <span>{classification.dominantFreq} Hz</span>
                  </div>
                </div>

                {/* Probability bars */}
                <div className="space-y-1.5">
                  <p className="text-[10px] font-mono text-slate-600 tracking-wider">
                    CLASSIFICATION PROBABILITIES
                  </p>
                  {CATEGORIES.map((cat, idx) => (
                    <div key={cat.label} className="flex items-center gap-2">
                      <span className="text-[10px] font-mono text-slate-500 w-14 text-right truncate">
                        {cat.label}
                      </span>
                      <div className="flex-1 h-1.5 bg-[rgba(255,255,255,0.04)] rounded-full overflow-hidden">
                        <motion.div
                          initial={{ width: 0 }}
                          animate={{
                            width: `${(classification.probabilities[idx] || 0) * 100}%`,
                          }}
                          transition={{
                            duration: prefersReducedMotion ? 0 : 0.15,
                          }}
                          className="h-full rounded-full"
                          style={{
                            background:
                              idx === topCategory
                                ? "linear-gradient(90deg, var(--accent-electric), var(--accent-violet))"
                                : "rgba(148, 163, 184, 0.2)",
                          }}
                        />
                      </div>
                      <span className="text-[10px] font-mono text-slate-600 w-8 text-right">
                        {Math.round(
                          (classification.probabilities[idx] || 0) * 100,
                        )}
                        %
                      </span>
                    </div>
                  ))}
                </div>

                {/* Volume meter */}
                <div className="flex items-center gap-2">
                  <span className="text-[10px] font-mono text-slate-600 w-14 text-right">
                    Volume
                  </span>
                  <div className="flex-1 h-1 bg-[rgba(255,255,255,0.04)] rounded-full overflow-hidden">
                    <motion.div
                      animate={{
                        width: `${Math.min(classification.rms * 300, 100)}%`,
                      }}
                      transition={{
                        duration: prefersReducedMotion ? 0 : 0.1,
                      }}
                      className="h-full rounded-full bg-gradient-to-r from-emerald-500 to-amber-500"
                    />
                  </div>
                </div>
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Idle state */}
      {!isListening && !permissionDenied && (
        <div className="flex items-center gap-2 text-slate-500 text-xs">
          <Mic size={12} className="text-cyan-500/40" />
          <span className="font-mono">
            Click &ldquo;Start Listening&rdquo; to begin audio analysis
          </span>
        </div>
      )}
    </motion.div>
  );
}
