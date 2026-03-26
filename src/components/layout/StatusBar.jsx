import React, { useState, useEffect, useRef } from "react";
import { Wifi, WifiOff, Volume2, VolumeX, Zap, ZapOff } from "lucide-react";
import { useUI } from "../../hooks/useUI";

/**
 * OS-style status bar — fixed at the top of the page.
 * Shows: system name, live clock, network status, sound toggle, FPS counter.
 * Mimics a real OS menu bar for immersive "VASU_OS" experience.
 */
export default function StatusBar() {
  const [time, setTime] = useState(() => formatTime());
  const [fps, setFps] = useState(60);
  const [isOnline, setIsOnline] = useState(
    typeof navigator !== "undefined" ? navigator.onLine : true,
  );
  const { soundEnabled, setSoundEnabled, reducedEffects, setReducedEffects } =
    useUI();
  const fpsRef = useRef({ frames: 0, lastTime: performance.now() });

  // Live clock
  useEffect(() => {
    const interval = setInterval(() => setTime(formatTime()), 1000);
    return () => clearInterval(interval);
  }, []);

  // Network status
  useEffect(() => {
    const handleOnline = () => setIsOnline(true);
    const handleOffline = () => setIsOnline(false);
    window.addEventListener("online", handleOnline);
    window.addEventListener("offline", handleOffline);
    return () => {
      window.removeEventListener("online", handleOnline);
      window.removeEventListener("offline", handleOffline);
    };
  }, []);

  // FPS counter
  useEffect(() => {
    let animId;
    const measure = (now) => {
      fpsRef.current.frames++;
      if (now - fpsRef.current.lastTime >= 1000) {
        setFps(fpsRef.current.frames);
        fpsRef.current.frames = 0;
        fpsRef.current.lastTime = now;
      }
      animId = requestAnimationFrame(measure);
    };
    animId = requestAnimationFrame(measure);
    return () => cancelAnimationFrame(animId);
  }, []);

  return (
    <div className="fixed top-0 left-0 right-0 z-50 h-7 bg-[rgba(6,8,15,0.85)] backdrop-blur-xl border-b border-[rgba(0,212,255,0.06)] hidden md:flex items-center justify-between px-4 md:px-6 select-none">
      {/* Left: System name */}
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-1.5">
          <div className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-neural-breathe" />
          <span
            className="text-[10px] font-bold text-white/80 tracking-wider"
            style={{ fontFamily: "var(--font-display)" }}
          >
            VASU_OS
          </span>
        </div>
        <span className="text-[9px] font-mono text-slate-600 hidden md:inline">
          v2.0
        </span>
      </div>

      {/* Right: Status indicators */}
      <div className="flex items-center gap-3">
        {/* FPS */}
        <span
          className={`text-[9px] font-mono hidden md:inline ${fps >= 50 ? "text-emerald-500/60" : fps >= 30 ? "text-amber-500/60" : "text-red-500/60"}`}
        >
          {fps} FPS
        </span>

        {/* Reduce effects */}
        <button
          onClick={() => setReducedEffects((prev) => !prev)}
          className="text-slate-500 hover:text-cyan-400 transition-colors cursor-pointer hidden md:block"
          title={reducedEffects ? "Enable effects" : "Reduce effects"}
        >
          {reducedEffects ? <ZapOff size={11} /> : <Zap size={11} />}
        </button>

        {/* Sound toggle */}
        <button
          onClick={() => setSoundEnabled((prev) => !prev)}
          className="text-slate-500 hover:text-cyan-400 transition-colors cursor-pointer"
          title={soundEnabled ? "Mute sounds" : "Enable sounds"}
        >
          {soundEnabled ? <Volume2 size={11} /> : <VolumeX size={11} />}
        </button>

        {/* Network */}
        <div className="flex items-center gap-1">
          {isOnline ? (
            <Wifi size={11} className="text-emerald-500/60" />
          ) : (
            <WifiOff size={11} className="text-red-500/60" />
          )}
        </div>

        {/* Clock */}
        <span className="text-[10px] font-mono text-slate-400 min-w-[52px] text-right tabular-nums">
          {time}
        </span>
      </div>
    </div>
  );
}

function formatTime() {
  const now = new Date();
  return now.toLocaleTimeString("en-US", {
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  });
}
