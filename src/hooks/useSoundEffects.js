import { useCallback, useRef } from "react";
import { useUI } from "./useUI";

/**
 * Programmatic sound effects using Web Audio API.
 * No external files — generates synth sounds from oscillators.
 * All sounds are short, subtle, and non-intrusive.
 * Muted by default; controlled via UIProvider's soundEnabled state.
 */
export function useSoundEffects() {
  const { soundEnabled } = useUI();
  const ctxRef = useRef(null);

  const getContext = useCallback(() => {
    if (!ctxRef.current) {
      ctxRef.current = new (window.AudioContext || window.webkitAudioContext)();
    }
    // Resume if suspended (autoplay policy)
    if (ctxRef.current.state === "suspended") {
      ctxRef.current.resume();
    }
    return ctxRef.current;
  }, []);

  /**
   * Short blip — for button clicks.
   * Quick sine wave with fast decay.
   */
  const playClick = useCallback(() => {
    if (!soundEnabled) return;
    try {
      const ctx = getContext();
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();

      osc.type = "sine";
      osc.frequency.setValueAtTime(800, ctx.currentTime);
      osc.frequency.exponentialRampToValueAtTime(400, ctx.currentTime + 0.08);

      gain.gain.setValueAtTime(0.06, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.08);

      osc.connect(gain);
      gain.connect(ctx.destination);

      osc.start(ctx.currentTime);
      osc.stop(ctx.currentTime + 0.1);
    } catch {
      // Silently fail if audio context not available
    }
  }, [soundEnabled, getContext]);

  /**
   * Soft hover — subtle high-frequency pad.
   */
  const playHover = useCallback(() => {
    if (!soundEnabled) return;
    try {
      const ctx = getContext();
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();

      osc.type = "sine";
      osc.frequency.setValueAtTime(1200, ctx.currentTime);

      gain.gain.setValueAtTime(0.02, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.06);

      osc.connect(gain);
      gain.connect(ctx.destination);

      osc.start(ctx.currentTime);
      osc.stop(ctx.currentTime + 0.08);
    } catch {
      // Silently fail
    }
  }, [soundEnabled, getContext]);

  /**
   * Transition sweep — ascending tone for page transitions.
   */
  const playTransition = useCallback(() => {
    if (!soundEnabled) return;
    try {
      const ctx = getContext();
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();

      osc.type = "sine";
      osc.frequency.setValueAtTime(200, ctx.currentTime);
      osc.frequency.exponentialRampToValueAtTime(800, ctx.currentTime + 0.2);

      gain.gain.setValueAtTime(0.04, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.25);

      osc.connect(gain);
      gain.connect(ctx.destination);

      osc.start(ctx.currentTime);
      osc.stop(ctx.currentTime + 0.3);
    } catch {
      // Silently fail
    }
  }, [soundEnabled, getContext]);

  /**
   * Success chime — two-note ascending for confirmations.
   */
  const playSuccess = useCallback(() => {
    if (!soundEnabled) return;
    try {
      const ctx = getContext();

      [600, 900].forEach((freq, i) => {
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        const offset = i * 0.1;

        osc.type = "sine";
        osc.frequency.setValueAtTime(freq, ctx.currentTime + offset);

        gain.gain.setValueAtTime(0.05, ctx.currentTime + offset);
        gain.gain.exponentialRampToValueAtTime(
          0.001,
          ctx.currentTime + offset + 0.15,
        );

        osc.connect(gain);
        gain.connect(ctx.destination);

        osc.start(ctx.currentTime + offset);
        osc.stop(ctx.currentTime + offset + 0.2);
      });
    } catch {
      // Silently fail
    }
  }, [soundEnabled, getContext]);

  return { playClick, playHover, playTransition, playSuccess };
}
