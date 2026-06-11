/**
 * Web Audio synthesis engine (Phase 7). Every sound is generated — zero
 * network weight, zero licensing. The palette is the site's voice: phosphor
 * terminal (short sine/triangle blips, a low ambient pad). Licensed audio
 * files can replace any voice later by swapping the play* internals.
 *
 * Lifecycle: construct lazily AFTER a user gesture (autoplay policy / iOS
 * unlock — the SoundGate click or first input). All public methods are
 * no-ops until unlock() runs and while disabled.
 */

const MASTER_LEVEL = 0.35;
const AMBIENT_LEVEL = 0.05;

class SoundEngine {
  private ctx: AudioContext | null = null;
  private master: GainNode | null = null;
  private ambientNodes: { stop: () => void } | null = null;
  private enabled = false;

  /** Must be called from a user-gesture call stack. Idempotent. */
  unlock(): void {
    if (this.ctx) {
      void this.ctx.resume();
      return;
    }
    if (typeof window === "undefined" || !("AudioContext" in window)) return;
    this.ctx = new AudioContext();
    this.master = this.ctx.createGain();
    this.master.gain.value = this.enabled ? MASTER_LEVEL : 0;
    this.master.connect(this.ctx.destination);
    // Preference was "on" before any gesture existed (return visitor):
    // the pad starts at the first unlock.
    if (this.enabled) this.startAmbient();
  }

  get unlocked(): boolean {
    return this.ctx !== null;
  }

  setEnabled(on: boolean): void {
    this.enabled = on;
    if (!this.ctx || !this.master) return;
    // 80ms ramp avoids the click of a hard gain step.
    const t = this.ctx.currentTime;
    this.master.gain.cancelScheduledValues(t);
    this.master.gain.setTargetAtTime(on ? MASTER_LEVEL : 0, t, 0.08);
    if (on) {
      void this.ctx.resume();
      this.startAmbient();
    }
  }

  /** Boot jingle: rising phosphor arpeggio, ~0.9s, mirrors the scramble. */
  playJingle(): void {
    const notes = [220, 330, 440, 660];
    notes.forEach((freq, i) => {
      this.blip(freq, 0.18, i * 0.16, "triangle", 0.5);
    });
    this.blip(880, 0.4, notes.length * 0.16, "sine", 0.3);
  }

  playHover(): void {
    this.blip(1800, 0.025, 0, "sine", 0.12);
  }

  playClick(): void {
    this.blip(880, 0.06, 0, "triangle", 0.25);
  }

  playOpen(): void {
    this.sweep(330, 660, 0.18, 0.25);
  }

  playClose(): void {
    this.sweep(660, 330, 0.15, 0.2);
  }

  /** Low two-oscillator pad with slow filter drift. Runs until disabled. */
  private startAmbient(): void {
    if (!this.ctx || !this.master || this.ambientNodes) return;
    const ctx = this.ctx;
    const gain = ctx.createGain();
    gain.gain.value = 0;
    gain.gain.setTargetAtTime(AMBIENT_LEVEL, ctx.currentTime, 2);

    const filter = ctx.createBiquadFilter();
    filter.type = "lowpass";
    filter.frequency.value = 320;

    const oscA = ctx.createOscillator();
    oscA.type = "sine";
    oscA.frequency.value = 55;
    const oscB = ctx.createOscillator();
    oscB.type = "triangle";
    oscB.frequency.value = 55.7; // beat against oscA: slow shimmer

    const lfo = ctx.createOscillator();
    lfo.frequency.value = 0.05;
    const lfoGain = ctx.createGain();
    lfoGain.gain.value = 120;
    lfo.connect(lfoGain).connect(filter.frequency);

    oscA.connect(filter);
    oscB.connect(filter);
    filter.connect(gain).connect(this.master);
    oscA.start();
    oscB.start();
    lfo.start();

    this.ambientNodes = {
      stop: () => {
        gain.gain.setTargetAtTime(0, ctx.currentTime, 0.3);
        // Let the fade finish before tearing the graph down.
        window.setTimeout(() => {
          oscA.stop();
          oscB.stop();
          lfo.stop();
        }, 1200);
      },
    };
  }

  stopAmbient(): void {
    this.ambientNodes?.stop();
    this.ambientNodes = null;
  }

  private blip(
    freq: number,
    duration: number,
    delay: number,
    type: OscillatorType,
    peak: number,
  ): void {
    if (!this.ctx || !this.master || !this.enabled) return;
    const ctx = this.ctx;
    const t = ctx.currentTime + delay;
    const osc = ctx.createOscillator();
    osc.type = type;
    osc.frequency.value = freq;
    const gain = ctx.createGain();
    gain.gain.setValueAtTime(0, t);
    gain.gain.linearRampToValueAtTime(peak, t + 0.01);
    gain.gain.exponentialRampToValueAtTime(0.001, t + duration);
    osc.connect(gain).connect(this.master);
    osc.start(t);
    osc.stop(t + duration + 0.05);
  }

  private sweep(from: number, to: number, duration: number, peak: number): void {
    if (!this.ctx || !this.master || !this.enabled) return;
    const ctx = this.ctx;
    const t = ctx.currentTime;
    const osc = ctx.createOscillator();
    osc.type = "sine";
    osc.frequency.setValueAtTime(from, t);
    osc.frequency.exponentialRampToValueAtTime(to, t + duration);
    const gain = ctx.createGain();
    gain.gain.setValueAtTime(0, t);
    gain.gain.linearRampToValueAtTime(peak, t + 0.02);
    gain.gain.exponentialRampToValueAtTime(0.001, t + duration);
    osc.connect(gain).connect(this.master);
    osc.start(t);
    osc.stop(t + duration + 0.05);
  }
}

/** Module singleton — one audio graph per page. */
export const soundEngine = new SoundEngine();
