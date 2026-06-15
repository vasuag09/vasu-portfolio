/**
 * Boot sequence (Phase 10) — pure timeline + text logic. The BootSequence
 * component owns the rAF loop and DOM; everything decidable without a
 * browser lives here.
 *
 * Phase-8 contract (lib/reveal-state.ts): the boot overlay layers ON TOP of
 * painted SSR content — it never gates or re-hides it. Skip decisions happen
 * pre-paint via the inline script in layout.tsx writing html[data-boot].
 */

export const BOOT_STORAGE_KEY = "v5:booted";

/** Phase 7 subscribes here for the boot jingle. */
export const BOOT_COMPLETE_EVENT = "v5:boot-complete";

/** Total 1400ms — under the 1.5s Phase-10 exit gate. */
export const BOOT_TIMELINE = {
  scrambleMs: 900,
  holdMs: 200,
  fadeMs: 300,
} as const;

export type BootPhase = "scramble" | "hold" | "fade" | "done";

export interface BootContext {
  hasBooted: boolean;
  reducedMotion: boolean;
}

/** First visit + motion allowed. Mirrors the inline pre-paint script. */
export function shouldPlayBoot(ctx: BootContext): boolean {
  return !ctx.hasBooted && !ctx.reducedMotion;
}

export function bootPhaseAt(elapsedMs: number): BootPhase {
  const { scrambleMs, holdMs, fadeMs } = BOOT_TIMELINE;
  if (elapsedMs < scrambleMs) return "scramble";
  if (elapsedMs < scrambleMs + holdMs) return "hold";
  if (elapsedMs < scrambleMs + holdMs + fadeMs) return "fade";
  return "done";
}

/** Terminal-style glyph pool for unresolved characters. */
const GLYPHS = "█▓▒░<>/\\|=+*#@$%&";

export function randomGlyph(): string {
  return GLYPHS[Math.floor(Math.random() * GLYPHS.length)];
}

/**
 * Characters lock in left-to-right as progress 0→1; unresolved positions
 * show a glyph from the pool. Spaces always render as spaces so the word
 * silhouette reads throughout the scramble.
 */
export function scrambleText(
  target: string,
  progress: number,
  glyph: () => string = randomGlyph,
): string {
  const clamped = Math.min(1, Math.max(0, progress));
  const lockedCount = Math.round(clamped * target.length);
  let out = "";
  for (let i = 0; i < target.length; i++) {
    if (target[i] === " ") {
      out += " ";
    } else if (i < lockedCount) {
      out += target[i];
    } else {
      out += glyph();
    }
  }
  return out;
}
