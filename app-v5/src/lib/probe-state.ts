import { resolveCursorEnabled, type CursorCaps } from "./cursor-state";

/**
 * Cursor probe (Living wave 2) — pointer-reactive particle lift. Pure
 * gating here; the world-space pointer unprojection and the uniform write
 * live in ProbeDriver (canvas, wave 2 phase 2). Same pure/wrapper split as
 * cursor-state.ts: decisions in lib, side effects in the component.
 *
 * The probe shares the custom cursor's EXACT enable gate — a fine pointer
 * that can hover, with motion not reduced. A probe on a touch device (no
 * hover) or under reduced motion would be either impossible or unwanted.
 */
export function resolveProbeEnabled(caps: CursorCaps): boolean {
  return resolveCursorEnabled(caps);
}

/** Probe strength target for the uProbeStrength uniform: full on, else off. */
export function probeStrengthTarget(caps: CursorCaps): number {
  return resolveProbeEnabled(caps) ? 1 : 0;
}
