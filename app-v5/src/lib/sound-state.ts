/**
 * Sound layer (Phase 7) — pure decision logic. PLAN Q4: explicit two-button
 * gate on first visit ("Enter with sound" encouraged), no autoplay, choice
 * persisted. The gate lives inside the boot overlay, so the enter click is
 * also the Web Audio unlock gesture (autoplay policy + iOS).
 */

import type { GraphUIState } from "./graph-store";

export const SOUND_STORAGE_KEY = "v5:sound";

export type SoundPreference = "on" | "off" | "unset";

export function resolveSoundPreference(stored: string | null): SoundPreference {
  if (stored === "1") return "on";
  if (stored === "0") return "off";
  return "unset";
}

/**
 * Gate only on a playing boot overlay (first visit, motion allowed) with no
 * stored choice. Return visitors and reduced-motion users get the corner
 * toggle instead — never a blocking gate.
 */
export function shouldShowSoundGate(ctx: {
  bootPlaying: boolean;
  preference: SoundPreference;
}): boolean {
  return ctx.bootPlaying && ctx.preference === "unset";
}

export type SoundEvent = "open" | "close";

type OverlaySlice = Pick<GraphUIState, "selectedProjectId" | "synapseOpen">;

/**
 * Overlay SFX derivation: store transition → sounds. A project switch while
 * the panel is open reads as a fresh open (content replaced).
 */
export function soundEventsForChange(
  prev: OverlaySlice,
  next: OverlaySlice,
): SoundEvent[] {
  const events: SoundEvent[] = [];
  const panelOpened =
    next.selectedProjectId !== null &&
    next.selectedProjectId !== prev.selectedProjectId;
  const panelClosed = prev.selectedProjectId !== null && next.selectedProjectId === null;
  const terminalOpened = !prev.synapseOpen && next.synapseOpen;
  const terminalClosed = prev.synapseOpen && !next.synapseOpen;

  if (panelOpened || terminalOpened) events.push("open");
  if (panelClosed || terminalClosed) events.push("close");
  return events;
}
