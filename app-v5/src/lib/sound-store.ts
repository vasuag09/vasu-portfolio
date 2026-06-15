/**
 * Sound preference store (Phase 7) — same shape as graph-store: one source
 * of truth, useSyncExternalStore on the React side, direct reads elsewhere.
 * SSR snapshot is always "unset"; hydrateSoundPreference() loads the real
 * choice in the first client effect.
 */

import { soundEngine } from "./sound-engine";
import {
  resolveSoundPreference,
  SOUND_STORAGE_KEY,
  type SoundPreference,
} from "./sound-state";

let preference: SoundPreference = "unset";
let hydrated = false;
const listeners = new Set<() => void>();

function notify(): void {
  listeners.forEach((listener) => listener());
}

export function hydrateSoundPreference(): void {
  if (hydrated) return;
  hydrated = true;
  try {
    preference = resolveSoundPreference(localStorage.getItem(SOUND_STORAGE_KEY));
  } catch {
    preference = "unset";
  }
  // Engine remembers enabled-ness; audio is silent until unlock() anyway.
  soundEngine.setEnabled(preference === "on");
  notify();
}

export function getSoundPreference(): SoundPreference {
  return preference;
}

export function getServerSoundPreference(): SoundPreference {
  return "unset";
}

export function setSoundPreference(next: "on" | "off"): void {
  preference = next;
  try {
    localStorage.setItem(SOUND_STORAGE_KEY, next === "on" ? "1" : "0");
  } catch {
    // Private mode: choice lives for this page only.
  }
  soundEngine.setEnabled(next === "on");
  if (next === "off") soundEngine.stopAmbient();
  notify();
}

export function subscribeSoundPreference(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}
