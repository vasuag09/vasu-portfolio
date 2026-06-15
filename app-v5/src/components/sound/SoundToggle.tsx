"use client";

import { useSyncExternalStore } from "react";
import {
  getServerSoundPreference,
  getSoundPreference,
  setSoundPreference,
  subscribeSoundPreference,
} from "@/lib/sound-store";
import { soundEngine } from "@/lib/sound-engine";

/**
 * Persistent sound toggle (Phase 7): fixed bottom-left, mirrors the gate
 * choice, works for visitors who never saw the gate (return visits,
 * reduced motion). Click is a gesture, so it can unlock audio itself.
 */
export function SoundToggle() {
  const preference = useSyncExternalStore(
    subscribeSoundPreference,
    getSoundPreference,
    getServerSoundPreference,
  );
  const on = preference === "on";

  const toggle = () => {
    soundEngine.unlock();
    setSoundPreference(on ? "off" : "on");
  };

  return (
    <button
      type="button"
      onClick={toggle}
      aria-pressed={on}
      aria-label={on ? "Turn sound off" : "Turn sound on"}
      className="fixed bottom-5 left-5 cursor-pointer rounded border px-3 py-1.5 text-[length:var(--text-xs)] tracking-[var(--tracking-terminal)] uppercase transition-colors duration-[var(--duration-fast)] hover:border-[var(--border-active)]"
      style={{
        zIndex: "var(--z-nav)",
        background: "var(--bg-elevated)",
        borderColor: on ? "var(--border-active)" : "var(--border)",
        color: on ? "var(--accent)" : "var(--text-muted)",
      }}
    >
      snd {on ? "on" : "off"}
    </button>
  );
}
