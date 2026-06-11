"use client";

import { useEffect } from "react";
import { getGraphState, subscribeGraphState } from "@/lib/graph-store";
import { soundEngine } from "@/lib/sound-engine";
import { soundEventsForChange } from "@/lib/sound-state";
import { hydrateSoundPreference } from "@/lib/sound-store";
import { INTERACTIVE_SELECTOR } from "@/lib/cursor-state";

const HOVER_THROTTLE_MS = 90;

/**
 * Invisible sound wiring (Phase 7):
 *  - hydrates the persisted preference
 *  - unlocks Web Audio on the first gesture (iOS/autoplay policy) so a
 *    returning sound-on visitor gets the ambient pad without re-choosing
 *  - overlay open/close whooshes derived from graph-store transitions
 *  - hover/click blips via event delegation on interactive elements
 * All engine calls are no-ops while sound is off — listeners stay cheap.
 */
export function SoundController() {
  useEffect(() => {
    hydrateSoundPreference();

    const unlock = () => {
      soundEngine.unlock();
      window.removeEventListener("pointerdown", unlock);
      window.removeEventListener("keydown", unlock);
    };
    window.addEventListener("pointerdown", unlock, { passive: true });
    window.addEventListener("keydown", unlock);

    let prev = {
      selectedProjectId: getGraphState().selectedProjectId,
      synapseOpen: getGraphState().synapseOpen,
    };
    const unsubscribe = subscribeGraphState(() => {
      const state = getGraphState();
      const next = {
        selectedProjectId: state.selectedProjectId,
        synapseOpen: state.synapseOpen,
      };
      for (const event of soundEventsForChange(prev, next)) {
        if (event === "open") soundEngine.playOpen();
        else soundEngine.playClose();
      }
      prev = next;
    });

    let lastHover = 0;
    const onOver = (event: PointerEvent) => {
      const target = event.target as Element | null;
      if (!target?.closest?.(INTERACTIVE_SELECTOR)) return;
      const now = performance.now();
      if (now - lastHover < HOVER_THROTTLE_MS) return;
      lastHover = now;
      soundEngine.playHover();
    };
    const onClick = (event: MouseEvent) => {
      const target = event.target as Element | null;
      if (!target?.closest?.(INTERACTIVE_SELECTOR)) return;
      soundEngine.playClick();
    };
    window.addEventListener("pointerover", onOver, { passive: true });
    window.addEventListener("click", onClick, { passive: true });

    return () => {
      window.removeEventListener("pointerdown", unlock);
      window.removeEventListener("keydown", unlock);
      window.removeEventListener("pointerover", onOver);
      window.removeEventListener("click", onClick);
      unsubscribe();
    };
  }, []);

  return null;
}
