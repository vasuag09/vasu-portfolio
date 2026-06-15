"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  BOOT_COMPLETE_EVENT,
  BOOT_STORAGE_KEY,
  BOOT_TIMELINE,
  bootPhaseAt,
  randomGlyph,
  scrambleText,
} from "@/lib/boot-state";
import { shouldShowSoundGate } from "@/lib/sound-state";
import {
  getSoundPreference,
  hydrateSoundPreference,
  setSoundPreference,
} from "@/lib/sound-store";
import { soundEngine } from "@/lib/sound-engine";

const NAME = "VASU AGRAWAL";
// Deterministic initial frame: server HTML and the client's first render
// must match (hydration), and the resolve animation reads forward from it.
const INITIAL_FRAME = scrambleText(NAME, 0, () => "█");

type Stage = "gate" | "anim" | "done";

interface BurstGlyph {
  char: string;
  x: number; // start offset, px
  y: number;
  delay: number; // ms
}

/**
 * Converging glyph burst (design elevation P1.9): ~26 glyphs fly from a
 * ring into the name as the scramble resolves. Generated client-side only
 * when the anim stage starts (never part of SSR HTML), compositor-only
 * keyframes, finishes inside the scramble window.
 */
function makeBurst(): BurstGlyph[] {
  const glyphs: BurstGlyph[] = [];
  for (let i = 0; i < 26; i += 1) {
    const angle = (i / 26) * Math.PI * 2 + Math.random() * 0.4;
    const radius = 180 + Math.random() * 240;
    glyphs.push({
      char: randomGlyph(),
      x: Math.round(Math.cos(angle) * radius * 1.6),
      y: Math.round(Math.sin(angle) * radius),
      delay: Math.round(Math.random() * 380),
    });
  }
  return glyphs;
}

/**
 * Boot ritual (Phase 10) + SoundGate (Phase 7, PLAN Q4). First visit:
 * two-button gate — the choice click is the Web Audio unlock gesture, then
 * the scramble plays (jingle alongside when sound is on; ≤1.5s from click).
 * Esc enters silently without persisting a choice. Return visits skip both
 * (decided pre-paint by the layout.tsx inline script). aria: real dialog
 * while the gate is up, decorative afterwards.
 */
export function BootSequence() {
  const [stage, setStage] = useState<Stage | null>(null);
  // One burst per anim entry — render must stay pure (review finding).
  const burst = useMemo(() => (stage === "anim" ? makeBurst() : []), [stage]);
  const overlayRef = useRef<HTMLDivElement>(null);
  const nameRef = useRef<HTMLSpanElement>(null);
  const primaryRef = useRef<HTMLButtonElement>(null);
  const finishRef = useRef<() => void>(() => {});

  useEffect(() => {
    const html = document.documentElement;
    if (html.getAttribute("data-boot") !== "play") {
      setStage("done");
      return;
    }
    // Hydration reached us — the CSS failsafe (for dead-JS loads) must not
    // clear an overlay that may legitimately wait on the gate.
    overlayRef.current?.setAttribute("data-hydrated", "");
    hydrateSoundPreference();
    setStage(
      shouldShowSoundGate({
        bootPlaying: true,
        preference: getSoundPreference(),
      })
        ? "gate"
        : "anim",
    );
  }, []);

  // Gate: focus the encouraged action; Esc enters silently (no persisted
  // choice — the corner toggle remains the escape hatch).
  useEffect(() => {
    if (stage !== "gate") return;
    primaryRef.current?.focus();
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") finishRef.current();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [stage]);

  // Scramble timeline + skip-on-any-input (never racing the gate buttons:
  // these listeners exist only during "anim").
  useEffect(() => {
    if (stage !== "anim") return;
    const html = document.documentElement;
    let raf = 0;
    let finished = false;
    const start = performance.now();

    const finish = () => {
      if (finished) return;
      finished = true;
      cancelAnimationFrame(raf);
      try {
        localStorage.setItem(BOOT_STORAGE_KEY, "1");
      } catch {
        // Private mode without storage: boot replays next visit. Harmless.
      }
      html.removeAttribute("data-boot");
      // Phase-7 hook: the sound layer / future listeners hear this on
      // finish AND on skip.
      window.dispatchEvent(new CustomEvent(BOOT_COMPLETE_EVENT));
      setStage("done");
    };
    finishRef.current = finish;

    const tick = (now: number) => {
      const elapsed = now - start;
      const phase = bootPhaseAt(elapsed);
      if (nameRef.current) {
        nameRef.current.textContent = scrambleText(
          NAME,
          elapsed / BOOT_TIMELINE.scrambleMs,
        );
      }
      if (phase === "fade") overlayRef.current?.setAttribute("data-fading", "");
      if (phase === "done") {
        finish();
        return;
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);

    const skip = () => finish();
    window.addEventListener("keydown", skip);
    window.addEventListener("pointerdown", skip);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("keydown", skip);
      window.removeEventListener("pointerdown", skip);
    };
  }, [stage]);

  // Gate finish path needs to work before the anim effect installs the real
  // finish (Esc during gate = enter silently, skipping the ritual entirely).
  useEffect(() => {
    finishRef.current = () => {
      try {
        localStorage.setItem(BOOT_STORAGE_KEY, "1");
      } catch {
        /* see above */
      }
      document.documentElement.removeAttribute("data-boot");
      window.dispatchEvent(new CustomEvent(BOOT_COMPLETE_EVENT));
      setStage("done");
    };
  }, []);

  const choose = (withSound: boolean) => {
    // Click stack = legitimate unlock gesture (autoplay policy, iOS).
    soundEngine.unlock();
    setSoundPreference(withSound ? "on" : "off");
    if (withSound) soundEngine.playJingle();
    setStage("anim");
  };

  if (stage === "done") return null;
  const gating = stage === "gate";

  return (
    <div
      ref={overlayRef}
      className="boot-overlay"
      role={gating ? "dialog" : "presentation"}
      aria-modal={gating || undefined}
      aria-label={gating ? "Welcome — choose sound" : undefined}
      aria-hidden={gating ? undefined : "true"}
    >
      <p
        className="text-[length:var(--text-xs)] tracking-[var(--tracking-terminal)] uppercase"
        style={{ color: "var(--accent)" }}
      >
        Neural core · online
      </p>
      <span className="relative inline-block">
        {stage === "anim" ? (
          <span aria-hidden="true" className="boot-burst">
            {burst.map((g, i) => (
              <span
                key={i}
                className="boot-burst-glyph"
                style={
                  {
                    "--bx": `${g.x}px`,
                    "--by": `${g.y}px`,
                    animationDelay: `${g.delay}ms`,
                  } as React.CSSProperties
                }
              >
                {g.char}
              </span>
            ))}
          </span>
        ) : null}
        <span
          ref={nameRef}
          aria-hidden="true"
          className="font-bold leading-[var(--leading-tight)]"
          style={{ fontSize: "var(--text-xl)" }}
        >
          {INITIAL_FRAME}
        </span>
      </span>
      {gating ? (
        <div className="mt-2 flex items-center gap-4">
          <button
            ref={primaryRef}
            type="button"
            onClick={() => choose(true)}
            className="cursor-pointer rounded border px-4 py-2 text-[length:var(--text-sm)] tracking-[var(--tracking-wide)] uppercase transition-colors duration-[var(--duration-fast)]"
            style={{
              borderColor: "var(--border-active)",
              color: "var(--accent)",
              boxShadow: "0 0 12px var(--accent-glow)",
            }}
          >
            Enter with sound
          </button>
          <button
            type="button"
            onClick={() => choose(false)}
            className="cursor-pointer rounded border px-4 py-2 text-[length:var(--text-sm)] tracking-[var(--tracking-wide)] uppercase transition-colors duration-[var(--duration-fast)] hover:border-[var(--border-active)]"
            style={{ borderColor: "var(--border)", color: "var(--text-muted)" }}
          >
            Enter muted
          </button>
        </div>
      ) : (
        <p
          className="text-[length:var(--text-xs)] tracking-[var(--tracking-wide)] uppercase"
          style={{ color: "var(--text-faint)" }}
        >
          any key to skip
        </p>
      )}
    </div>
  );
}
