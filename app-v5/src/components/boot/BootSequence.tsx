"use client";

import { useEffect, useRef, useState } from "react";
import {
  BOOT_COMPLETE_EVENT,
  BOOT_STORAGE_KEY,
  BOOT_TIMELINE,
  bootPhaseAt,
  scrambleText,
} from "@/lib/boot-state";

const NAME = "VASU AGRAWAL";
// Deterministic initial frame: server HTML and the client's first render
// must match (hydration), and the resolve animation reads forward from it.
const INITIAL_FRAME = scrambleText(NAME, 0, () => "█");

/**
 * Boot ritual (Phase 10): scramble → name resolve → fade, 1.4s total.
 * Any key/pointer input skips. localStorage marks the visit so return
 * visits never see it (decided pre-paint by the layout.tsx inline script —
 * this component only animates what that script already revealed).
 * aria-hidden: purely decorative; the real content is painted underneath.
 */
export function BootSequence() {
  const [dismissed, setDismissed] = useState(false);
  const overlayRef = useRef<HTMLDivElement>(null);
  const nameRef = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    const html = document.documentElement;
    if (html.getAttribute("data-boot") !== "play") {
      setDismissed(true);
      return;
    }

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
      // Phase-7 jingle hook: the sound layer subscribes to this event.
      window.dispatchEvent(new CustomEvent(BOOT_COMPLETE_EVENT));
      setDismissed(true);
    };

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
  }, []);

  if (dismissed) return null;

  return (
    <div
      ref={overlayRef}
      className="boot-overlay"
      aria-hidden="true"
      role="presentation"
    >
      <p
        className="text-[length:var(--text-xs)] tracking-[var(--tracking-terminal)] uppercase"
        style={{ color: "var(--accent)" }}
      >
        Neural core · online
      </p>
      <span
        ref={nameRef}
        className="font-bold leading-[var(--leading-tight)]"
        style={{ fontSize: "var(--text-xl)" }}
      >
        {INITIAL_FRAME}
      </span>
      <p
        className="text-[length:var(--text-xs)] tracking-[var(--tracking-wide)] uppercase"
        style={{ color: "var(--text-faint)" }}
      >
        any key to skip
      </p>
    </div>
  );
}
