"use client";

import { useEffect, useRef, useState } from "react";
import {
  cursorMode,
  INTERACTIVE_SELECTOR,
  resolveCursorEnabled,
} from "@/lib/cursor-state";

/** Ring trails the dot with this lerp per frame (damped follow). */
const RING_LERP = 0.22;

/**
 * Morphing cursor (Phase 10): dot snaps to the pointer, ring lerps behind
 * it; both morph over interactive targets. Touch and reduced-motion users
 * never get it (native cursor untouched) — including Safari on iPad, which
 * reports a fine pointer when a trackpad attaches but fails the hover
 * check otherwise. Position writes happen in a rAF loop on refs; React
 * renders exactly once.
 */
export function CustomCursor() {
  const [enabled, setEnabled] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);
  const dotRef = useRef<HTMLDivElement>(null);
  const ringRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const caps = {
      finePointer: matchMedia("(pointer: fine)").matches,
      canHover: matchMedia("(hover: hover)").matches,
      reducedMotion: matchMedia("(prefers-reduced-motion: reduce)").matches,
    };
    if (!resolveCursorEnabled(caps)) return;
    setEnabled(true);
  }, []);

  useEffect(() => {
    if (!enabled) return;
    const html = document.documentElement;
    html.setAttribute("data-custom-cursor", "");

    let x = window.innerWidth / 2;
    let y = window.innerHeight / 2;
    let ringX = x;
    let ringY = y;
    let seen = false;
    let pressed = false;
    let overInteractive = false;
    let raf = 0;

    const setMode = () => {
      rootRef.current?.setAttribute(
        "data-mode",
        cursorMode({ overInteractive, pressed }),
      );
    };

    const onMove = (event: PointerEvent) => {
      x = event.clientX;
      y = event.clientY;
      if (!seen) {
        // First contact: snap the ring too, no fly-in from center.
        seen = true;
        ringX = x;
        ringY = y;
        rootRef.current?.setAttribute("data-visible", "");
      }
    };
    // pointerover fires on target change only — cheaper than closest() per move.
    const onOver = (event: PointerEvent) => {
      const target = event.target as Element | null;
      overInteractive = Boolean(target?.closest?.(INTERACTIVE_SELECTOR));
      setMode();
    };
    const onDown = () => {
      pressed = true;
      setMode();
    };
    const onUp = () => {
      pressed = false;
      setMode();
    };
    const onLeaveDoc = () => rootRef.current?.removeAttribute("data-visible");
    const onEnterDoc = () => {
      if (seen) rootRef.current?.setAttribute("data-visible", "");
    };

    const tick = () => {
      ringX += (x - ringX) * RING_LERP;
      ringY += (y - ringY) * RING_LERP;
      if (dotRef.current) {
        dotRef.current.style.transform = `translate3d(${x}px, ${y}px, 0)`;
      }
      if (ringRef.current) {
        ringRef.current.style.transform = `translate3d(${ringX}px, ${ringY}px, 0)`;
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);

    window.addEventListener("pointermove", onMove, { passive: true });
    window.addEventListener("pointerover", onOver, { passive: true });
    window.addEventListener("pointerdown", onDown, { passive: true });
    window.addEventListener("pointerup", onUp, { passive: true });
    document.documentElement.addEventListener("pointerleave", onLeaveDoc);
    document.documentElement.addEventListener("pointerenter", onEnterDoc);

    return () => {
      cancelAnimationFrame(raf);
      html.removeAttribute("data-custom-cursor");
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerover", onOver);
      window.removeEventListener("pointerdown", onDown);
      window.removeEventListener("pointerup", onUp);
      document.documentElement.removeEventListener("pointerleave", onLeaveDoc);
      document.documentElement.removeEventListener("pointerenter", onEnterDoc);
    };
  }, [enabled]);

  if (!enabled) return null;

  return (
    <div ref={rootRef} className="custom-cursor" aria-hidden="true">
      <div ref={ringRef} className="cursor-ring">
        <span />
      </div>
      <div ref={dotRef} className="cursor-dot">
        <span />
      </div>
    </div>
  );
}
