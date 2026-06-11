"use client";

import { useEffect } from "react";
import { scrollState } from "@/lib/scroll-state";

/**
 * Page scroll lock for modal overlays. Two layers:
 *  - lenis.stop(): swallows wheel/touch (Lenis listens on window, so wheel
 *    over a fixed panel would otherwise scroll the page behind it)
 *  - html overflow hidden: blocks keyboard scrolling and the reduced-motion
 *    path where Lenis doesn't exist
 * scrollbar-gutter: stable (globals.css) keeps layout still when the
 * scrollbar disappears.
 */
export function useScrollLock(active: boolean): void {
  useEffect(() => {
    if (!active) return;
    scrollState.lenis?.stop();
    const html = document.documentElement;
    const previous = html.style.overflow;
    html.style.overflow = "hidden";
    return () => {
      html.style.overflow = previous;
      scrollState.lenis?.start();
    };
  }, [active]);
}
