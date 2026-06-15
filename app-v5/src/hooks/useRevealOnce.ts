"use client";

import { useEffect, useRef, useState } from "react";
import { useReducedMotion } from "./useReducedMotion";
import {
  resolveHydrationReveal,
  resolveInitialRevealPhase,
  type RevealPhase,
} from "@/lib/reveal-state";

// Flips once the first hydration-pass effect runs. Reveals mounted after
// this (panel content) start hidden; reveals present at hydration start in
// "ssr" so the server paint is never blanked (Phase 8: LCP, no-JS).
let hydrationDone = false;

/**
 * Reveal-on-scroll that fires exactly ONCE (AWARD-RESEARCH §4 / NN-g:
 * repeat animations read as "slow"). Once revealed, stays revealed — the
 * observer disconnects. Under reduced motion content is visible immediately.
 *
 * SSR HTML ships VISIBLE. After hydration, only elements the user provably
 * cannot see (entirely below the fold, or unrendered) are re-hidden and
 * armed for their scroll-in reveal; anything in or above the first viewport
 * keeps its paint (it may be the LCP).
 *
 * Pass a `root` for elements inside a scrollable panel; default observes
 * against the viewport.
 */
export function useRevealOnce<T extends HTMLElement>(
  options: { root?: React.RefObject<HTMLElement | null>; threshold?: number } = {},
): { ref: React.RefObject<T | null>; revealed: boolean } {
  const ref = useRef<T>(null);
  const reducedMotion = useReducedMotion();
  const [phase, setPhase] = useState<RevealPhase>(() =>
    resolveInitialRevealPhase(hydrationDone),
  );
  // Derived, not stored: reduced motion means "always revealed" without a
  // state write (setState-in-effect causes cascading renders).
  const revealed = reducedMotion || phase !== "hidden";

  // Hydration pass: decide whether this element keeps its SSR paint.
  // Geometry is read live, so content the user scrolled to before
  // hydration finished stays visible.
  useEffect(() => {
    hydrationDone = true;
    if (phase !== "ssr") return;
    const el = ref.current;
    if (!el) return;
    const root = options.root?.current ?? null;
    const viewportHeight = root ? root.clientHeight : window.innerHeight;
    setPhase(resolveHydrationReveal(el.getBoundingClientRect(), viewportHeight));
    // Mount-only by design: a one-time decision about the hydration paint.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (revealed) return;
    const el = ref.current;
    if (!el) return;
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries.some((entry) => entry.isIntersecting)) {
          setPhase("revealed");
          observer.disconnect(); // once means once
        }
      },
      {
        root: options.root?.current ?? null,
        threshold: options.threshold ?? 0.15,
      },
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, [revealed, options.root, options.threshold]);

  return { ref, revealed };
}
