"use client";

import { useEffect, useRef, useState } from "react";
import { useReducedMotion } from "./useReducedMotion";

/**
 * Reveal-on-scroll that fires exactly ONCE (AWARD-RESEARCH §4 / NN-g:
 * repeat animations read as "slow"). Once revealed, stays revealed — the
 * observer disconnects. Under reduced motion content is visible immediately.
 *
 * Pass a `root` for elements inside a scrollable panel; default observes
 * against the viewport.
 */
export function useRevealOnce<T extends HTMLElement>(
  options: { root?: React.RefObject<HTMLElement | null>; threshold?: number } = {},
): { ref: React.RefObject<T | null>; revealed: boolean } {
  const ref = useRef<T>(null);
  const reducedMotion = useReducedMotion();
  const [intersected, setIntersected] = useState(false);
  // Derived, not stored: reduced motion means "always revealed" without a
  // state write (setState-in-effect causes cascading renders).
  const revealed = reducedMotion || intersected;

  useEffect(() => {
    if (revealed) return;
    const el = ref.current;
    if (!el) return;
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries.some((entry) => entry.isIntersecting)) {
          setIntersected(true);
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
