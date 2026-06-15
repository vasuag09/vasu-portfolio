/**
 * Phase-8 reveal policy. The server must paint content (LCP + no-JS safety),
 * so Reveal ships VISIBLE in SSR HTML and only re-hides what the user cannot
 * see yet. Pure decision logic — useRevealOnce gathers geometry and calls
 * this (same pattern as particle-config.ts).
 */

export type RevealPhase = "ssr" | "hidden" | "revealed";

export interface RevealRect {
  top: number;
  bottom: number;
  width: number;
  height: number;
}

/**
 * Initial phase for a Reveal mount. During hydration the SSR paint is
 * already on screen, so we start in "ssr" and decide per-element after
 * layout. Anything mounted later (case-study panel content) was never
 * painted, so it can start hidden and animate in without a flash.
 */
export function resolveInitialRevealPhase(
  mountedAfterHydration: boolean,
): RevealPhase {
  return mountedAfterHydration ? "hidden" : "ssr";
}

/**
 * Hydration-time decision for an element that is currently painted.
 * Hiding is only safe when the user provably cannot see the element.
 */
export function resolveHydrationReveal(
  rect: RevealRect,
  viewportHeight: number,
): Extract<RevealPhase, "revealed" | "hidden"> {
  // Not laid out (display:none ancestor, e.g. a closed panel) — invisible
  // anyway; let the IntersectionObserver decide when it becomes real.
  if (rect.width === 0 && rect.height === 0) return "hidden";

  // Entirely below the fold: hiding is invisible to the user and the
  // scroll-in reveal (Phase 5 behavior) still fires.
  if (rect.top >= viewportHeight) return "hidden";

  // Intersecting the viewport or above it: this paint may be the LCP, or
  // content the user already scrolled past. Never blink it away.
  return "revealed";
}
