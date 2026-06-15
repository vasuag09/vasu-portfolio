import type Lenis from "lenis";

/**
 * Shared scroll state — ADR-1/ADR-4 frame-sync channel.
 *
 * This is a deliberate exception to the immutability rule: it is the rAF hot
 * path between the DOM scroll authority (SmoothScroll writes) and the canvas
 * (CameraRig reads in useFrame). Routing this through React state would
 * re-render the tree on every scroll frame. Nothing else may write to it.
 */
export interface ScrollState {
  /** Document scroll progress 0–1 (scrubbed by GSAP, damped by Lenis). */
  progress: number;
  /** Lenis velocity, for future shader/FX use. */
  velocity: number;
  /** Section midpoints in normalized scroll space (ascending). */
  sectionCenters: number[];
  /** Live Lenis instance (null before init / under reduced motion). */
  lenis: Lenis | null;
}

export const scrollState: ScrollState = {
  progress: 0,
  velocity: 0,
  sectionCenters: [],
  lenis: null,
};
