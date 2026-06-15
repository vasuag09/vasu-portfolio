import type { SectionAnchor } from "./types";

/**
 * Hand-authored section anchors (ADR-3) — the five regions of the neural core.
 * The camera spline is a CubicBezierCurve3 chain through these rest-poses,
 * built at Phase 1 by buildCameraSpline(anchors). Anchors are locked after the
 * Phase-0 design review; tune via the Phase-1 debug spline visualization, not
 * ad hoc.
 *
 * Spatial story: boot at the core (hero), fly outward to the project cluster,
 * across to the skills constellation, dive below the plane for about, then
 * pull back and up for contact — ending with the whole network in frame.
 */
export const sections: readonly SectionAnchor[] = [
  {
    id: "hero",
    label: "Neural Core",
    cameraPos: [0, 0, 16],
    cameraTarget: [0, 0, 0],
  },
  {
    id: "projects",
    label: "Projects",
    cameraPos: [22, 5, 11],
    cameraTarget: [24, 2, 0],
  },
  {
    id: "signal",
    label: "Signal",
    // Elevated, centered "broadcast" vantage on the transit from the project
    // cluster (+X) to the skills constellation (-X): the camera rises above
    // the network to survey it, then descends to skills. NeuralNetwork drops
    // a section-core node at the target — a signal node above the core.
    cameraPos: [3, 13, 16],
    cameraTarget: [0, 5, 0],
  },
  {
    id: "skills",
    label: "Skills Graph",
    cameraPos: [-20, 7, 13],
    cameraTarget: [-22, 1, 0],
  },
  {
    id: "about",
    label: "About",
    cameraPos: [4, -16, 12],
    cameraTarget: [6, -18, 0],
  },
  {
    id: "contact",
    label: "Contact",
    cameraPos: [0, 10, 26],
    cameraTarget: [0, 0, 0],
  },
] as const;

export function getSection(id: string): SectionAnchor {
  const section = sections.find((candidate) => candidate.id === id);
  if (!section) {
    throw new Error(`Unknown section id: ${id}`);
  }
  return section;
}
