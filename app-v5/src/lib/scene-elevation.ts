import { projectNodes, projects } from "@/data/projects-v5";
import { skills } from "@/data/skills-graph";
import type { Vec3 } from "@/data/types";

/**
 * Pure helpers for the design-elevation scene pass (ADR-7/ADR-8 +
 * line art direction). Deterministic — safe at module load, SSR-parity.
 */

/** Quadratic-bezier polyline between two nodes with a perpendicular bow. */
export function curveSegmentPoints(
  from: Vec3,
  to: Vec3,
  steps: number,
  bulgeRatio = 0.14,
): Vec3[] {
  const dx = to[0] - from[0];
  const dy = to[1] - from[1];
  const dz = to[2] - from[2];
  const len = Math.hypot(dx, dy, dz);

  // Perpendicular via cross(dir, up); near-vertical edges fall back to X.
  let px = -dz;
  const py = 0;
  let pz = dx;
  const pLen = Math.hypot(px, py, pz);
  if (pLen < 1e-4) {
    px = 1;
    pz = 0;
  } else {
    px /= pLen;
    pz /= pLen;
  }
  // Deterministic bow direction + slight vertical lift from endpoint hash.
  const hash = Math.abs(
    Math.sin(from[0] * 12.9898 + from[1] * 78.233 + to[0] * 37.719 + to[2] * 4.581),
  );
  const sign = hash > 0.5 ? 1 : -1;
  const bulge = len * bulgeRatio;
  const mid: Vec3 = [
    (from[0] + to[0]) / 2 + px * bulge * sign,
    (from[1] + to[1]) / 2 + (hash - 0.5) * bulge + py,
    (from[2] + to[2]) / 2 + pz * bulge * sign,
  ];

  const pts: Vec3[] = [];
  for (let i = 0; i <= steps; i += 1) {
    const t = i / steps;
    if (i === 0) {
      pts.push([...from]);
      continue;
    }
    if (i === steps) {
      pts.push([...to]);
      continue;
    }
    const a = 1 - t;
    pts.push([
      a * a * from[0] + 2 * a * t * mid[0] + t * t * to[0],
      a * a * from[1] + 2 * a * t * mid[1] + t * t * to[1],
      a * a * from[2] + 2 * a * t * mid[2] + t * t * to[2],
    ]);
  }
  return pts;
}

/**
 * Idle visibility by edge length: short intra-cluster links read fully,
 * long cross-section edges become a whisper (0.12 floor) — until activation
 * restores them (the hover payoff). Kills the full-frame streak artifact.
 */
export function edgeLengthFade(length: number): number {
  const t = Math.min(1, Math.max(0, (length - 6) / (20 - 6)));
  const smooth = t * t * (3 - 2 * t);
  return 1 - smooth * 0.88;
}

export interface SceneLabel {
  id: string;
  text: string;
  position: Vec3;
  kind: "project" | "skill";
}

/** ADR-8 label set: every flagship + the core GenAI skills. ≤14 total. */
export function pickLabelNodes(): SceneLabel[] {
  const titleById = new Map(projects.map((p) => [p.id, p.title]));
  const flagships: SceneLabel[] = projectNodes
    .filter((n) => n.flagship)
    .map((n) => ({
      id: n.id,
      text: titleById.get(n.id) ?? n.id,
      position: n.position,
      kind: "project" as const,
    }));
  const genai: SceneLabel[] = skills
    .filter((s) => s.category === "genai")
    .slice(0, 14 - flagships.length)
    .map((s) => ({
      id: s.id,
      text: s.label,
      position: s.position,
      kind: "skill" as const,
    }));
  return [...flagships, ...genai];
}
