import type { SectionAnchor, Vec3 } from "@/data/types";
import { seededRandom } from "./seeded-random";

/**
 * ADR-3 cluster placement: nodes orbit their section anchor's camera target on
 * a golden-angle spiral (even angular coverage at any count) plus seeded
 * jitter keyed on the node id. Pure and deterministic — safe at module load.
 */

const GOLDEN_ANGLE = Math.PI * (3 - Math.sqrt(5));

interface ClusterOptions {
  /** Base orbit radius around the anchor target. */
  radius?: number;
  /** Max random displacement applied per axis. */
  jitter?: number;
  /** Vertical spread of the cluster disc. */
  depth?: number;
}

export function placeInCluster(
  anchor: SectionAnchor,
  index: number,
  total: number,
  seedId: string,
  options: ClusterOptions = {},
): Vec3 {
  const { radius = 4, jitter = 0.9, depth = 2.5 } = options;
  const rand = seededRandom(seedId);
  const [cx, cy, cz] = anchor.cameraTarget;

  // Spiral: radius grows with sqrt(index/total) for even areal density.
  const angle = index * GOLDEN_ANGLE;
  const ring = radius * Math.sqrt((index + 0.5) / Math.max(total, 1));

  const x = cx + ring * Math.cos(angle) + (rand() - 0.5) * 2 * jitter;
  const y = cy + (rand() - 0.5) * depth + (rand() - 0.5) * 2 * jitter * 0.5;
  const z = cz + ring * Math.sin(angle) * 0.6 + (rand() - 0.5) * 2 * jitter;

  return [round3(x), round3(y), round3(z)];
}

/** Round to 3 decimals so SSR/client serialization can never drift. */
function round3(value: number): number {
  return Math.round(value * 1000) / 1000;
}
