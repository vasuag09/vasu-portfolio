import * as THREE from "three";
import type { CameraPose } from "./camera-spline";
import type { Vec3 } from "@/data/types";

/**
 * Neuron Dive pose math (ADR-9). Pure and deterministic: the dive camera
 * sits on the line from the node toward the current rest pose, pulled in
 * close — entering the node's glow, not orbiting it.
 */

/** Camera distance = node visual radius × this factor… */
const NEAR_FACTOR = 7;
/** …but never closer than this (avoids clipping into the sphere). */
const NEAR_FLOOR = 1.2;

export function deriveDivePose(
  nodePosition: Vec3,
  nodeScale: number,
  restPose: { position: THREE.Vector3; target: THREE.Vector3 },
): CameraPose {
  const node = new THREE.Vector3(...nodePosition);
  const back = restPose.position.clone().sub(node);
  if (back.lengthSq() < 1e-6) {
    // Degenerate: rest pose at the node — back off along +Z.
    back.set(0, 0, 1);
  }
  back.normalize();
  const distance = Math.max(NEAR_FLOOR, nodeScale * NEAR_FACTOR);
  return {
    position: node.clone().addScaledVector(back, distance),
    target: node,
  };
}

/**
 * Frame-rate-independent exponential damping (the codebase's standard
 * easing idiom — same family as the activation glow). Never overshoots;
 * retargeting mid-flight is just a new target.
 */
export function dampTowards(
  current: number,
  target: number,
  dt: number,
  rate: number,
): number {
  return current + (target - current) * (1 - Math.exp(-rate * dt));
}
