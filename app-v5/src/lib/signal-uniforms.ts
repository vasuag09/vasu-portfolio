import * as THREE from "three";

/**
 * Signal Pulse uniform channel — ONE set of uniform objects shared by
 * reference between the particle material, the connection material, and
 * SignalDriver (which mutates the values each frame). Sharing references
 * means zero prop drilling and a single write lights both materials.
 */
export const signalUniforms = {
  /** Pulse world position — travels the target spline ahead of scroll. */
  uPulsePos: { value: new THREE.Vector3(0, 0, 0) },
  /** Pulse visibility 0..1 (0 while idle/reduced tiers). */
  uPulseStrength: { value: 0 },
  /** Section-core position of the most recent chapter arrival. */
  uFirePos: { value: new THREE.Vector3(0, -100, 0) },
  /** Arrival flare 0..1, exp-decayed by SignalDriver (≤700ms visible). */
  uFireStrength: { value: 0 },
};
