import * as THREE from "three";

import { SCENE_COLORS } from "./scene-colors";

/**
 * Signal Pulse uniform channel — ONE set of uniform objects shared by
 * reference between the particle material, the connection material, and
 * SignalDriver (which mutates the values each frame). Sharing references
 * means zero prop drilling and a single write lights both materials.
 *
 * Wave 2 (ADR-9 probe, ADR-10 worlds) extends this same channel: ProbeDriver
 * writes the probe uniforms and ProjectWorldEnv writes the world uniforms, so
 * one singleton lights every consumer by reference. World colour defaults are
 * the BASE palette, making uWorldBlend = 0 a true no-op.
 *
 * WARNING: never material.clone() a material holding these — THREE's clone
 * shallow-copies uniform OBJECTS, so the clone would stop seeing
 * SignalDriver's writes. Both consumers are module-singleton materials.
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

  /** Cursor probe world position — ProbeDriver unprojects the pointer here. */
  uPointerPos: { value: new THREE.Vector3(0, 0, 0) },
  /** Probe visibility 0..1 (0 on touch / reduced motion / off). */
  uProbeStrength: { value: 0 },

  /** World cross-fade 0..1 — ProjectWorldEnv damps this on dive arrival. */
  uWorldBlend: { value: 0 },
  /** World palette, defaulting to the base scene palette (blend = 0 no-op). */
  uWorldColor1: { value: new THREE.Color(SCENE_COLORS.accent) },
  uWorldColor2: { value: new THREE.Color(SCENE_COLORS.accentBright) },
  uWorldAccent: { value: new THREE.Color(SCENE_COLORS.accentBright) },
};
