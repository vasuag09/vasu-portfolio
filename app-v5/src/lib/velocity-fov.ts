import { dampTowards } from "./camera-dive";

/**
 * Velocity FOV (Living wave 2) — fast scroll widens the lens a touch for a
 * "leaning into speed" feel. Pure target mapping lives here; CameraRig damps
 * the camera toward it inside its existing useFrame, so the camera stays a
 * single-writer system (ADR-9).
 *
 * Hard ceiling +6°: the neuron dive owns the camera, and a wide lens
 * fighting the dive blend would read as a wobble. So the boost is both
 * capped AND fully suppressed whenever a dive is active or motion is reduced.
 */

export const BASE_FOV = 50; // matches SceneCanvas camera init (fov: 50)
export const MAX_FOV_BOOST = 6; // ADR-9 ceiling: never wider than +6°
/**
 * Lenis velocity that maps to the full boost; beyond this the boost clamps.
 * Tuned against real Lenis velocity during the phase-2 CameraRig wiring.
 */
export const FOV_VELOCITY_REF = 12;

interface FovInputs {
  /** Lenis scroll velocity (signed); magnitude drives the boost. */
  velocity: number;
  /** Neuron-dive blend 0..1 — any active dive suppresses the boost. */
  diveBlend: number;
  reducedMotion: boolean;
}

/** Target FOV for the current scroll velocity, with both opt-outs applied. */
export function targetFov({ velocity, diveBlend, reducedMotion }: FovInputs): number {
  // Dive owns the camera; reduced motion opts out of decorative motion.
  if (reducedMotion || diveBlend > 0) return BASE_FOV;
  // NaN backstop (mirrors camera-state startDive): a poisoned velocity must
  // never reach camera.fov / updateProjectionMatrix.
  if (!Number.isFinite(velocity)) return BASE_FOV;
  const t = Math.min(Math.abs(velocity), FOV_VELOCITY_REF) / FOV_VELOCITY_REF;
  return BASE_FOV + t * MAX_FOV_BOOST;
}

/** Damp rate for the FOV approach (~settles in ~600ms). */
const FOV_DAMP_RATE = 3.5;
/** Below this delta the lens is settled — skip the projection refresh. */
const FOV_EPSILON = 1e-4;

interface FovCamera {
  fov: number;
  updateProjectionMatrix: () => void;
}

/**
 * Damp the camera's FOV toward `target` and refresh the projection only when
 * the change is meaningful. Skipping a no-op updateProjectionMatrix keeps the
 * lens from doing per-frame matrix work once it has settled.
 */
export function stepFov(camera: FovCamera, target: number, delta: number): void {
  const next = dampTowards(camera.fov, target, delta, FOV_DAMP_RATE);
  if (Math.abs(next - camera.fov) < FOV_EPSILON) return;
  camera.fov = next;
  camera.updateProjectionMatrix();
}
