import * as THREE from "three";
import type { SectionAnchor } from "@/data/types";

/**
 * ADR-3: camera spline = CubicBezierCurve3 chain through section anchor
 * rest-poses, built once by buildCameraSpline(anchors). Control handles come
 * from Catmull-Rom tangents so the chain is C1-continuous — no velocity jumps
 * at section joins. Sampling is per-segment (t = i/(n-1) lands EXACTLY on
 * anchor i), with a smoothstep inside each segment so the camera decelerates
 * into every rest-pose instead of drifting through it.
 */

export interface CameraPose {
  position: THREE.Vector3;
  target: THREE.Vector3;
}

export interface CameraSpline {
  positionSegments: readonly THREE.CubicBezierCurve3[];
  targetSegments: readonly THREE.CubicBezierCurve3[];
  segmentCount: number;
}

function toVectors(points: readonly [number, number, number][]): THREE.Vector3[] {
  return points.map(([x, y, z]) => new THREE.Vector3(x, y, z));
}

/** Catmull-Rom tangents → cubic Bézier chain through every point. */
function buildBezierChain(points: THREE.Vector3[]): THREE.CubicBezierCurve3[] {
  const n = points.length;
  const tangents = points.map((_, i) => {
    const prev = points[Math.max(0, i - 1)];
    const next = points[Math.min(n - 1, i + 1)];
    // One-sided difference at the ends, central difference inside.
    const scale = i === 0 || i === n - 1 ? 1 : 0.5;
    return next.clone().sub(prev).multiplyScalar(scale);
  });

  const segments: THREE.CubicBezierCurve3[] = [];
  for (let i = 0; i < n - 1; i += 1) {
    segments.push(
      new THREE.CubicBezierCurve3(
        points[i].clone(),
        points[i].clone().addScaledVector(tangents[i], 1 / 3),
        points[i + 1].clone().addScaledVector(tangents[i + 1], -1 / 3),
        points[i + 1].clone(),
      ),
    );
  }
  return segments;
}

export function buildCameraSpline(
  anchors: readonly SectionAnchor[],
): CameraSpline {
  if (anchors.length < 2) {
    throw new Error("buildCameraSpline needs at least two anchors");
  }
  return {
    positionSegments: buildBezierChain(
      toVectors(anchors.map((a) => a.cameraPos)),
    ),
    targetSegments: buildBezierChain(
      toVectors(anchors.map((a) => a.cameraTarget)),
    ),
    segmentCount: anchors.length - 1,
  };
}

function smoothstep(u: number): number {
  return u * u * (3 - 2 * u);
}

/**
 * Sample the camera pose at t ∈ [0,1]. Pass `out` on the useFrame hot path —
 * the pose is written in place and the same instance returned, so steady-state
 * scrolling allocates nothing per frame.
 */
export function sampleCameraPose(
  spline: CameraSpline,
  t: number,
  out: CameraPose = {
    position: new THREE.Vector3(),
    target: new THREE.Vector3(),
  },
): CameraPose {
  const clamped = Math.min(1, Math.max(0, t));
  const scaled = clamped * spline.segmentCount;
  const index = Math.min(Math.floor(scaled), spline.segmentCount - 1);
  const local = smoothstep(scaled - index);

  spline.positionSegments[index].getPoint(local, out.position);
  spline.targetSegments[index].getPoint(local, out.target);
  return out;
}

/**
 * Map normalized document-scroll progress to spline parameter. `centers` are
 * the section midpoints in scroll space (measured from the DOM, ascending).
 * Piecewise-linear through (center_i → i/(n-1)): the camera rests exactly at
 * anchor i when section i is centered in the viewport, and flies between
 * anchors while the scroll travels between sections.
 */
export function progressToSplineParam(
  progress: number,
  centers: readonly number[],
): number {
  const n = centers.length;
  if (n < 2) return 0;
  if (progress <= centers[0]) return 0;
  if (progress >= centers[n - 1]) return 1;

  let i = 0;
  while (i < n - 2 && progress > centers[i + 1]) i += 1;
  const span = centers[i + 1] - centers[i];
  const local = span > 0 ? (progress - centers[i]) / span : 0;
  return (i + local) / (n - 1);
}
