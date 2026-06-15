import * as THREE from "three";
import { projectNodes } from "@/data/projects-v5";
import { sections } from "@/data/sections-v5";
import { deriveDivePose } from "./camera-dive";
import type { CameraPose } from "./camera-spline";

/**
 * Neuron Dive channel (ADR-9) — same deliberate-mutation exception as
 * scroll-state: DOM components write the dive intent, CameraRig reads it in
 * useFrame and blends. CameraRig stays the ONLY camera writer; rapid
 * open/close is just blend retargeting, the camera can never be stranded.
 */

export interface CameraDiveState {
  /** Where the dive wants the camera; null = no dive requested. */
  divePose: CameraPose | null;
  /** 0 = pure scroll pose, 1 = fully inside the node. Damped by CameraRig. */
  diveBlend: number;
  /** Blend target: 1 while a dive is active, 0 while returning. */
  diveTargetBlend: number;
}

export const cameraDiveState: CameraDiveState = {
  divePose: null,
  diveBlend: 0,
  diveTargetBlend: 0,
};

const NODE_SCALE_FLAGSHIP = 0.52;
const NODE_SCALE_ARCHIVE = 0.24;

/**
 * Start (or retarget) a dive into a project's node. `instant` lands at the
 * end state with no flight — deep links must not replay (spec).
 */
export function startDive(projectId: string, instant = false): void {
  const node = projectNodes.find((n) => n.id === projectId);
  if (!node) return;
  const scale = node.scale * (node.flagship ? NODE_SCALE_FLAGSHIP : NODE_SCALE_ARCHIVE);
  // Rest pose for the dive direction: where the camera is NOW is the most
  // honest origin — CameraRig records it every frame. Before the rig's
  // first frame (instant deep links) it holds the hero rest pose, which is
  // a best-effort origin: the dive direction is cosmetic in the instant
  // case (no flight is shown).
  const pose = deriveDivePose(node.position, scale, {
    position: lastCameraPose.position,
    target: lastCameraPose.target,
  });
  // NaN backstop (review finding): a poisoned pose would propagate through
  // CameraRig's lerp straight into camera.position — refuse it instead.
  if (
    !Number.isFinite(pose.position.x + pose.position.y + pose.position.z) ||
    !Number.isFinite(pose.target.x + pose.target.y + pose.target.z)
  ) {
    return;
  }
  cameraDiveState.divePose = pose;
  cameraDiveState.diveTargetBlend = 1;
  if (instant) cameraDiveState.diveBlend = 1;
}

/** Reverse the dive; CameraRig damps the blend back to the scroll pose. */
export function endDive(instant = false): void {
  cameraDiveState.diveTargetBlend = 0;
  if (instant) {
    cameraDiveState.diveBlend = 0;
    cameraDiveState.divePose = null;
  }
}

/**
 * Live camera pose, recorded by CameraRig each frame so dive derivation and
 * future effects can read "where the camera actually is" without touching
 * the three camera object from DOM-land.
 */
export const lastCameraPose: CameraPose = {
  // Hero rest pose until CameraRig's first frame — instant deep-link dives
  // need a sane origin before the rig has ever run.
  position: new THREE.Vector3(...sections[0].cameraPos),
  target: new THREE.Vector3(...sections[0].cameraTarget),
};
