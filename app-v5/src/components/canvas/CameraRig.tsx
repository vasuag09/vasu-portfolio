"use client";

import { useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import {
  buildCameraSpline,
  sampleCameraPose,
  progressToSplineParam,
  type CameraPose,
} from "@/lib/camera-spline";
import { scrollState } from "@/lib/scroll-state";
import { cameraDiveState, lastCameraPose } from "@/lib/camera-state";
import { dampTowards } from "@/lib/camera-dive";
import { stepFov, targetFov } from "@/lib/velocity-fov";
import { useReducedMotion } from "@/hooks/useReducedMotion";
import { sections } from "@/data/sections-v5";

/**
 * ADR-1 frame sync: Lenis/ScrollTrigger have already written progress into
 * scrollState by the time useFrame runs; we sample the spline and write the
 * camera directly. All smoothing lives upstream (Lenis lerp + scrub) plus the
 * per-segment smoothstep — no extra lag here, keeping skew under one frame.
 *
 * ADR-9 Neuron Dive: the scroll pose is computed EVERY frame, then blended
 * toward cameraDiveState.divePose by an exp-damped diveBlend. Single writer,
 * blended authority — open/close/rapid-toggle are blend retargets only.
 */

/** Damping rate for the dive blend (~9/s ≈ visually settled in ~500ms). */
const DIVE_RATE = 7;

export function CameraRig() {
  const reducedMotion = useReducedMotion();
  const spline = useMemo(() => buildCameraSpline(sections), []);
  const poseRef = useRef<CameraPose>({
    position: new THREE.Vector3(...sections[0].cameraPos),
    target: new THREE.Vector3(...sections[0].cameraTarget),
  });
  const blendedTarget = useRef(new THREE.Vector3());

  useFrame(({ camera }, delta) => {
    const centers = scrollState.sectionCenters;
    if (centers.length < 2) return;

    const param = progressToSplineParam(scrollState.progress, centers);
    const pose = sampleCameraPose(spline, param, poseRef.current);

    // Dive blend: damped toward its target; pose math only while active.
    const dive = cameraDiveState;
    dive.diveBlend = dampTowards(
      dive.diveBlend,
      dive.diveTargetBlend,
      delta,
      DIVE_RATE,
    );
    if (dive.diveBlend < 1e-3 && dive.diveTargetBlend === 0) {
      dive.diveBlend = 0;
      dive.divePose = null;
    }

    if (dive.divePose && dive.diveBlend > 0) {
      camera.position.lerpVectors(
        pose.position,
        dive.divePose.position,
        dive.diveBlend,
      );
      blendedTarget.current.lerpVectors(
        pose.target,
        dive.divePose.target,
        dive.diveBlend,
      );
      camera.lookAt(blendedTarget.current);
      lastCameraPose.target.copy(blendedTarget.current);
    } else {
      camera.position.copy(pose.position);
      camera.lookAt(pose.target);
      lastCameraPose.target.copy(pose.target);
    }
    lastCameraPose.position.copy(camera.position);

    // Velocity FOV (ADR-9 wave 2): fast scroll widens the lens (≤ +6°),
    // suppressed during a dive / reduced motion. Orthogonal to position +
    // target authority above — CameraRig stays the single camera writer.
    stepFov(
      camera as THREE.PerspectiveCamera,
      targetFov({
        velocity: scrollState.velocity,
        diveBlend: dive.diveBlend,
        reducedMotion,
      }),
      delta,
    );
  });

  return null;
}
