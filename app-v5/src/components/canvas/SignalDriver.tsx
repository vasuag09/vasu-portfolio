"use client";

import { useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import {
  buildCameraSpline,
  progressToSplineParam,
  sampleCameraPose,
  type CameraPose,
} from "@/lib/camera-spline";
import {
  createArrivalState,
  detectArrival,
  pulseProgress,
  type ArrivalState,
} from "@/lib/signal-state";
import { signalUniforms } from "@/lib/signal-uniforms";
import { scrollState } from "@/lib/scroll-state";
import { soundEngine } from "@/lib/sound-engine";
import { sections } from "@/data/sections-v5";

/** Flare decay rate (1/s): visible ~600ms, inside the spec's 700ms cap. */
const FIRE_DECAY = 6;

/**
 * Signal Pulse driver (Living Network wave 1): every frame, place the pulse
 * on the camera-target spline slightly AHEAD of scroll and run the arrival
 * FSM — entering a chapter band fires that section's core (particle/edge
 * flare via shared uniforms + per-chapter chord). Zero draw calls of its
 * own; it only writes the uniform channel both materials already read.
 */
export function SignalDriver() {
  const spline = useMemo(() => buildCameraSpline(sections), []);
  const poseRef = useRef<CameraPose>({
    position: new THREE.Vector3(),
    target: new THREE.Vector3(),
  });
  const arrivalRef = useRef<ArrivalState>(createArrivalState());

  useFrame((_, delta) => {
    const centers = scrollState.sectionCenters;
    if (centers.length < 2) return;

    // Pulse rides the TARGET spline (where cores and the network live).
    const param = progressToSplineParam(
      pulseProgress(scrollState.progress),
      centers,
    );
    const pose = sampleCameraPose(spline, param, poseRef.current);
    signalUniforms.uPulsePos.value.copy(pose.target);
    signalUniforms.uPulseStrength.value = 1;

    // Arrival: fire the entered chapter's core, then exp-decay the flare.
    const [next, fired] = detectArrival(
      arrivalRef.current,
      scrollState.progress,
      centers,
    );
    arrivalRef.current = next;
    if (fired !== null) {
      signalUniforms.uFirePos.value.set(...sections[fired].cameraTarget);
      signalUniforms.uFireStrength.value = 1;
      soundEngine.playArrival(fired);
    } else {
      signalUniforms.uFireStrength.value *= Math.exp(-FIRE_DECAY * delta);
    }
  });

  return null;
}
