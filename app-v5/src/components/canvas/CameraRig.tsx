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
import { sections } from "@/data/sections-v5";

/**
 * ADR-1 frame sync: Lenis/ScrollTrigger have already written progress into
 * scrollState by the time useFrame runs; we sample the spline and write the
 * camera directly. All smoothing lives upstream (Lenis lerp + scrub) plus the
 * per-segment smoothstep — no extra lag here, keeping skew under one frame.
 */
export function CameraRig() {
  const spline = useMemo(() => buildCameraSpline(sections), []);
  const poseRef = useRef<CameraPose>({
    position: new THREE.Vector3(...sections[0].cameraPos),
    target: new THREE.Vector3(...sections[0].cameraTarget),
  });

  useFrame(({ camera }) => {
    const centers = scrollState.sectionCenters;
    if (centers.length < 2) return;

    const param = progressToSplineParam(scrollState.progress, centers);
    const pose = sampleCameraPose(spline, param, poseRef.current);
    camera.position.copy(pose.position);
    camera.lookAt(pose.target);
  });

  return null;
}
