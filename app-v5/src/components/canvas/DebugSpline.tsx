"use client";

import { useMemo } from "react";
import * as THREE from "three";
import { Line } from "@react-three/drei";
import { sections } from "@/data/sections-v5";
import { buildCameraSpline, sampleCameraPose } from "@/lib/camera-spline";
import { SCENE_COLORS } from "@/lib/scene-colors";

/** ?debug only: visualizes the camera flight path (ADR-3 risk mitigation). */
export function DebugSpline() {
  const points = useMemo(() => {
    const spline = buildCameraSpline(sections);
    const samples: THREE.Vector3[] = [];
    for (let i = 0; i <= 120; i += 1) {
      samples.push(sampleCameraPose(spline, i / 120).position.clone());
    }
    return samples;
  }, []);

  return <Line points={points} color={SCENE_COLORS.accentBright} lineWidth={1} />;
}
