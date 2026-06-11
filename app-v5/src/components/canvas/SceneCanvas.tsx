"use client";

import { useState } from "react";
import { Canvas } from "@react-three/fiber";
import { CameraRig } from "./CameraRig";
import { PlaceholderScene } from "./PlaceholderScene";
import { sections } from "@/data/sections-v5";

/**
 * The R3F canvas (ADR-4): transparent so the CSS token background shows
 * through; camera starts at the hero rest-pose and is driven exclusively by
 * CameraRig.
 */
export default function SceneCanvas() {
  // Read once on mount; client-only component so location is available.
  const [debug] = useState(
    () =>
      typeof window !== "undefined" &&
      new URLSearchParams(window.location.search).has("debug"),
  );

  return (
    <Canvas
      camera={{ position: sections[0].cameraPos, fov: 50, near: 0.1, far: 200 }}
      gl={{ antialias: true, alpha: true, powerPreference: "high-performance" }}
      dpr={[1, 2]}
      onCreated={(state) => {
        if (debug) {
          // Debug-only introspection handle (?debug): used by Phase-1 verification.
          (window as unknown as Record<string, unknown>).__r3fState = state;
        }
      }}
    >
      <CameraRig />
      <PlaceholderScene debug={debug} />
    </Canvas>
  );
}
