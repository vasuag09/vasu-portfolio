"use client";

import { useState } from "react";
import { Canvas } from "@react-three/fiber";
import { CameraRig } from "./CameraRig";
import { NeuralNetwork } from "./NeuralNetwork";
import { NodeLabels } from "./NodeLabels";
import { SignalDriver } from "./SignalDriver";
import { SynapseNode } from "./SynapseNode";
import { Particles } from "./Particles";
import { ProbeDriver } from "./ProbeDriver";
import { ProjectWorldEnv } from "./ProjectWorldEnv";
import { Effects } from "./Effects";
import { DebugSpline } from "./DebugSpline";
import { useParticleConfig } from "@/hooks/useParticleConfig";
import { sections } from "@/data/sections-v5";
import { SCENE_COLORS } from "@/lib/scene-colors";

/**
 * The R3F canvas (ADR-4): transparent so the CSS token background shows
 * through; camera starts at the hero rest-pose and is driven exclusively by
 * CameraRig. Scene content scales with the ADR-2 device tier. MSAA stays
 * off — the EffectComposer renders to its own buffers and the glow
 * aesthetic hides edge aliasing.
 */
export default function SceneCanvas() {
  const tier = useParticleConfig();

  // Read once on mount; client-only component so location is available.
  const [debug] = useState(
    () =>
      typeof window !== "undefined" &&
      new URLSearchParams(window.location.search).has("debug"),
  );
  // ?noeffects / ?nodof: bypass parts of the post pipeline to isolate issues.
  const [noEffects] = useState(
    () =>
      typeof window !== "undefined" &&
      new URLSearchParams(window.location.search).has("noeffects"),
  );
  const [noDof] = useState(
    () =>
      typeof window !== "undefined" &&
      new URLSearchParams(window.location.search).has("nodof"),
  );

  return (
    <Canvas
      camera={{ position: sections[0].cameraPos, fov: 50, near: 0.1, far: 200 }}
      // Opaque canvas: the EffectComposer composite writes garbage alpha on a
      // transparent canvas (scene vanishes against the page bg). The scene
      // paints its own background, matching the --bg-base token.
      gl={{ antialias: false, alpha: false, powerPreference: "high-performance" }}
      dpr={[1, tier.maxDpr]}
      onCreated={(state) => {
        if (debug) {
          // Debug-only introspection handle (?debug): used by phase verification.
          (window as unknown as Record<string, unknown>).__r3fState = state;
        }
      }}
    >
      {/* Adaptive scroll-DPR was tried and REMOVED: each setDpr resize
          reallocates the composer target (~175ms spike) — costlier than
          the frames it saved. MSAA 4x + DPR 1.75 + no Bokeh already hold
          the budget (p95 9.2ms measured). */}
      <color attach="background" args={[SCENE_COLORS.background]} />
      <CameraRig />
      <SignalDriver />
      <NeuralNetwork />
      {/* ADR-8 labels: desktop tier only (small screens have no room). */}
      {tier.name === "desktop" ? <NodeLabels /> : null}
      <SynapseNode />
      <Particles count={tier.particleCount} />
      {/* Wave 2: cursor probe writes pointer uniforms; ProjectWorldEnv
          cross-fades the world palette on dive arrival. Both write only
          signalUniforms (read by Particles' shader), never the camera. */}
      <ProbeDriver />
      <ProjectWorldEnv />
      {/* Mount Effects ONLY when the tier wants bloom — its prioritized
          useFrame takes over rendering entirely (see Effects.tsx). */}
      {noEffects || !tier.bloom ? null : (
        <Effects tier={noDof ? { ...tier, depthOfField: false } : tier} />
      )}
      {debug ? <DebugSpline /> : null}
    </Canvas>
  );
}
