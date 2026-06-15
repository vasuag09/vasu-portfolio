"use client";

import { useEffect, useRef } from "react";
import { useFrame, useThree } from "@react-three/fiber";
import * as THREE from "three";
import { EffectComposer } from "three/examples/jsm/postprocessing/EffectComposer.js";
import { RenderPass } from "three/examples/jsm/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/examples/jsm/postprocessing/UnrealBloomPass.js";
import { OutputPass } from "three/examples/jsm/postprocessing/OutputPass.js";
import type { ParticleTier } from "@/lib/particle-config";

/**
 * Post pipeline per ADR-2 tier, built on three's OWN addons composer —
 * the postprocessing/@react-three/postprocessing stack silently rendered
 * black against this three/R3F combination (composer ran, EffectPass output
 * nothing, zero errors). The addons ship inside the three package itself,
 * so renderer/addon version skew cannot happen.
 *
 * Bloom stays SELECTIVE: threshold 1.0 only passes HDR output (flagship
 * nodes, section cores, signal particles with intensity > 1) — the dim
 * field never blooms (AWARD-RESEARCH §5 restraint). BokehPass (DoF) runs
 * on the desktop tier only.
 *
 * IMPORTANT: only mount this component when the tier wants effects — the
 * prioritized useFrame suppresses R3F's default render unconditionally.
 */
export function Effects({ tier }: { tier: ParticleTier }) {
  const gl = useThree((s) => s.gl);
  const scene = useThree((s) => s.scene);
  const camera = useThree((s) => s.camera);
  const size = useThree((s) => s.size);
  const composerRef = useRef<EffectComposer | null>(null);

  useEffect(() => {
    // MSAA happens HERE, not on the canvas: with a composer, the scene is
    // drawn into this target, so `samples` on it is the only anti-aliasing
    // that counts. HalfFloat keeps HDR headroom for the bloom threshold.
    const drawSize = gl.getDrawingBufferSize(new THREE.Vector2());
    // 4x MSAA (design-elevation P2): at these dot sizes 8x was visually
    // indistinguishable and the single largest fill cost. Before/after
    // screenshots in docs/v5/design-audit/after/ back the downgrade.
    const msaaTarget = new THREE.WebGLRenderTarget(drawSize.x, drawSize.y, {
      type: THREE.HalfFloatType,
      samples: 4,
    });
    const composer = new EffectComposer(gl, msaaTarget);
    composer.addPass(new RenderPass(scene, camera));

    // BokehPass removed (design-elevation P2 decision): full-res DoF barely
    // read in screenshots and cost a full-scene depth pass per frame.
    // tier.depthOfField stays in the config schema for a future half-res
    // implementation if Phase 11 wants it back.

    composer.addPass(
      new UnrealBloomPass(
        new THREE.Vector2(size.width, size.height),
        0.55, // strength
        0.55, // radius
        1.0, // threshold: only HDR (>1) output blooms — selective by design
      ),
    );

    // Tone mapping + sRGB conversion happen here, after bloom ran in HDR.
    composer.addPass(new OutputPass());

    composerRef.current = composer;
    if (new URLSearchParams(window.location.search).has("debug")) {
      (window as unknown as Record<string, unknown>).__composer = composer;
    }

    return () => {
      composerRef.current = null;
      composer.dispose();
    };
    // size is handled by the dedicated resize effect below.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [gl, scene, camera, tier]);

  useEffect(() => {
    composerRef.current?.setSize(size.width, size.height);
  }, [size]);

  // Priority 1: R3F sees a prioritized subscriber and skips its own render —
  // the composer becomes the sole path to the screen.
  useFrame(() => {
    composerRef.current?.render();
  }, 1);

  return null;
}
