"use client";

import { Suspense, useRef, useState } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";

import { cameraDiveState } from "@/lib/camera-state";
import { dampTowards } from "@/lib/camera-dive";
import { getGraphState } from "@/lib/graph-store";
import { SCENE_COLORS } from "@/lib/scene-colors";
import { signalUniforms } from "@/lib/signal-uniforms";
import { getWorldDefinition } from "@/lib/world-registry";
import { useGraphState } from "@/hooks/useGraphState";

/**
 * Project World cross-fade animator (ADR-10). Sole owner of the world
 * uniforms. Reads diveBlend + selectedProjectId (module singletons, no React
 * subscription in the frame loop) and damps uWorldBlend toward 1 while diving
 * into a project that HAS a bespoke world, 0 otherwise; palette colours creep
 * toward the active world. Single damped value, retargets on project switch —
 * never stranded, base palette always live underneath (the ADR-9 idiom).
 *
 * Writes ONLY signalUniforms — never the camera (ADR-9 orthogonality). Target
 * palette lives on a ref (frame-loop mutation, React Compiler escape hatch).
 * The motif Suspense boundary is added with the flagship motifs (step 7).
 */

/** diveBlend above this counts as "inside a world". */
const DIVE_ACTIVE = 0.05;
/** uWorldBlend damping rate (~settles in ~600ms). */
const WORLD_FADE_RATE = 4;
/** Palette colour damping rate — slightly ahead of the blend so colour is
 *  ready as the cross-fade ramps. Frame-rate independent (see useFrame). */
const COLOR_RATE = 5;

// Base palette = the no-world target (blend rides to 0, so colour is masked).
const BASE_PRIMARY = new THREE.Color(SCENE_COLORS.accent);
const BASE_SECONDARY = new THREE.Color(SCENE_COLORS.accentBright);
const BASE_ACCENT = new THREE.Color(SCENE_COLORS.accentBright);

export function ProjectWorldEnv() {
  // React subscription (selection changes are rare) drives the motif mount.
  const { selectedProjectId } = useGraphState();
  const motifId =
    selectedProjectId && getWorldDefinition(selectedProjectId)?.motifComponent
      ? selectedProjectId
      : null;

  // Mount as soon as a motif-world is selected (starts the lazy import). The
  // render-phase setState pattern: only fires on the transition.
  const [mounted, setMounted] = useState<string | null>(null);
  if (motifId && motifId !== mounted) setMounted(motifId);

  const targetRef = useRef({
    primary: BASE_PRIMARY.clone(),
    secondary: BASE_SECONDARY.clone(),
    accent: BASE_ACCENT.clone(),
    activeId: null as string | null,
  });

  useFrame((_, delta) => {
    const target = targetRef.current;
    const { diveBlend } = cameraDiveState;
    const { selectedProjectId } = getGraphState();
    const diving = diveBlend > DIVE_ACTIVE && Boolean(selectedProjectId);

    const id = diving ? selectedProjectId : null;
    const def = getWorldDefinition(id);

    // Retarget palette only on a change — immediate swap, blend keeps creeping
    // (no fade-out/in flicker on project switch).
    if (id !== target.activeId) {
      target.activeId = id;
      if (def) {
        target.primary.set(def.palette.primary);
        target.secondary.set(def.palette.secondary);
        target.accent.set(def.palette.accent);
      } else {
        target.primary.copy(BASE_PRIMARY);
        target.secondary.copy(BASE_SECONDARY);
        target.accent.copy(BASE_ACCENT);
      }
    }

    // Only a project WITH a bespoke world fades in; palette-only / no-dive → 0.
    signalUniforms.uWorldBlend.value = dampTowards(
      signalUniforms.uWorldBlend.value,
      def ? 1 : 0,
      delta,
      WORLD_FADE_RATE,
    );

    // Frame-rate-independent colour creep — same exp form as dampTowards, so
    // the palette settles identically at 30 and 60fps (a fixed per-frame lerp
    // would creep twice as fast at 60fps).
    const colorAlpha = 1 - Math.exp(-COLOR_RATE * delta);
    signalUniforms.uWorldColor1.value.lerp(target.primary, colorAlpha);
    signalUniforms.uWorldColor2.value.lerp(target.secondary, colorAlpha);
    signalUniforms.uWorldAccent.value.lerp(target.accent, colorAlpha);

    // Unmount the motif only once it has fully faded out — deferring past the
    // dive-exit avoids popping it while the camera is still pulling back.
    // Edge-triggered: the `mounted` guard means this fires at most once.
    if (mounted && !motifId && signalUniforms.uWorldBlend.value < 0.01) {
      setMounted(null);
    }
  });

  const Motif = mounted ? getWorldDefinition(mounted)?.motifComponent : null;

  // Motif-only Suspense boundary (ADR-10): a lazy motif yields null while
  // loading, so the persistent particle field never blanks.
  return Motif ? (
    <Suspense fallback={null}>
      <Motif />
    </Suspense>
  ) : null;
}
