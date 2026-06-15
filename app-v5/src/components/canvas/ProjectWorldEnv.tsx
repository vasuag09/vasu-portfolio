"use client";

import { useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";

import { cameraDiveState } from "@/lib/camera-state";
import { dampTowards } from "@/lib/camera-dive";
import { getGraphState } from "@/lib/graph-store";
import { SCENE_COLORS } from "@/lib/scene-colors";
import { signalUniforms } from "@/lib/signal-uniforms";
import { getWorldDefinition } from "@/lib/world-registry";

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
/** Per-frame palette colour creep toward the target. */
const COLOR_LERP = 0.08;

// Base palette = the no-world target (blend rides to 0, so colour is masked).
const BASE_PRIMARY = new THREE.Color(SCENE_COLORS.accent);
const BASE_SECONDARY = new THREE.Color(SCENE_COLORS.accentBright);
const BASE_ACCENT = new THREE.Color(SCENE_COLORS.accentBright);

export function ProjectWorldEnv() {
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

    signalUniforms.uWorldColor1.value.lerp(target.primary, COLOR_LERP);
    signalUniforms.uWorldColor2.value.lerp(target.secondary, COLOR_LERP);
    signalUniforms.uWorldAccent.value.lerp(target.accent, COLOR_LERP);
  });

  return null;
}
