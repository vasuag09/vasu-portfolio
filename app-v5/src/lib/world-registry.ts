import type { ComponentType, LazyExoticComponent } from "react";

import { PROJECT_WORLD_COLORS, type WorldPalette } from "./scene-colors";

/**
 * Project World registry (ADR-10) — keyed by projectId directly. A dive into
 * a project that has a definition cross-fades into its bespoke world; projects
 * registered as null fall back to the base scene palette (no override, no
 * motif). All seven dive-able projects are derived from WORLD_PROJECT_IDS so
 * a missing world is a structural null, never a silent gap.
 */

/** Props every lazy motif receives from ProjectWorldEnv. */
export interface WorldMotifProps {
  /** 0..1 world cross-fade (mirrors signalUniforms.uWorldBlend). */
  worldBlend: number;
  /** 0..1 neuron-dive blend. */
  diveBlend: number;
}

export interface WorldDefinition {
  /** Matches the projectId. */
  id: string;
  /** Palette the particle field blends toward during the cross-fade. */
  palette: WorldPalette;
  /**
   * Particle-shader knobs to ease toward while the world is active; absent
   * fields keep the scene default. Wired into the shader at phase-2 step 6.
   */
  particleOverride?: {
    noiseScale?: number;
    flowSpeed?: number;
    turbulence?: number;
    proximityDamping?: number;
  };
  /**
   * Bespoke 3D motif, lazy-loaded under a motif-only Suspense boundary.
   * Added with the flagship motif components (phase-2 step 7).
   */
  motifComponent?: LazyExoticComponent<ComponentType<WorldMotifProps>>;
}

/**
 * Every dive-able project, in graph order — the single source for "what
 * worlds exist". Must stay in sync with projectNodes in data/projects-v5.ts.
 */
export const WORLD_PROJECT_IDS = [
  "fundlymart",
  "nm-gpt",
  "ray-serve",
  "insightify",
  "geovision",
  "harness-claude",
  "streamlit-oss",
] as const;

/** Flagship worlds with bespoke definitions; everything else is palette-only. */
const FLAGSHIP_WORLDS: Record<string, WorldDefinition> = {
  fundlymart: { id: "fundlymart", palette: PROJECT_WORLD_COLORS.fundlymart },
  "nm-gpt": { id: "nm-gpt", palette: PROJECT_WORLD_COLORS["nm-gpt"] },
};

/** Derived so all seven ids are registered; non-flagships resolve to null. */
const WORLD_REGISTRY: Record<string, WorldDefinition | null> = Object.fromEntries(
  WORLD_PROJECT_IDS.map((id) => [id, FLAGSHIP_WORLDS[id] ?? null]),
);

/**
 * Active world for a selection. Null = no dive, an unknown project, or a
 * project registered without a bespoke world (palette-only fallback).
 */
export function getWorldDefinition(
  projectId: string | null,
): WorldDefinition | null {
  if (!projectId) return null;
  return WORLD_REGISTRY[projectId] ?? null;
}
