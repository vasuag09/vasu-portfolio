"use client";

import { useLayoutEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import { Line } from "@react-three/drei";
import { sections } from "@/data/sections-v5";
import { projectNodes } from "@/data/projects-v5";
import { skills } from "@/data/skills-graph";
import { buildCameraSpline, sampleCameraPose } from "@/lib/camera-spline";

/**
 * Phase-1 placeholder world: enough geometry to SEE the camera flight.
 * Replaced by the ported v4 network + GPU particles in Phase 2.
 *
 * Colors are hex approximations of the OKLCH tokens (three.js cannot parse
 * oklch() strings) — single source of truth remains tokens.css.
 */
const COLOR = {
  accent: "#27b173", // ≈ oklch(58% 0.18 150)
  accentBright: "#4fe3a1", // ≈ oklch(72% 0.20 150)
  node: "#c3cce0", // ≈ oklch(85% 0.03 250)
  ml: "#5c8fc2", // ≈ oklch(62% 0.10 230)
  faint: "#2a3040", // ≈ border tone
} as const;

function InstancedNodes({
  positions,
  scales,
  color,
  radius,
}: {
  positions: readonly (readonly [number, number, number])[];
  scales?: readonly number[];
  color: string;
  radius: number;
}) {
  const meshRef = useRef<THREE.InstancedMesh>(null);

  useLayoutEffect(() => {
    const mesh = meshRef.current;
    if (!mesh) return;
    const matrix = new THREE.Matrix4();
    positions.forEach(([x, y, z], i) => {
      const s = scales?.[i] ?? 1;
      matrix.makeScale(s, s, s).setPosition(x, y, z);
      mesh.setMatrixAt(i, matrix);
    });
    mesh.instanceMatrix.needsUpdate = true;
  }, [positions, scales]);

  return (
    <instancedMesh ref={meshRef} args={[undefined, undefined, positions.length]}>
      <icosahedronGeometry args={[radius, 1]} />
      <meshBasicMaterial color={color} wireframe />
    </instancedMesh>
  );
}

function DebugSpline() {
  const points = useMemo(() => {
    const spline = buildCameraSpline(sections);
    const samples: THREE.Vector3[] = [];
    for (let i = 0; i <= 120; i += 1) {
      samples.push(sampleCameraPose(spline, i / 120).position.clone());
    }
    return samples;
  }, []);

  return <Line points={points} color={COLOR.accentBright} lineWidth={1} />;
}

export function PlaceholderScene({ debug = false }: { debug?: boolean }) {
  const anchorTargets = useMemo(
    () => sections.map((s) => s.cameraTarget),
    [],
  );
  const projectPositions = useMemo(
    () => projectNodes.map((n) => n.position),
    [],
  );
  const projectScales = useMemo(() => projectNodes.map((n) => n.scale), []);
  const skillPositions = useMemo(() => skills.map((s) => s.position), []);

  return (
    <>
      {/* Section beacons — one landmark per region so flight is legible */}
      <InstancedNodes
        positions={anchorTargets}
        color={COLOR.accent}
        radius={1.2}
      />
      {/* Project cluster */}
      <InstancedNodes
        positions={projectPositions}
        scales={projectScales}
        color={COLOR.node}
        radius={0.35}
      />
      {/* Skills constellation */}
      <InstancedNodes
        positions={skillPositions}
        color={COLOR.ml}
        radius={0.25}
      />
      {debug ? <DebugSpline /> : null}
    </>
  );
}
