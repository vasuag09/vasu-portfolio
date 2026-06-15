"use client";

import { useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";

import { projectNodes } from "@/data/projects-v5";
import { PROJECT_WORLD_COLORS } from "@/lib/scene-colors";
import { signalUniforms } from "@/lib/signal-uniforms";

/**
 * FundlyMart world motif (ADR-10): a slow ring of "order packets" orbiting
 * the project node — WhatsApp-native B2B commerce, orders in flight. Lazy-
 * loaded under ProjectWorldEnv's Suspense boundary. Opacity rides uWorldBlend
 * so it fades in/out with the dive cross-fade — no separate lifecycle, and it
 * is invisible (opacity 0) whenever no FundlyMart dive is active.
 */

const PACKET_COUNT = 9;
const RING_RADIUS = 1.7;
const MAX_OPACITY = 0.85;

export function FundlymartWorld() {
  const node = useMemo(() => projectNodes.find((n) => n.id === "fundlymart"), []);
  const groupRef = useRef<THREE.Group>(null);
  // The packets share one material; we mutate its opacity each frame via this
  // mesh ref (the codebase's Particles idiom — mutate through a ref, never a
  // useMemo value directly, which the React Compiler treats as immutable).
  const meshRef = useRef<THREE.Mesh>(null);

  const { positions, material, geometry } = useMemo(() => {
    const positions: [number, number, number][] = [];
    for (let i = 0; i < PACKET_COUNT; i += 1) {
      const a = (i / PACKET_COUNT) * Math.PI * 2;
      positions.push([
        Math.cos(a) * RING_RADIUS,
        Math.sin(a * 2) * 0.35,
        Math.sin(a) * RING_RADIUS,
      ]);
    }
    const material = new THREE.MeshBasicMaterial({
      color: new THREE.Color(PROJECT_WORLD_COLORS.fundlymart.accent),
      transparent: true,
      opacity: 0,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    const geometry = new THREE.BoxGeometry(0.22, 0.22, 0.22);
    return { positions, material, geometry };
  }, []);

  useFrame((_, delta) => {
    const group = groupRef.current;
    const mesh = meshRef.current;
    if (!group || !mesh) return;
    group.rotation.y += delta * 0.3;
    // Opacity tracks the cross-fade — 0 when no dive is active (invisible).
    (mesh.material as THREE.MeshBasicMaterial).opacity =
      signalUniforms.uWorldBlend.value * MAX_OPACITY;
  });

  if (!node) return null;

  return (
    <group ref={groupRef} position={node.position}>
      {positions.map((p, i) => (
        <mesh
          key={i}
          ref={i === 0 ? meshRef : undefined}
          position={p}
          geometry={geometry}
          material={material}
        />
      ))}
    </group>
  );
}
