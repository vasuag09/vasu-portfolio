"use client";

import { useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";

import { projectNodes } from "@/data/projects-v5";
import { PROJECT_WORLD_COLORS } from "@/lib/scene-colors";
import { signalUniforms } from "@/lib/signal-uniforms";

/**
 * NM-GPT world motif (ADR-10): a slow carousel of "document cards" around the
 * project node — RAG over institutional documents, sources orbiting the
 * answer. Lazy-loaded under ProjectWorldEnv's Suspense boundary; opacity rides
 * uWorldBlend so it fades with the dive cross-fade (invisible when inactive).
 */

const CARD_COUNT = 6;
const RING_RADIUS = 1.4;
const MAX_OPACITY = 0.6;

export function NmGptWorld() {
  const node = useMemo(() => projectNodes.find((n) => n.id === "nm-gpt"), []);
  const groupRef = useRef<THREE.Group>(null);
  // Cards share one material; opacity is mutated each frame via this mesh ref
  // (Particles idiom — mutate through a ref, not the useMemo value).
  const meshRef = useRef<THREE.Mesh>(null);

  const { cards, material, geometry } = useMemo(() => {
    const cards: { pos: [number, number, number]; rot: [number, number, number] }[] = [];
    for (let i = 0; i < CARD_COUNT; i += 1) {
      const a = (i / CARD_COUNT) * Math.PI * 2;
      cards.push({
        pos: [Math.cos(a) * RING_RADIUS, (i - CARD_COUNT / 2) * 0.26, Math.sin(a) * RING_RADIUS],
        // Face outward from the ring centre.
        rot: [0, -a + Math.PI / 2, 0],
      });
    }
    const material = new THREE.MeshBasicMaterial({
      color: new THREE.Color(PROJECT_WORLD_COLORS["nm-gpt"].primary),
      transparent: true,
      opacity: 0,
      side: THREE.DoubleSide,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    const geometry = new THREE.PlaneGeometry(0.55, 0.75);
    return { cards, material, geometry };
  }, []);

  useFrame((_, delta) => {
    const group = groupRef.current;
    const mesh = meshRef.current;
    if (!group || !mesh) return;
    group.rotation.y += delta * 0.18;
    (mesh.material as THREE.MeshBasicMaterial).opacity =
      signalUniforms.uWorldBlend.value * MAX_OPACITY;
  });

  if (!node) return null;

  return (
    <group ref={groupRef} position={node.position}>
      {cards.map((c, i) => (
        <mesh
          key={i}
          ref={i === 0 ? meshRef : undefined}
          position={c.pos}
          rotation={c.rot}
          geometry={geometry}
          material={material}
        />
      ))}
    </group>
  );
}
