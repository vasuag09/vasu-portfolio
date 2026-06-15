"use client";

import { useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { createNodeMaterial } from "./materials/node-material";
import { getGraphState } from "@/lib/graph-store";
import { SCENE_COLORS } from "@/lib/scene-colors";

/**
 * The Synapse node (Phase 6): one distinct breathing node floating beside
 * the core, visible from the hero and contact rest-poses. Per ADR-4 the
 * canvas is pointer-inert — DOM "boot synapse" triggers open the terminal;
 * this node answers visually (breathing at rest, HDR-bright while open).
 */

const POSITION: [number, number, number] = [4.4, 1.8, 4];
const SCALE = 0.5;

export function SynapseNode() {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const activeRef = useRef(0);

  const { geometry, material } = useMemo(() => {
    const geometry = new THREE.SphereGeometry(1, 48, 48);
    const color = new THREE.Color(SCENE_COLORS.accentBright);
    geometry.setAttribute(
      "aColor",
      new THREE.InstancedBufferAttribute(
        new Float32Array([color.r, color.g, color.b]),
        3,
      ),
    );
    geometry.setAttribute(
      "aIntensity",
      new THREE.InstancedBufferAttribute(new Float32Array([1.5]), 1),
    );
    geometry.setAttribute(
      "aActive",
      new THREE.InstancedBufferAttribute(new Float32Array([0]), 1),
    );
    return { geometry, material: createNodeMaterial() };
  }, []);

  useFrame(({ clock }, delta) => {
    const mesh = meshRef.current;
    if (!mesh) return;
    const t = clock.getElapsedTime();
    (mesh.material as THREE.ShaderMaterial).uniforms.uTime.value = t;

    // Slow breathing at rest; terminal-open ramps activation to full glow.
    const target = getGraphState().synapseOpen ? 1 : 0.25 + Math.sin(t * 0.9) * 0.12;
    activeRef.current += (target - activeRef.current) * (1 - Math.exp(-8 * delta));
    const attr = geometry.getAttribute("aActive") as THREE.InstancedBufferAttribute;
    (attr.array as Float32Array)[0] = activeRef.current;
    attr.needsUpdate = true;

    const matrix = new THREE.Matrix4()
      .makeScale(SCALE, SCALE, SCALE)
      .setPosition(...POSITION);
    mesh.setMatrixAt(0, matrix);
    mesh.instanceMatrix.needsUpdate = true;
  });

  return (
    <instancedMesh
      ref={meshRef}
      args={[geometry, material, 1]}
      frustumCulled={false}
    />
  );
}
