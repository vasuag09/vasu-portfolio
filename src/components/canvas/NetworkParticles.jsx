import React, { useRef, useEffect } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";

/**
 * GPU-accelerated particle system.
 * Particles flow along neural connections as glowing data points.
 */

const COLORS = {
  cyan: new THREE.Color(0x00f0ff),
  violet: new THREE.Color(0xa855f7),
};

export default function NetworkParticles({
  nodes,
  connections,
  activeLayer = 0,
  count = 150,
}) {
  const pointsRef = useRef();

  // Initialize particle data in effect to keep render pure
  const particlesRef = useRef([]);
  useEffect(() => {
    if (particlesRef.current.length > 0 || connections.length === 0) return;
    particlesRef.current = Array.from({ length: count }, () => {
      const connIdx = Math.floor(Math.random() * connections.length);
      return {
        connIdx,
        progress: Math.random(),
        speed: 0.003 + Math.random() * 0.006,
        size: 1.5 + Math.random() * 2,
      };
    });
  }, [connections, count]);
  const particles = particlesRef.current;

  // Pre-allocate geometry buffers
  const buffersRef = useRef(null);
  if (!buffersRef.current) {
    buffersRef.current = {
      positions: new Float32Array(count * 3),
      colors: new Float32Array(count * 3),
      sizes: new Float32Array(count),
    };
  }
  const { positions, colors, sizes } = buffersRef.current;

  useFrame(() => {
    if (!pointsRef.current || !nodes.length || !connections.length) return;

    particles.forEach((p, i) => {
      // Advance along connection
      p.progress += p.speed;
      if (p.progress > 1) {
        p.progress = 0;
        p.connIdx = Math.floor(Math.random() * connections.length);
      }

      const conn = connections[p.connIdx];
      if (!conn) return;
      const from = nodes[conn.from];
      const to = nodes[conn.to];
      if (!from || !to) return;

      // Interpolate position
      const x = from.position.x + (to.position.x - from.position.x) * p.progress;
      const y = from.position.y + (to.position.y - from.position.y) * p.progress;
      const z = from.position.z + (to.position.z - from.position.z) * p.progress;

      positions[i * 3] = x;
      positions[i * 3 + 1] = y;
      positions[i * 3 + 2] = z;

      // Color based on source layer
      const t = from.layer / 4;
      const color = new THREE.Color().copy(COLORS.cyan).lerp(COLORS.violet, t);
      const isActive = from.layer === activeLayer || to.layer === activeLayer;

      colors[i * 3] = color.r;
      colors[i * 3 + 1] = color.g;
      colors[i * 3 + 2] = color.b;

      // Active particles are larger
      sizes[i] = isActive ? p.size * 2 : p.size;
    });

    const geo = pointsRef.current.geometry;
    geo.attributes.position.needsUpdate = true;
    geo.attributes.color.needsUpdate = true;
    geo.attributes.size.needsUpdate = true;
  });

  if (!connections.length || !nodes.length) return null;

  return (
    <points ref={pointsRef}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          count={count}
          array={positions}
          itemSize={3}
        />
        <bufferAttribute
          attach="attributes-color"
          count={count}
          array={colors}
          itemSize={3}
        />
        <bufferAttribute
          attach="attributes-size"
          count={count}
          array={sizes}
          itemSize={1}
        />
      </bufferGeometry>
      <pointsMaterial
        size={0.06}
        vertexColors
        transparent
        opacity={0.8}
        depthWrite={false}
        blending={THREE.AdditiveBlending}
        sizeAttenuation
      />
    </points>
  );
}
