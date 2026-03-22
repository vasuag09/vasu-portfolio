import React, { useRef, useMemo, useEffect } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";

/**
 * GPU-instanced neural network nodes rendered as glowing spheres.
 * 5 layers with topology [4,6,8,6,4] desktop / [3,4,5,4,3] mobile.
 * Active layer nodes pulse brighter with volumetric glow.
 */

const COLORS = {
  cyan: new THREE.Color(0x00f0ff),
  violet: new THREE.Color(0xa855f7),
};

function generateNetwork(isMobile) {
  const layers = isMobile ? [3, 4, 5, 4, 3] : [4, 6, 8, 6, 4];
  const nodes = [];
  const spreadX = isMobile ? 8 : 14;
  const spreadY = isMobile ? 5 : 7;

  layers.forEach((count, layerIdx) => {
    const x = (layerIdx / (layers.length - 1) - 0.5) * spreadX;
    const t = layerIdx / (layers.length - 1);
    const color = new THREE.Color().copy(COLORS.cyan).lerp(COLORS.violet, t);

    for (let i = 0; i < count; i++) {
      const y = (i / (count - 1 || 1) - 0.5) * spreadY;
      nodes.push({
        position: new THREE.Vector3(x, y, (Math.random() - 0.5) * 1.5),
        layer: layerIdx,
        color,
        phase: Math.random() * Math.PI * 2,
        baseScale: isMobile ? 0.08 : 0.12,
      });
    }
  });

  return { nodes, layers };
}

function generateConnections(nodes) {
  const connections = [];
  for (let i = 0; i < nodes.length; i++) {
    for (let j = i + 1; j < nodes.length; j++) {
      if (
        nodes[j].layer === nodes[i].layer + 1 &&
        Math.random() > 0.35
      ) {
        connections.push({
          from: i,
          to: j,
          opacity: 0.04 + Math.random() * 0.06,
        });
      }
    }
  }
  return connections;
}

export default function NetworkNodes({ activeLayer = 0, isMobile = false }) {
  const meshRef = useRef();
  const glowMeshRef = useRef();

  const { nodes, connections } = useMemo(() => {
    const net = generateNetwork(isMobile);
    const conns = generateConnections(net.nodes);
    return { nodes: net.nodes, connections: conns };
  }, [isMobile]);

  const dummy = useMemo(() => new THREE.Object3D(), []);
  const colorArray = useMemo(() => new Float32Array(nodes.length * 3), [nodes]);

  // Set initial positions and colors
  useEffect(() => {
    if (!meshRef.current) return;
    nodes.forEach((node, i) => {
      dummy.position.copy(node.position);
      dummy.scale.setScalar(node.baseScale);
      dummy.updateMatrix();
      meshRef.current.setMatrixAt(i, dummy.matrix);
      colorArray[i * 3] = node.color.r;
      colorArray[i * 3 + 1] = node.color.g;
      colorArray[i * 3 + 2] = node.color.b;
    });
    meshRef.current.instanceMatrix.needsUpdate = true;
    meshRef.current.geometry.setAttribute(
      "color",
      new THREE.InstancedBufferAttribute(colorArray, 3)
    );
  }, [nodes, dummy, colorArray]);

  // Animate nodes
  useFrame(({ clock }) => {
    if (!meshRef.current) return;
    const t = clock.getElapsedTime();

    nodes.forEach((node, i) => {
      const isActive = node.layer === activeLayer;
      const pulse = Math.sin(t * 2 + node.phase) * 0.3 + 0.7;
      const scale = node.baseScale * (isActive ? 1.6 + pulse * 0.4 : 0.8 + pulse * 0.1);

      // Subtle float
      const floatY = Math.sin(t * 0.5 + node.phase) * 0.05;

      dummy.position.set(
        node.position.x,
        node.position.y + floatY,
        node.position.z,
      );
      dummy.scale.setScalar(scale);
      dummy.updateMatrix();
      meshRef.current.setMatrixAt(i, dummy.matrix);
    });

    meshRef.current.instanceMatrix.needsUpdate = true;

    // Update glow meshes
    if (glowMeshRef.current) {
      nodes.forEach((node, i) => {
        const isActive = node.layer === activeLayer;
        const pulse = Math.sin(t * 2 + node.phase) * 0.3 + 0.7;
        const glowScale = isActive ? node.baseScale * (3 + pulse * 1.5) : node.baseScale * 1.5;
        const floatY = Math.sin(t * 0.5 + node.phase) * 0.05;

        dummy.position.set(node.position.x, node.position.y + floatY, node.position.z);
        dummy.scale.setScalar(glowScale);
        dummy.updateMatrix();
        glowMeshRef.current.setMatrixAt(i, dummy.matrix);
      });
      glowMeshRef.current.instanceMatrix.needsUpdate = true;
    }
  });

  return (
    <group>
      {/* Core nodes */}
      <instancedMesh ref={meshRef} args={[null, null, nodes.length]}>
        <sphereGeometry args={[1, 16, 16]} />
        <meshStandardMaterial
          color="#00f0ff"
          emissive="#00f0ff"
          emissiveIntensity={1.5}
          toneMapped={false}
        />
      </instancedMesh>

      {/* Glow halos (larger, transparent spheres picked up by bloom) */}
      <instancedMesh ref={glowMeshRef} args={[null, null, nodes.length]}>
        <sphereGeometry args={[1, 12, 12]} />
        <meshBasicMaterial
          color="#00f0ff"
          transparent
          opacity={0.08}
          depthWrite={false}
          blending={THREE.AdditiveBlending}
        />
      </instancedMesh>

      {/* Connections */}
      {connections.map((conn, idx) => {
        const from = nodes[conn.from];
        const to = nodes[conn.to];
        const isActive = from.layer === activeLayer || to.layer === activeLayer;
        const t = from.layer / 4;
        const color = new THREE.Color().copy(COLORS.cyan).lerp(COLORS.violet, t);

        const points = [from.position, to.position];

        return (
          <line key={idx}>
            <bufferGeometry>
              <bufferAttribute
                attach="attributes-position"
                count={2}
                array={new Float32Array([
                  points[0].x, points[0].y, points[0].z,
                  points[1].x, points[1].y, points[1].z,
                ])}
                itemSize={3}
              />
            </bufferGeometry>
            <lineBasicMaterial
              color={color}
              transparent
              opacity={isActive ? conn.opacity * 3 : conn.opacity}
              depthWrite={false}
              blending={THREE.AdditiveBlending}
            />
          </line>
        );
      })}
    </group>
  );
}

// Network generation functions are kept internal — particles get data via props.
