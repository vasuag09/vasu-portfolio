import React, { useRef, useState } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { Float, MeshDistortMaterial } from "@react-three/drei";
import * as THREE from "three";

/**
 * Interactive 3D floating geometry.
 * Each project gets a unique shape that rotates, distorts, and responds to hover.
 * Shapes: icosahedron (S-tier), torus (A-tier), octahedron (B-tier)
 */

const SHAPES = {
  S: "icosahedron",
  A: "torus",
  B: "octahedron",
};

const TIER_COLORS = {
  S: "#00f0ff",
  A: "#a855f7",
  B: "#3b82f6",
};

function Shape({ type = "icosahedron", color = "#00f0ff", hovered }) {
  const meshRef = useRef();
  const [targetSpeed, setTargetSpeed] = useState(0.3);

  useFrame((_, delta) => {
    if (!meshRef.current) return;
    const speed = hovered ? 1.2 : 0.3;
    const current = targetSpeed;
    const newSpeed = current + (speed - current) * delta * 3;
    setTargetSpeed(newSpeed);

    meshRef.current.rotation.x += delta * newSpeed;
    meshRef.current.rotation.y += delta * newSpeed * 0.7;
  });

  const getGeometry = () => {
    switch (type) {
      case "torus":
        return <torusGeometry args={[0.7, 0.25, 16, 32]} />;
      case "octahedron":
        return <octahedronGeometry args={[0.8, 0]} />;
      case "icosahedron":
      default:
        return <icosahedronGeometry args={[0.8, 1]} />;
    }
  };

  return (
    <Float speed={2} rotationIntensity={0.5} floatIntensity={0.8}>
      <mesh ref={meshRef} scale={hovered ? 1.15 : 1}>
        {getGeometry()}
        <MeshDistortMaterial
          color={color}
          emissive={color}
          emissiveIntensity={hovered ? 0.8 : 0.3}
          roughness={0.3}
          metalness={0.8}
          distort={hovered ? 0.4 : 0.2}
          speed={2}
          transparent
          opacity={0.85}
          toneMapped={false}
        />
      </mesh>
    </Float>
  );
}

export default function FloatingGeometry({
  tier = "B",
  size = 80,
  className = "",
}) {
  const [hovered, setHovered] = useState(false);
  const type = SHAPES[tier] || SHAPES.B;
  const color = TIER_COLORS[tier] || TIER_COLORS.B;

  return (
    <div
      className={`${className}`}
      style={{ width: size, height: size }}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      <Canvas
        camera={{ position: [0, 0, 3], fov: 45 }}
        dpr={[1, 2]}
        gl={{ antialias: true, alpha: true }}
        style={{ background: "transparent" }}
      >
        <ambientLight intensity={0.3} />
        <pointLight position={[3, 3, 3]} intensity={0.5} color={color} />
        <pointLight position={[-2, -2, 2]} intensity={0.3} color="#a855f7" />
        <Shape type={type} color={color} hovered={hovered} />
      </Canvas>
    </div>
  );
}
