import React, { Suspense, useRef, useMemo, useState, useCallback } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { Text, Float } from "@react-three/drei";
import * as THREE from "three";

/**
 * 3D Interactive Skill Orb — a rotating sphere of skill text nodes.
 * Users can drag to rotate. Skills are distributed on a sphere surface
 * using Fibonacci spiral for even spacing.
 */

function fibonacciSphere(count) {
  const points = [];
  const goldenRatio = (1 + Math.sqrt(5)) / 2;
  for (let i = 0; i < count; i++) {
    const theta = Math.acos(1 - (2 * (i + 0.5)) / count);
    const phi = (2 * Math.PI * i) / goldenRatio;
    points.push({
      x: Math.sin(theta) * Math.cos(phi),
      y: Math.sin(theta) * Math.sin(phi),
      z: Math.cos(theta),
    });
  }
  return points;
}

function SkillNode({ position, label, index, total }) {
  const ref = useRef();
  const [hovered, setHovered] = useState(false);
  const t = index / total;

  // Color gradient: cyan → violet across sphere
  const color = useMemo(() => {
    const c = new THREE.Color();
    c.setHSL(0.52 + t * 0.15, 0.9, hovered ? 0.75 : 0.55);
    return c;
  }, [t, hovered]);

  useFrame(({ camera }) => {
    if (ref.current) {
      ref.current.quaternion.copy(camera.quaternion);
    }
  });

  return (
    <group position={[position.x * 3.2, position.y * 3.2, position.z * 3.2]}>
      {/* Glow sphere behind text */}
      <mesh>
        <sphereGeometry args={[hovered ? 0.15 : 0.06, 8, 8]} />
        <meshBasicMaterial
          color={color}
          transparent
          opacity={hovered ? 0.6 : 0.3}
          depthWrite={false}
          blending={THREE.AdditiveBlending}
        />
      </mesh>

      {/* Skill label — uses drei's built-in default font */}
      <Text
        ref={ref}
        fontSize={hovered ? 0.22 : 0.16}
        color={hovered ? "#ffffff" : color}
        anchorX="center"
        anchorY="middle"
        onPointerOver={() => setHovered(true)}
        onPointerOut={() => setHovered(false)}
      >
        {label}
      </Text>
    </group>
  );
}

function OrbScene({ skills, isDragging }) {
  const groupRef = useRef();
  const velocityRef = useRef({ x: 0.002, y: 0.003 });
  const points = useMemo(() => fibonacciSphere(skills.length), [skills.length]);

  useFrame(() => {
    if (!groupRef.current) return;

    // Auto-rotate when not dragging
    if (!isDragging) {
      velocityRef.current.x *= 0.99; // Gentle deceleration
      velocityRef.current.y *= 0.99;
      velocityRef.current.x = Math.max(velocityRef.current.x, 0.001);
      velocityRef.current.y = Math.max(velocityRef.current.y, 0.0015);
    }

    groupRef.current.rotation.y += velocityRef.current.x;
    groupRef.current.rotation.x += velocityRef.current.y * 0.3;
  });

  return (
    <group ref={groupRef}>
      {/* Central glowing core */}
      <Float speed={1.5} floatIntensity={0.3}>
        <mesh>
          <icosahedronGeometry args={[0.3, 2]} />
          <meshBasicMaterial
            color="#00f0ff"
            transparent
            opacity={0.1}
            wireframe
            depthWrite={false}
          />
        </mesh>
      </Float>

      {/* Connection lines from core to nodes */}
      {points.map((p, i) => {
        const positions = new Float32Array([0, 0, 0, p.x * 3.2, p.y * 3.2, p.z * 3.2]);
        return (
          <line key={`line-${i}`}>
            <bufferGeometry>
              <bufferAttribute
                attach="attributes-position"
                count={2}
                array={positions}
                itemSize={3}
              />
            </bufferGeometry>
            <lineBasicMaterial
              color="#00f0ff"
              transparent
              opacity={0.04}
              depthWrite={false}
              blending={THREE.AdditiveBlending}
            />
          </line>
        );
      })}

      {/* Skill nodes */}
      {skills.map((skill, i) => (
        <SkillNode
          key={skill}
          position={points[i]}
          label={skill}
          index={i}
          total={skills.length}
        />
      ))}
    </group>
  );
}

export default function SkillOrb({ skills = [], className = "" }) {
  const [isDragging, setIsDragging] = useState(false);
  const lastMouse = useRef({ x: 0, y: 0 });

  const handlePointerDown = useCallback((e) => {
    setIsDragging(true);
    lastMouse.current = { x: e.clientX, y: e.clientY };
  }, []);

  const handlePointerUp = useCallback(() => {
    setIsDragging(false);
  }, []);

  return (
    <div
      className={`${className} cursor-grab active:cursor-grabbing`}
      onPointerDown={handlePointerDown}
      onPointerUp={handlePointerUp}
      onPointerLeave={handlePointerUp}
    >
      <Canvas
        camera={{ position: [0, 0, 8], fov: 50 }}
        dpr={[1, 2]}
        gl={{ antialias: true, alpha: true }}
        style={{ background: "transparent" }}
      >
        <ambientLight intensity={0.5} />
        <Suspense fallback={null}>
          <OrbScene skills={skills} isDragging={isDragging} />
        </Suspense>
      </Canvas>
    </div>
  );
}
