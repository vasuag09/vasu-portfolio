import React, { Suspense, useMemo, useState, useEffect } from "react";
import { Canvas } from "@react-three/fiber";
import { EffectComposer, Bloom, Vignette } from "@react-three/postprocessing";
import * as THREE from "three";
import NetworkNodes from "./NetworkNodes";
import NetworkParticles from "./NetworkParticles";
import CameraRig from "./CameraRig";

/**
 * 3D Neural Network scene — the immersive background.
 * Renders a full Three.js scene with instanced nodes, glowing connections,
 * flowing particles, and post-processing (bloom + chromatic aberration + vignette).
 */

// Generate network data at module level so particles can share it
function generateNetworkData(isMobile) {
  const layers = isMobile ? [3, 4, 5, 4, 3] : [4, 6, 8, 6, 4];
  const nodes = [];
  const spreadX = isMobile ? 8 : 14;
  const spreadY = isMobile ? 5 : 7;

  layers.forEach((count, layerIdx) => {
    const x = (layerIdx / (layers.length - 1) - 0.5) * spreadX;
    const t = layerIdx / (layers.length - 1);
    const color = new THREE.Color(0x00f0ff).lerp(new THREE.Color(0xa855f7), t);

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

  const connections = [];
  for (let i = 0; i < nodes.length; i++) {
    for (let j = i + 1; j < nodes.length; j++) {
      if (nodes[j].layer === nodes[i].layer + 1 && Math.random() > 0.35) {
        connections.push({ from: i, to: j, opacity: 0.04 + Math.random() * 0.06 });
      }
    }
  }

  return { nodes, connections };
}

function Scene({ activeLayer, isMobile, reducedMotion }) {
  const network = useMemo(() => generateNetworkData(isMobile), [isMobile]);
  const particleCount = isMobile ? 30 : 80;

  return (
    <>
      {/* Lighting */}
      <ambientLight intensity={0.15} />
      <pointLight position={[5, 5, 8]} intensity={0.3} color="#00f0ff" />
      <pointLight position={[-5, -3, 6]} intensity={0.2} color="#a855f7" />

      {/* Fog for depth */}
      <fog attach="fog" args={["#06080f", 12, 30]} />

      {/* Neural nodes + connections */}
      <NetworkNodes
        activeLayer={activeLayer}
        isMobile={isMobile}
      />

      {/* Flowing particles */}
      {!reducedMotion && (
        <NetworkParticles
          nodes={network.nodes}
          connections={network.connections}
          activeLayer={activeLayer}
          count={particleCount}
        />
      )}

      {/* Camera movement */}
      {!reducedMotion && <CameraRig />}

      {/* Post-processing — lightweight, no mipmapBlur */}
      {!reducedMotion && !isMobile && (
        <EffectComposer>
          <Bloom
            luminanceThreshold={0.6}
            luminanceSmoothing={0.9}
            intensity={0.5}
          />
          <Vignette darkness={0.4} offset={0.3} />
        </EffectComposer>
      )}

      {/* Mobile — no post-processing at all */}
    </>
  );
}

export default function NeuralNetwork3D({ activeLayer = 0 }) {
  const [isMobile, setIsMobile] = useState(
    () => typeof window !== "undefined" && window.innerWidth < 768
  );
  const reducedMotion = typeof window !== "undefined" &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const [ready, setReady] = useState(false);
  const [isVisible, setIsVisible] = useState(true);

  // Pause rendering when tab is hidden (perf fix #16)
  useEffect(() => {
    const handleVisibilityChange = () => {
      setIsVisible(!document.hidden);
    };
    document.addEventListener("visibilitychange", handleVisibilityChange);
    return () => document.removeEventListener("visibilitychange", handleVisibilityChange);
  }, []);

  useEffect(() => {
    const handleResize = () => setIsMobile(window.innerWidth < 768);
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  // Fade in after mount
  useEffect(() => {
    const timer = setTimeout(() => setReady(true), 100);
    return () => clearTimeout(timer);
  }, []);

  return (
    <div
      className="fixed inset-0 z-0 transition-opacity duration-1000"
      style={{ opacity: ready ? 0.5 : 0 }}
      aria-hidden="true"
    >
      <Canvas
        camera={{ position: [0, 0, isMobile ? 12 : 16], fov: 60 }}
        dpr={[1, 1]}
        frameloop={isVisible ? "always" : "never"}
        gl={{
          antialias: false,
          alpha: true,
          powerPreference: "default",
        }}
        style={{ background: "transparent" }}
        onCreated={({ gl }) => {
          gl.setClearColor(0x000000, 0);
        }}
      >
        <Suspense fallback={null}>
          <Scene
            activeLayer={activeLayer}
            isMobile={isMobile}
            reducedMotion={reducedMotion}
          />
        </Suspense>
      </Canvas>
    </div>
  );
}
