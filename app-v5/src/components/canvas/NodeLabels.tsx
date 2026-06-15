"use client";

import { useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import { Html } from "@react-three/drei";
import * as THREE from "three";
import { pickLabelNodes, type SceneLabel } from "@/lib/scene-elevation";

/**
 * ADR-8: graph node labels as REAL DOM, projected by drei's Html (CSS
 * transform — no troika, no canvas text, zero new deps). Opacity fades by
 * camera distance so labels exist only when their cluster is the subject.
 * Decorative duplicates of the DOM lists → the layer inherits aria-hidden
 * and pointer-events:none from .canvas-container. Desktop tier only.
 */

const FADE_FAR = 17; // fully hidden beyond this camera distance
const FADE_NEAR = 10; // fully visible inside this

function LabelItem({ label }: { label: SceneLabel }) {
  const divRef = useRef<HTMLDivElement>(null);
  const world = useMemo(() => new THREE.Vector3(...label.position), [label]);

  useFrame(({ camera }) => {
    const el = divRef.current;
    if (!el) return;
    const distance = camera.position.distanceTo(world);
    const opacity = THREE.MathUtils.clamp(
      (FADE_FAR - distance) / (FADE_FAR - FADE_NEAR),
      0,
      1,
    );
    el.style.opacity = opacity.toFixed(3);
  });

  return (
    <Html
      position={label.position}
      center
      zIndexRange={[0, 0]}
      wrapperClass="graph-label-wrap"
    >
      <div ref={divRef} className={`graph-label graph-label--${label.kind}`}>
        {label.text}
      </div>
    </Html>
  );
}

export function NodeLabels() {
  const labels = useMemo(() => pickLabelNodes(), []);
  return (
    <group>
      {labels.map((label) => (
        <LabelItem key={label.id} label={label} />
      ))}
    </group>
  );
}
