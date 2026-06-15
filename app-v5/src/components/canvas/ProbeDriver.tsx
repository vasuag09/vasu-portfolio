"use client";

import { useEffect, useRef } from "react";
import { useFrame, useThree } from "@react-three/fiber";
import * as THREE from "three";

import { dampTowards } from "@/lib/camera-dive";
import { probeStrengthTarget } from "@/lib/probe-state";
import { signalUniforms } from "@/lib/signal-uniforms";

/**
 * Cursor probe driver (ADR-9 wave 2). The canvas has pointerEvents:"none"
 * (so the DOM custom cursor keeps its z-index), so we read the pointer from
 * a window listener, unproject it through the live camera, and write the
 * world position + strength into the shared signalUniforms — the particle
 * shader brightens the field near it. Same gate as the custom cursor: no
 * probe on touch / reduced motion (probeStrengthTarget → 0, no listener).
 *
 * Frame-loop state lives on a ref (the React Compiler escape hatch — same as
 * Particles / CameraRig); useMemo values are immutable to the compiler.
 */

/** Distance along the pointer ray where the probe point sits (near the field). */
const PROBE_DEPTH = 16;
/** Strength damping rate (~settles in ~400ms). */
const PROBE_RATE = 6;

export function ProbeDriver() {
  const camera = useThree((state) => state.camera);
  const stateRef = useRef({
    ndc: new THREE.Vector2(),
    raycaster: new THREE.Raycaster(),
    world: new THREE.Vector3(),
    seen: false,
    enabled: false,
  });

  useEffect(() => {
    const s = stateRef.current;
    s.enabled =
      probeStrengthTarget({
        finePointer: matchMedia("(pointer: fine)").matches,
        canHover: matchMedia("(hover: hover)").matches,
        reducedMotion: matchMedia("(prefers-reduced-motion: reduce)").matches,
      }) > 0;
    if (!s.enabled) return;

    const onMove = (e: PointerEvent) => {
      s.ndc.set(
        (e.clientX / window.innerWidth) * 2 - 1,
        -(e.clientY / window.innerHeight) * 2 + 1,
      );
      s.seen = true;
    };
    const onLeave = () => {
      s.seen = false;
    };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerout", onLeave);
    return () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerout", onLeave);
      signalUniforms.uProbeStrength.value = 0;
    };
  }, []);

  useFrame((_, delta) => {
    const s = stateRef.current;
    const active = s.enabled && s.seen;
    if (active) {
      s.raycaster.setFromCamera(s.ndc, camera);
      s.raycaster.ray.at(PROBE_DEPTH, s.world);
      signalUniforms.uPointerPos.value.copy(s.world);
    }
    signalUniforms.uProbeStrength.value = dampTowards(
      signalUniforms.uProbeStrength.value,
      active ? 1 : 0,
      delta,
      PROBE_RATE,
    );
  });

  return null;
}
