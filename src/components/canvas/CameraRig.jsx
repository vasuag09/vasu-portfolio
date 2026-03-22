import { useRef, useEffect } from "react";
import { useFrame } from "@react-three/fiber";

/**
 * Mouse-responsive camera rig.
 * Subtly orbits the scene based on cursor position.
 * Adds gentle idle drift when mouse is inactive.
 */
export default function CameraRig() {
  const mouseRef = useRef({ x: 0, y: 0 });
  const targetRef = useRef({ x: 0, y: 0 });
  const idleTimeRef = useRef(0);

  useEffect(() => {
    const handler = (e) => {
      mouseRef.current.x = (e.clientX / window.innerWidth - 0.5) * 2;
      mouseRef.current.y = (e.clientY / window.innerHeight - 0.5) * 2;
      idleTimeRef.current = 0;
    };

    window.addEventListener("mousemove", handler);
    return () => window.removeEventListener("mousemove", handler);
  }, []);

  useFrame((state, delta) => {
    idleTimeRef.current += delta;

    // When idle, add subtle auto-drift
    const idleDrift = Math.min(idleTimeRef.current / 5, 1);
    const time = performance.now() * 0.0003;
    const driftX = Math.sin(time) * 0.3 * idleDrift;
    const driftY = Math.cos(time * 0.7) * 0.15 * idleDrift;

    // Blend mouse influence with idle drift
    const mouseInfluence = 1 - idleDrift;
    targetRef.current.x = mouseRef.current.x * 1.5 * mouseInfluence + driftX;
    targetRef.current.y = -mouseRef.current.y * 0.8 * mouseInfluence + driftY;

    // Smooth lerp to target
    const cam = state.camera;
    cam.position.x += (targetRef.current.x - cam.position.x) * delta * 1.5;
    cam.position.y += (targetRef.current.y - cam.position.y) * delta * 1.5;

    cam.lookAt(0, 0, 0);
  });

  return null;
}
