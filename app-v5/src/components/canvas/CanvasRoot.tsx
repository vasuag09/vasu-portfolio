"use client";

import dynamic from "next/dynamic";
import { useReducedMotion } from "@/hooks/useReducedMotion";
import { CanvasErrorBoundary } from "./CanvasErrorBoundary";

/**
 * ADR-4 mount: fullscreen fixed, behind the DOM (z -1), inert to pointers and
 * screen readers. WebGL is client-only — ssr:false keeps the server render
 * clean, and reduced-motion skips mounting the canvas entirely (the CSS
 * backstop in reduced-motion.css also hides .canvas-container).
 */
const SceneCanvas = dynamic(() => import("./SceneCanvas"), { ssr: false });

export function CanvasRoot() {
  const reducedMotion = useReducedMotion();
  if (reducedMotion) return null;

  return (
    <div
      className="canvas-container fixed inset-0"
      style={{ zIndex: "var(--z-canvas)", pointerEvents: "none" }}
      aria-hidden="true"
    >
      <CanvasErrorBoundary>
        <SceneCanvas />
      </CanvasErrorBoundary>
    </div>
  );
}
