"use client";

import dynamic from "next/dynamic";
import { useEffect, useState } from "react";
import { useReducedMotion } from "@/hooks/useReducedMotion";
import { CanvasErrorBoundary } from "./CanvasErrorBoundary";

/**
 * ADR-4 mount: fullscreen fixed, behind the DOM (z -1), inert to pointers and
 * screen readers. WebGL is client-only — ssr:false keeps the server render
 * clean, and reduced-motion skips mounting the canvas entirely (the CSS
 * backstop in reduced-motion.css also hides .canvas-container).
 */
const SceneCanvas = dynamic(() => import("./SceneCanvas"), { ssr: false });

/**
 * ADR-2: the scene chunk loads AFTER first paint. ~250KB gz of three.js
 * evaluating during hydration pushes mobile LCP/TBT past budget (Phase-8
 * measurement), so the import is gated on an idle slot. The 2s timeout
 * bounds the wait on busy main threads; Safari lacks requestIdleCallback,
 * hence the timer fallback.
 */
function useIdleMount(): boolean {
  const [ready, setReady] = useState(false);
  useEffect(() => {
    if (typeof window.requestIdleCallback === "function") {
      const id = window.requestIdleCallback(() => setReady(true), {
        timeout: 2000,
      });
      return () => window.cancelIdleCallback(id);
    }
    const timer = window.setTimeout(() => setReady(true), 300);
    return () => window.clearTimeout(timer);
  }, []);
  return ready;
}

export function CanvasRoot() {
  const reducedMotion = useReducedMotion();
  const idle = useIdleMount();
  if (reducedMotion || !idle) return null;

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
