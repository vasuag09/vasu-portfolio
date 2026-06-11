"use client";

import { Component, type ReactNode } from "react";

/**
 * GPU-failure backstop (Phase 6): if WebGL context creation or the scene
 * throws, the canvas subtree unmounts and the site continues as the plain
 * DOM column — same shape as the reduced-motion tier (ADR-6). The content
 * never depends on the scene.
 */
export class CanvasErrorBoundary extends Component<
  { children: ReactNode },
  { failed: boolean }
> {
  state = { failed: false };

  static getDerivedStateFromError(): { failed: boolean } {
    return { failed: true };
  }

  componentDidCatch(error: unknown): void {
    console.error("Canvas failed — continuing without the 3D scene:", error);
  }

  render(): ReactNode {
    return this.state.failed ? null : this.props.children;
  }
}
