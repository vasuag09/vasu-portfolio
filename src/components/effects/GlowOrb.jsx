import React from "react";

/**
 * Animated glow orb accent element.
 * Positioned absolutely in a parent container for ambiance.
 */
export default function GlowOrb({ className = "" }) {
  return (
    <div
      className={`absolute rounded-full blur-3xl opacity-20 animate-glow-shift pointer-events-none ${className}`}
      aria-hidden="true"
      style={{
        background:
          "radial-gradient(circle, rgba(16,185,129,0.4) 0%, transparent 70%)",
      }}
    />
  );
}
