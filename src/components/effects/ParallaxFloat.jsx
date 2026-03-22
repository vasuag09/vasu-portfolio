import React, { useRef, useState, useEffect } from "react";

/**
 * Wrapper that adds mouse-responsive parallax movement.
 * `depth` controls intensity: 0.01 (subtle) to 0.05 (strong).
 */
export default function ParallaxFloat({ children, depth = 0.02, className = "" }) {
  const ref = useRef(null);
  const [offset, setOffset] = useState({ x: 0, y: 0 });

  useEffect(() => {
    const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (prefersReducedMotion) return;

    const handleMouseMove = (e) => {
      const centerX = window.innerWidth / 2;
      const centerY = window.innerHeight / 2;
      setOffset({
        x: (e.clientX - centerX) * depth,
        y: (e.clientY - centerY) * depth,
      });
    };

    window.addEventListener("mousemove", handleMouseMove);
    return () => window.removeEventListener("mousemove", handleMouseMove);
  }, [depth]);

  return (
    <div
      ref={ref}
      className={className}
      style={{
        transform: `translate(${offset.x}px, ${offset.y}px)`,
        transition: "transform 0.3s ease-out",
      }}
    >
      {children}
    </div>
  );
}
