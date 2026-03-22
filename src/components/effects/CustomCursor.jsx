import React, { useEffect, useRef, useState } from "react";

/**
 * Custom cursor with glowing dot + trailing particle effect.
 * Replaces the default cursor with a neural-themed interactive indicator.
 * Hidden on touch devices. Shows a ring that scales on hovering interactive elements.
 */
const TRAIL_COUNT = 8;

export default function CustomCursor() {
  const dotRef = useRef(null);
  const ringRef = useRef(null);
  const trailRefs = useRef([]);
  const posRef = useRef({ x: -100, y: -100 });
  const trailPositions = useRef(
    Array.from({ length: TRAIL_COUNT }, () => ({ x: -100, y: -100 }))
  );
  const [isHovering, setIsHovering] = useState(false);
  const [visible, setVisible] = useState(false);
  const rafRef = useRef(null);

  useEffect(() => {
    // Don't show on touch devices
    if ("ontouchstart" in window) return;

    const handleMouseMove = (e) => {
      posRef.current = { x: e.clientX, y: e.clientY };
      if (!visible) setVisible(true);
    };

    const handleMouseOver = (e) => {
      const target = e.target;
      const isInteractive =
        target.closest("a, button, [role='button'], input, textarea, select, [data-cursor-hover]") ||
        target.tagName === "A" ||
        target.tagName === "BUTTON" ||
        window.getComputedStyle(target).cursor === "pointer";
      setIsHovering(!!isInteractive);
    };

    const handleMouseLeave = () => {
      setVisible(false);
    };

    const handleMouseEnter = () => {
      setVisible(true);
    };

    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("mouseover", handleMouseOver);
    document.documentElement.addEventListener("mouseleave", handleMouseLeave);
    document.documentElement.addEventListener("mouseenter", handleMouseEnter);

    // Animation loop
    const animate = () => {
      const { x, y } = posRef.current;

      // Update dot position instantly
      if (dotRef.current) {
        dotRef.current.style.transform = `translate(${x}px, ${y}px)`;
      }

      // Update ring with slight lag
      if (ringRef.current) {
        const ringStyle = ringRef.current.style;
        const currentX = parseFloat(ringStyle.getPropertyValue("--rx") || x);
        const currentY = parseFloat(ringStyle.getPropertyValue("--ry") || y);
        const newX = currentX + (x - currentX) * 0.15;
        const newY = currentY + (y - currentY) * 0.15;
        ringStyle.setProperty("--rx", newX);
        ringStyle.setProperty("--ry", newY);
        ringStyle.transform = `translate(${newX}px, ${newY}px)`;
      }

      // Update trail particles with increasing lag
      trailPositions.current.forEach((tp, i) => {
        const prev = i === 0 ? posRef.current : trailPositions.current[i - 1];
        const lag = 0.12 - i * 0.01;
        tp.x += (prev.x - tp.x) * lag;
        tp.y += (prev.y - tp.y) * lag;

        if (trailRefs.current[i]) {
          trailRefs.current[i].style.transform = `translate(${tp.x}px, ${tp.y}px)`;
        }
      });

      rafRef.current = requestAnimationFrame(animate);
    };

    rafRef.current = requestAnimationFrame(animate);

    return () => {
      cancelAnimationFrame(rafRef.current);
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseover", handleMouseOver);
      document.documentElement.removeEventListener("mouseleave", handleMouseLeave);
      document.documentElement.removeEventListener("mouseenter", handleMouseEnter);
    };
  }, [visible]);

  // Hide on touch devices
  if (typeof window !== "undefined" && "ontouchstart" in window) return null;

  return (
    <div
      className="pointer-events-none fixed inset-0 z-[9999]"
      style={{ opacity: visible ? 1 : 0, transition: "opacity 0.3s" }}
    >
      {/* Trail particles */}
      {Array.from({ length: TRAIL_COUNT }, (_, i) => (
        <div
          key={i}
          ref={(el) => { trailRefs.current[i] = el; }}
          className="fixed top-0 left-0 -translate-x-1/2 -translate-y-1/2 rounded-full"
          style={{
            width: Math.max(2, 6 - i * 0.6),
            height: Math.max(2, 6 - i * 0.6),
            background: `radial-gradient(circle, rgba(0, 240, 255, ${0.3 - i * 0.03}), transparent)`,
            filter: `blur(${i * 0.3}px)`,
          }}
        />
      ))}

      {/* Ring (follows with lag, scales on hover) */}
      <div
        ref={ringRef}
        className="fixed top-0 left-0 -translate-x-1/2 -translate-y-1/2 rounded-full border transition-[width,height,border-color] duration-200"
        style={{
          width: isHovering ? 48 : 32,
          height: isHovering ? 48 : 32,
          marginLeft: isHovering ? -24 : -16,
          marginTop: isHovering ? -24 : -16,
          borderColor: isHovering
            ? "rgba(168, 85, 247, 0.5)"
            : "rgba(0, 240, 255, 0.2)",
          background: isHovering
            ? "rgba(168, 85, 247, 0.05)"
            : "transparent",
        }}
      />

      {/* Center dot */}
      <div
        ref={dotRef}
        className="fixed top-0 left-0 rounded-full"
        style={{
          width: 6,
          height: 6,
          marginLeft: -3,
          marginTop: -3,
          background: isHovering
            ? "rgba(168, 85, 247, 0.9)"
            : "rgba(0, 240, 255, 0.8)",
          boxShadow: isHovering
            ? "0 0 12px rgba(168, 85, 247, 0.6)"
            : "0 0 8px rgba(0, 240, 255, 0.4)",
          transition: "background 0.2s, box-shadow 0.2s",
        }}
      />
    </div>
  );
}
