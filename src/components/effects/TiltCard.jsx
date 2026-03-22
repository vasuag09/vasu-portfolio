import React, { useRef, useState, useCallback } from "react";

/**
 * Perspective-aware tilt card with holographic shimmer effect.
 * Tracks mouse position relative to card center and applies 3D rotation.
 * Includes a moving gradient overlay that follows the cursor.
 */
export default function TiltCard({
  children,
  className = "",
  maxTilt = 8,
  glare = true,
  ...props
}) {
  const cardRef = useRef(null);
  const [transform, setTransform] = useState("");
  const [shimmerStyle, setShimmerStyle] = useState({});
  const [isHovering, setIsHovering] = useState(false);

  // Check reduced motion preference
  const prefersReducedMotion =
    typeof window !== "undefined" &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  const handleMouseMove = useCallback(
    (e) => {
      if (prefersReducedMotion || !cardRef.current) return;

      const rect = cardRef.current.getBoundingClientRect();
      const centerX = rect.left + rect.width / 2;
      const centerY = rect.top + rect.height / 2;
      const mouseX = e.clientX - centerX;
      const mouseY = e.clientY - centerY;

      // Normalize to -1...1
      const normalX = mouseX / (rect.width / 2);
      const normalY = mouseY / (rect.height / 2);

      // Tilt angles (inverted Y for natural feel)
      const rotateX = -normalY * maxTilt;
      const rotateY = normalX * maxTilt;

      setTransform(
        `perspective(800px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) scale3d(1.02, 1.02, 1.02)`
      );

      // Shimmer gradient follows cursor
      if (glare) {
        const angle = Math.atan2(mouseY, mouseX) * (180 / Math.PI) + 180;
        setShimmerStyle({
          background: `linear-gradient(${angle}deg, transparent 0%, rgba(0, 240, 255, 0.04) 30%, rgba(168, 85, 247, 0.08) 50%, rgba(0, 240, 255, 0.04) 70%, transparent 100%)`,
          opacity: 1,
        });
      }
    },
    [maxTilt, glare, prefersReducedMotion]
  );

  const handleMouseLeave = useCallback(() => {
    setTransform("");
    setShimmerStyle({ opacity: 0 });
    setIsHovering(false);
  }, []);

  const handleMouseEnter = useCallback(() => {
    setIsHovering(true);
  }, []);

  return (
    <div
      ref={cardRef}
      className={`relative ${className}`}
      style={{
        transform: transform || "perspective(800px) rotateX(0deg) rotateY(0deg)",
        transition: isHovering
          ? "transform 0.1s ease-out"
          : "transform 0.5s cubic-bezier(0.23, 1, 0.32, 1)",
        transformStyle: "preserve-3d",
        willChange: "transform",
      }}
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
      onMouseEnter={handleMouseEnter}
      {...props}
    >
      {children}

      {/* Holographic shimmer overlay */}
      {glare && (
        <div
          className="absolute inset-0 rounded-xl pointer-events-none z-10"
          style={{
            ...shimmerStyle,
            transition: "opacity 0.4s ease",
            borderRadius: "inherit",
          }}
        />
      )}

      {/* Iridescent border glow */}
      {isHovering && (
        <div
          className="absolute inset-0 rounded-xl pointer-events-none"
          style={{
            border: "1px solid rgba(0, 240, 255, 0.15)",
            boxShadow: "0 0 30px rgba(0, 240, 255, 0.06), inset 0 0 30px rgba(168, 85, 247, 0.03)",
            borderRadius: "inherit",
            transition: "all 0.3s ease",
          }}
        />
      )}
    </div>
  );
}
