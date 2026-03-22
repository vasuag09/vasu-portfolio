import React, { useRef, useState, useCallback } from "react";

/**
 * Wrapper that creates a gravitational pull toward the cursor.
 * Elements subtly shift toward the mouse when hovering nearby.
 */
export default function MagneticElement({
  children,
  strength = 0.3,
  className = "",
  as: Tag = "div",
  ...props
}) {
  const ref = useRef(null);
  const [transform, setTransform] = useState("translate(0px, 0px)");

  const prefersReducedMotion =
    typeof window !== "undefined" &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  const handleMouseMove = useCallback(
    (e) => {
      if (prefersReducedMotion || !ref.current) return;

      const rect = ref.current.getBoundingClientRect();
      const centerX = rect.left + rect.width / 2;
      const centerY = rect.top + rect.height / 2;

      const dx = (e.clientX - centerX) * strength;
      const dy = (e.clientY - centerY) * strength;

      setTransform(`translate(${dx}px, ${dy}px)`);
    },
    [strength, prefersReducedMotion]
  );

  const handleMouseLeave = useCallback(() => {
    setTransform("translate(0px, 0px)");
  }, []);

  return (
    <Tag
      ref={ref}
      className={className}
      style={{
        transform,
        transition: "transform 0.3s cubic-bezier(0.23, 1, 0.32, 1)",
        willChange: "transform",
      }}
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
      {...props}
    >
      {children}
    </Tag>
  );
}
