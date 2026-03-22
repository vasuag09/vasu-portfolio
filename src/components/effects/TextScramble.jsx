import React, { useState, useRef, useCallback } from "react";

/**
 * Hover-triggered text scramble effect.
 * On mouse enter, scrambles text with random characters then resolves
 * character-by-character from left to right.
 */
const CHARS = "!@#$%^&*()_+-=[]{}|;:,.<>?0123456789ABCDEF";

export default function TextScramble({
  text,
  className = "",
  as: Tag = "span",
  duration = 400,
  ...props
}) {
  const [displayText, setDisplayText] = useState(text);
  const animRef = useRef(null);
  const isAnimating = useRef(false);

  const prefersReducedMotion =
    typeof window !== "undefined" &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  const scramble = useCallback(() => {
    if (prefersReducedMotion || isAnimating.current) return;
    isAnimating.current = true;

    const chars = text.split("");
    const startTime = performance.now();
    const charDelay = duration / chars.length;

    const animate = (now) => {
      const elapsed = now - startTime;
      const resolved = Math.floor(elapsed / charDelay);

      const result = chars.map((char, i) => {
        if (char === " ") return " ";
        if (i < resolved) return char;
        return CHARS[Math.floor(Math.random() * CHARS.length)];
      });

      setDisplayText(result.join(""));

      if (resolved < chars.length) {
        animRef.current = requestAnimationFrame(animate);
      } else {
        setDisplayText(text);
        isAnimating.current = false;
      }
    };

    animRef.current = requestAnimationFrame(animate);
  }, [text, duration, prefersReducedMotion]);

  const reset = useCallback(() => {
    if (animRef.current) cancelAnimationFrame(animRef.current);
    setDisplayText(text);
    isAnimating.current = false;
  }, [text]);

  return (
    <Tag
      className={className}
      onMouseEnter={scramble}
      onMouseLeave={reset}
      {...props}
    >
      {displayText}
    </Tag>
  );
}
