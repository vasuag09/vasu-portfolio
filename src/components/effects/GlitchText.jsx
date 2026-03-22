import React, { useState, useEffect } from "react";

/**
 * Text that "decodes" from random characters to the actual text.
 * Used for the hero title on the landing page.
 */
const CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789@#$%&";

export default function GlitchText({
  text,
  className = "",
  delay = 0,
  duration = 1200,
  as: Tag = "span",
}) {
  const [displayText, setDisplayText] = useState("");
  const [started, setStarted] = useState(false);

  useEffect(() => {
    const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (prefersReducedMotion) {
      setDisplayText(text);
      return;
    }

    const startTimer = setTimeout(() => setStarted(true), delay);
    return () => clearTimeout(startTimer);
  }, [delay, text]);

  useEffect(() => {
    if (!started) return;

    const steps = Math.ceil(duration / 30);
    let step = 0;

    const interval = setInterval(() => {
      step++;
      const progress = step / steps;

      const result = text
        .split("")
        .map((char, i) => {
          if (char === " ") return " ";
          const charProgress = (progress - i * 0.02);
          if (charProgress >= 1) return char;
          if (charProgress <= 0) return CHARS[Math.floor(Math.random() * CHARS.length)];
          return Math.random() > charProgress
            ? CHARS[Math.floor(Math.random() * CHARS.length)]
            : char;
        })
        .join("");

      setDisplayText(result);

      if (step >= steps) {
        setDisplayText(text);
        clearInterval(interval);
      }
    }, 30);

    return () => clearInterval(interval);
  }, [started, text, duration]);

  return <Tag className={className}>{displayText || "\u00A0"}</Tag>;
}
