import React, { useEffect, useRef, useState } from "react";

/**
 * Animated number counter — counts from 0 to target value when scrolled into view.
 * Handles both numeric values ("10", "3.82") and text values ("FULL").
 * Uses spring-based easing for organic, decelerating feel.
 */
export default function AnimatedCounter({
  value,
  duration = 2000,
  className = "",
}) {
  const ref = useRef(null);
  const [displayValue, setDisplayValue] = useState("0");
  const [hasAnimated, setHasAnimated] = useState(false);

  const isNumeric = !isNaN(parseFloat(value)) && isFinite(value);
  const targetNum = isNumeric ? parseFloat(value) : 0;
  const decimals = isNumeric && value.includes(".") ? value.split(".")[1].length : 0;

  useEffect(() => {
    const el = ref.current;
    if (!el || hasAnimated) return;

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setHasAnimated(true);
          observer.disconnect();

          if (!isNumeric) {
            // For text values, do a scramble reveal
            const chars = "!@#$%^&*0123456789ABCDEF";
            const target = value;
            let frame = 0;
            const totalFrames = 20;

            const interval = setInterval(() => {
              frame++;
              const progress = frame / totalFrames;
              const resolved = Math.floor(progress * target.length);

              const result = target
                .split("")
                .map((char, i) => {
                  if (i < resolved) return char;
                  return chars[Math.floor(Math.random() * chars.length)];
                })
                .join("");

              setDisplayValue(result);

              if (frame >= totalFrames) {
                setDisplayValue(target);
                clearInterval(interval);
              }
            }, duration / totalFrames);
            return;
          }

          // Numeric count-up with spring easing
          const startTime = performance.now();

          const animate = (now) => {
            const elapsed = now - startTime;
            const progress = Math.min(elapsed / duration, 1);

            // Spring-like ease-out: fast start, slow deceleration
            const eased = 1 - Math.pow(1 - progress, 4);
            const current = eased * targetNum;

            setDisplayValue(current.toFixed(decimals));

            if (progress < 1) {
              requestAnimationFrame(animate);
            } else {
              setDisplayValue(targetNum.toFixed(decimals));
            }
          };

          requestAnimationFrame(animate);
        }
      },
      { threshold: 0.3 },
    );

    observer.observe(el);
    return () => observer.disconnect();
  }, [value, duration, isNumeric, targetNum, decimals, hasAnimated]);

  return (
    <span ref={ref} className={className}>
      {hasAnimated ? displayValue : isNumeric ? "0" : value}
    </span>
  );
}
