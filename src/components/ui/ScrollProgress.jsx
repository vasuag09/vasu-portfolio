import React, { useState, useEffect } from "react";

/**
 * Fixed top bar showing page scroll progress as a gradient line.
 */
export default function ScrollProgress() {
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    const handleScroll = () => {
      const scrollTop = document.documentElement.scrollTop;
      const scrollHeight =
        document.documentElement.scrollHeight - window.innerHeight;
      setProgress(scrollHeight > 0 ? (scrollTop / scrollHeight) * 100 : 0);
    };

    window.addEventListener("scroll", handleScroll, { passive: true });
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  if (progress < 1) return null;

  return (
    <div
      className="fixed top-0 left-0 z-50 h-[2px] hover:h-[3px] transition-[height]"
      style={{
        width: `${progress}%`,
        background: "linear-gradient(90deg, var(--accent-electric), var(--accent-violet))",
      }}
      role="progressbar"
      aria-valuenow={Math.round(progress)}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-label="Page scroll progress"
    />
  );
}
