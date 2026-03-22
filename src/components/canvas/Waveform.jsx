import React from "react";

/**
 * SVG waveform decoration for blog post cards.
 * Generates a unique waveform based on a seed string.
 */
export default function Waveform({ seed = "default", className = "" }) {
  // Generate deterministic wave from seed
  let hash = 0;
  for (let i = 0; i < seed.length; i++) {
    hash = ((hash << 5) - hash) + seed.charCodeAt(i);
    hash |= 0;
  }

  const points = [];
  const segments = 40;
  const width = 400;
  const height = 30;
  const mid = height / 2;

  for (let i = 0; i <= segments; i++) {
    const x = (i / segments) * width;
    const noise1 = Math.sin(i * 0.5 + hash) * 6;
    const noise2 = Math.sin(i * 1.2 + hash * 0.7) * 4;
    const noise3 = Math.sin(i * 0.3 + hash * 1.3) * 3;
    const y = mid + noise1 + noise2 + noise3;
    points.push(`${x},${y}`);
  }

  const pathData = `M ${points.join(" L ")}`;

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      className={`w-full h-6 ${className}`}
      preserveAspectRatio="none"
      aria-hidden="true"
    >
      <defs>
        <linearGradient id={`waveGrad-${hash}`} x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stopColor="var(--accent-cyan)" stopOpacity="0.4" />
          <stop offset="100%" stopColor="var(--accent-purple)" stopOpacity="0.4" />
        </linearGradient>
      </defs>
      <path
        d={pathData}
        fill="none"
        stroke={`url(#waveGrad-${hash})`}
        strokeWidth="1.5"
        strokeLinecap="round"
      />
    </svg>
  );
}
