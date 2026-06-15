"use client";

import type { CSSProperties, ReactNode, RefObject } from "react";
import { useRevealOnce } from "@/hooks/useRevealOnce";

interface RevealProps {
  children: ReactNode;
  /** Stagger delay in ms. */
  delay?: number;
  /** Scroll root when used inside a scrollable panel. */
  root?: RefObject<HTMLElement | null>;
  className?: string;
}

/**
 * Once-only reveal wrapper. Animates opacity + translateY ONLY
 * (compositor-friendly, zero layout shift — the element always occupies
 * its final box). reduced-motion.css force-disables the transition, and
 * useRevealOnce reports revealed immediately, so content is never hidden.
 */
export function Reveal({ children, delay = 0, root, className }: RevealProps) {
  const { ref, revealed } = useRevealOnce<HTMLDivElement>({ root });

  const style: CSSProperties = {
    opacity: revealed ? 1 : 0,
    transform: revealed ? "translateY(0)" : "translateY(18px)",
    transition: `opacity var(--duration-slow) var(--ease-out-expo) ${delay}ms, transform var(--duration-slow) var(--ease-out-expo) ${delay}ms`,
  };

  return (
    <div ref={ref} className={className} style={style}>
      {children}
    </div>
  );
}
