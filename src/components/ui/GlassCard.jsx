import React from "react";
import { motion } from "framer-motion";

/**
 * Glassmorphism card with neural-themed hover glow.
 * Use `interactive` for cards that lift on hover.
 * Use `glow` to specify glow color: "cyan" | "purple" | "amber".
 */
export default function GlassCard({
  children,
  className = "",
  interactive = true,
  glow = null,
  delay = 0,
  onClick,
  ...props
}) {
  const glowStyles = {
    cyan: "hover:border-[rgba(0,212,255,0.25)] hover:shadow-[0_0_40px_rgba(0,212,255,0.08)]",
    purple: "hover:border-[rgba(139,92,246,0.25)] hover:shadow-[0_0_40px_rgba(139,92,246,0.08)]",
    amber: "hover:border-[rgba(245,158,11,0.25)] hover:shadow-[0_0_40px_rgba(245,158,11,0.08)]",
  };

  const glowClass = glow ? glowStyles[glow] || glowStyles.cyan : glowStyles.cyan;

  return (
    <motion.div
      initial={{ opacity: 0, y: 24 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay, ease: "easeOut" }}
      whileHover={interactive ? { y: -3 } : undefined}
      onClick={onClick}
      className={`
        glass-card
        ${interactive ? `cursor-pointer ${glowClass}` : ""}
        ${className}
      `}
      {...props}
    >
      {children}
    </motion.div>
  );
}
