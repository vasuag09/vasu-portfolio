import React, { useRef, useEffect } from "react";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";

gsap.registerPlugin(ScrollTrigger);

/**
 * GSAP ScrollTrigger wrapper for cinematic scroll-driven animations.
 * Wraps children and animates them into view when scrolled to.
 */

const ANIMATIONS = {
  fadeUp: { from: { opacity: 0, y: 60 }, to: { opacity: 1, y: 0 } },
  fadeIn: { from: { opacity: 0 }, to: { opacity: 1 } },
  slideLeft: { from: { opacity: 0, x: -40 }, to: { opacity: 1, x: 0 } },
  slideRight: { from: { opacity: 0, x: 40 }, to: { opacity: 1, x: 0 } },
  scaleIn: { from: { opacity: 0, scale: 0.92 }, to: { opacity: 1, scale: 1 } },
};

export default function ScrollReveal({
  children,
  animation = "fadeUp",
  stagger = 0.08,
  delay = 0,
  duration = 0.8,
  threshold = "top 88%",
  className = "",
  once = true,
}) {
  const containerRef = useRef(null);

  const prefersReducedMotion =
    typeof window !== "undefined" &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  useEffect(() => {
    if (prefersReducedMotion || !containerRef.current) return;

    const anim = ANIMATIONS[animation] || ANIMATIONS.fadeUp;
    const targets = containerRef.current.children;

    if (targets.length === 0) return;

    // Set initial state
    gsap.set(targets, anim.from);

    const el = containerRef.current;

    const tl = gsap.to(targets, {
      ...anim.to,
      duration,
      delay,
      stagger,
      ease: "power3.out",
      scrollTrigger: {
        trigger: el,
        start: threshold,
        toggleActions: once ? "play none none none" : "play none none reverse",
      },
    });

    return () => {
      tl.kill();
      ScrollTrigger.getAll().forEach((st) => {
        if (st.trigger === el) st.kill();
      });
    };
  }, [animation, stagger, delay, duration, threshold, once, prefersReducedMotion]);

  return (
    <div ref={containerRef} className={className}>
      {children}
    </div>
  );
}
