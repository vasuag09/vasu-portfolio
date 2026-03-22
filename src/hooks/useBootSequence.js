import { useState, useEffect, useRef } from "react";

const BOOT_SEQUENCE = [
  "Initializing kernel...",
  "Loading neural modules...",
  "Mounting React frontend...",
  "Establishing secure uplink...",
  "System ready.",
];

const STEP_DELAY = 400;

function prefersReducedMotion() {
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

/**
 * Manages the boot sequence animation with skip support.
 * Automatically skips if the user prefers reduced motion.
 * Returns { bootSequence, isBooted, skipBoot }
 */
export function useBootSequence() {
  const [bootSequence, setBootSequence] = useState(() =>
    prefersReducedMotion() ? BOOT_SEQUENCE : [],
  );
  const [isBooted, setIsBooted] = useState(() => prefersReducedMotion());
  const timeoutIds = useRef([]);

  useEffect(() => {
    if (isBooted) return; // Already booted (reduced motion)

    timeoutIds.current = [];

    let delay = 0;
    BOOT_SEQUENCE.forEach((step, index) => {
      const id = setTimeout(() => {
        setBootSequence((prev) => [...prev, step]);
        if (index === BOOT_SEQUENCE.length - 1) setIsBooted(true);
      }, delay);
      timeoutIds.current.push(id);
      delay += STEP_DELAY;
    });

    return () => {
      timeoutIds.current.forEach(clearTimeout);
      timeoutIds.current = [];
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const skipBoot = () => {
    timeoutIds.current.forEach(clearTimeout);
    timeoutIds.current = [];
    setBootSequence(BOOT_SEQUENCE);
    setIsBooted(true);
  };

  return { bootSequence, isBooted, skipBoot };
}
