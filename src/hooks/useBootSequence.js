import { useState, useEffect, useRef } from "react";

const BOOT_SEQUENCE = [
  "Initializing kernel...",
  "Loading neural modules...",
  "Mounting React frontend...",
  "Establishing secure uplink...",
  "System ready.",
];

const STEP_DELAY = 400;

/**
 * Manages the boot sequence animation with skip support.
 * Returns { bootSequence, isBooted, skipBoot }
 */
export function useBootSequence() {
  const [bootSequence, setBootSequence] = useState([]);
  const [isBooted, setIsBooted] = useState(false);
  const timeoutIds = useRef([]);

  useEffect(() => {
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
  }, []);

  const skipBoot = () => {
    timeoutIds.current.forEach(clearTimeout);
    timeoutIds.current = [];
    setBootSequence(BOOT_SEQUENCE);
    setIsBooted(true);
  };

  return { bootSequence, isBooted, skipBoot };
}
