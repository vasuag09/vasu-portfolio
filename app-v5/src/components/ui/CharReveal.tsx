"use client";

import { useEffect, useState } from "react";
import { BOOT_COMPLETE_EVENT } from "@/lib/boot-state";

interface CharRevealProps {
  text: string;
  /** ms between successive characters. */
  stagger?: number;
  className?: string;
}

/**
 * Char-level staggered reveal for the hero name (design elevation P1.6 —
 * hand-rolled, no SplitText dependency).
 *
 * SSR/no-boot safety: characters render fully visible with NO animation
 * class — the Phase-8 reveal contract (server paint never hidden) holds.
 * The animation plays only when the boot overlay was up (content was
 * covered anyway) and fires on BOOT_COMPLETE_EVENT, so the name assembles
 * exactly as the overlay fades — the boot→hero handoff moment.
 */
export function CharReveal({ text, stagger = 26, className }: CharRevealProps) {
  const [play, setPlay] = useState(false);

  useEffect(() => {
    if (document.documentElement.getAttribute("data-boot") !== "play") return;
    const onComplete = () => setPlay(true);
    window.addEventListener(BOOT_COMPLETE_EVENT, onComplete, { once: true });
    return () => window.removeEventListener(BOOT_COMPLETE_EVENT, onComplete);
  }, []);

  let charIndex = 0;
  return (
    <span aria-label={text} className={className}>
      {text.split(" ").map((word, w) => (
        <span key={w} aria-hidden="true" className="inline-block whitespace-nowrap">
          {word.split("").map((char, c) => {
            const delay = charIndex * stagger;
            charIndex += 1;
            return (
              <span
                key={c}
                className={play ? "char-reveal-in" : "inline-block"}
                style={play ? { animationDelay: `${delay}ms` } : undefined}
              >
                {char}
              </span>
            );
          })}
          {w < text.split(" ").length - 1 ? " " : null}
        </span>
      ))}
    </span>
  );
}
