"use client";

import { useEffect, useRef, useState } from "react";
import { useReducedMotion } from "@/hooks/useReducedMotion";
import type { VeoClipSource } from "@/lib/veo-sources";

interface VeoClipProps {
  sources: readonly VeoClipSource[];
  poster: string;
  /** Accessible description of what the clip shows. */
  label: string;
  /** CSS aspect-ratio — explicit so the slot can never cause CLS. */
  aspectRatio?: string;
  className?: string;
}

/**
 * Lazy cinematic clip slot (Phase 4, consumed by the Phase-5 case-study
 * template). Behavior contract:
 *  - reduced motion → poster image only, no video element at all (ADR-6)
 *  - video mounts only when scrolled near (300px margin), preload=none
 *  - plays only while on screen; pauses when scrolled away
 *  - muted/loop/playsInline — never audio, never controls, pure ambience
 */
export function VeoClip({
  sources,
  poster,
  label,
  aspectRatio = "16 / 9",
  className,
}: VeoClipProps) {
  const reducedMotion = useReducedMotion();
  const containerRef = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const [nearViewport, setNearViewport] = useState(false);

  // Mount the <video> once the slot approaches the viewport.
  useEffect(() => {
    if (reducedMotion || nearViewport) return;
    const el = containerRef.current;
    if (!el) return;
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries.some((entry) => entry.isIntersecting)) {
          setNearViewport(true);
          observer.disconnect();
        }
      },
      { rootMargin: "300px" },
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, [reducedMotion, nearViewport]);

  // Play only while actually visible.
  useEffect(() => {
    if (!nearViewport) return;
    const el = containerRef.current;
    const video = videoRef.current;
    if (!el || !video) return;
    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries.some((entry) => entry.isIntersecting);
        if (visible) {
          // Autoplay can reject (e.g. power saving) — poster keeps showing.
          video.play().catch(() => undefined);
        } else {
          video.pause();
        }
      },
      { threshold: 0.2 },
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, [nearViewport]);

  return (
    <div
      ref={containerRef}
      className={className}
      style={{
        aspectRatio,
        background: "var(--bg-elevated)",
        overflow: "hidden",
      }}
    >
      {reducedMotion || !nearViewport ? (
        /* Posters are pre-sized AVIF stills in a fixed-aspect slot;
           next/image adds nothing here. */
        // eslint-disable-next-line @next/next/no-img-element
        <img
          src={poster}
          alt={label}
          loading="lazy"
          className="h-full w-full object-cover"
        />
      ) : (
        <video
          ref={videoRef}
          muted
          loop
          playsInline
          preload="none"
          poster={poster}
          aria-label={label}
          className="h-full w-full object-cover"
        >
          {sources.map((source) => (
            <source key={source.src} src={source.src} type={source.type} />
          ))}
        </video>
      )}
    </div>
  );
}
