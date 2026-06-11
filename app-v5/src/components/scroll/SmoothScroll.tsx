"use client";

import { useEffect } from "react";
import Lenis from "lenis";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { scrollState } from "@/lib/scroll-state";
import { useReducedMotion } from "@/hooks/useReducedMotion";

/**
 * The single scroll authority (ADR-1): native scroll + Lenis smoothing +
 * ONE global ScrollTrigger that scrubs document progress 0–1 into
 * scrollState. No preventDefault anywhere in the chain — keyboard, trackpad
 * momentum, and touch stay native.
 */

const LENIS_LERP = 0.1; // ADR-1: ≤0.12

function measureSectionCenters(): number[] {
  const sections = Array.from(
    document.querySelectorAll<HTMLElement>("[data-chapter]"),
  );
  const maxScroll = document.documentElement.scrollHeight - window.innerHeight;
  if (maxScroll <= 0) return sections.map(() => 0);

  return sections.map((el) => {
    const centered = el.offsetTop + el.offsetHeight / 2 - window.innerHeight / 2;
    return Math.min(1, Math.max(0, centered / maxScroll));
  });
}

export function SmoothScroll() {
  const reducedMotion = useReducedMotion();

  useEffect(() => {
    if (reducedMotion) {
      scrollState.lenis = null;
      return;
    }

    gsap.registerPlugin(ScrollTrigger);

    const lenis = new Lenis({ lerp: LENIS_LERP });
    scrollState.lenis = lenis;

    lenis.on("scroll", () => {
      scrollState.velocity = lenis.velocity;
      ScrollTrigger.update();
    });

    const raf = (time: number) => lenis.raf(time * 1000);
    gsap.ticker.add(raf);
    gsap.ticker.lagSmoothing(0);

    const progressProxy = { value: 0 };
    const tween = gsap.to(progressProxy, {
      value: 1,
      ease: "none",
      onUpdate: () => {
        scrollState.progress = progressProxy.value;
      },
      scrollTrigger: {
        trigger: document.body,
        start: "top top",
        end: "bottom bottom",
        scrub: 1, // ADR-1: ≤1
      },
    });

    const measure = () => {
      scrollState.sectionCenters = measureSectionCenters();
    };
    measure();

    const onResize = () => {
      measure();
      ScrollTrigger.refresh();
    };
    window.addEventListener("resize", onResize);

    return () => {
      window.removeEventListener("resize", onResize);
      tween.scrollTrigger?.kill();
      tween.kill();
      gsap.ticker.remove(raf);
      lenis.destroy();
      scrollState.lenis = null;
    };
  }, [reducedMotion]);

  return null;
}
