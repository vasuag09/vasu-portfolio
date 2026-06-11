"use client";

import { useMemo } from "react";
import {
  resolveParticleTier,
  type DeviceCaps,
  type ParticleTier,
} from "@/lib/particle-config";

/** Narrow typing for the non-standard navigator.deviceMemory (Chromium only). */
interface NavigatorWithMemory extends Navigator {
  deviceMemory?: number;
}

function readDeviceCaps(): DeviceCaps {
  const nav = navigator as NavigatorWithMemory;
  return {
    isMobile:
      /Android|iPhone|iPad|iPod/i.test(nav.userAgent) ||
      // iPadOS 13+ reports as Mac; coarse pointer disambiguates.
      (nav.maxTouchPoints > 1 && /Mac/i.test(nav.userAgent)),
    deviceMemory: nav.deviceMemory,
    hardwareConcurrency: nav.hardwareConcurrency,
  };
}

/**
 * ADR-2: resolved once per mount — tier changes require a reload, which is
 * fine (hardware does not hot-swap). Client-only; call below the ssr:false
 * canvas boundary.
 */
export function useParticleConfig(): ParticleTier {
  return useMemo(() => resolveParticleTier(readDeviceCaps()), []);
}
