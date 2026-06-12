/**
 * ADR-2 device tiers. Pure resolution logic — the useParticleConfig hook
 * gathers DeviceCaps from the browser and calls this. Safari and Firefox
 * never expose navigator.deviceMemory, so `undefined` must read as
 * "unknown, assume capable", never as low-end.
 */

export interface DeviceCaps {
  isMobile: boolean;
  deviceMemory: number | undefined;
  hardwareConcurrency: number | undefined;
}

export type TierName = "desktop" | "mobile" | "low";

export interface ParticleTier {
  name: TierName;
  particleCount: number;
  bloom: boolean;
  depthOfField: boolean;
  /** Max device-pixel-ratio the canvas should render at. */
  maxDpr: number;
}

const TIERS: Record<TierName, ParticleTier> = {
  desktop: {
    name: "desktop",
    particleCount: 80_000,
    bloom: true,
    // Disabled in the design-elevation perf pass: full-res Bokeh barely
    // read visually. Schema kept for a future half-res variant.
    depthOfField: false,
    // 1.75 (was 2): ~23% fewer fragments, invisible on a particle field.
    maxDpr: 1.75,
  },
  mobile: {
    name: "mobile",
    particleCount: 10_000,
    bloom: true,
    depthOfField: false,
    maxDpr: 1.5,
  },
  low: {
    name: "low",
    particleCount: 5_000,
    bloom: false,
    depthOfField: false,
    maxDpr: 1,
  },
};

export function resolveParticleTier(caps: DeviceCaps): ParticleTier {
  // Explicitly-known weak hardware → floor tier (ADR-2 video-fallback
  // candidates; Phase 8 layers the video tier on top of this signal).
  const lowMemory = caps.deviceMemory !== undefined && caps.deviceMemory <= 4;
  const lowCores =
    caps.hardwareConcurrency !== undefined && caps.hardwareConcurrency <= 2;
  if (lowMemory || lowCores) return TIERS.low;

  return caps.isMobile ? TIERS.mobile : TIERS.desktop;
}
