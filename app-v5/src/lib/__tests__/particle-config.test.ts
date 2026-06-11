import { describe, expect, it } from "vitest";
import {
  resolveParticleTier,
  type DeviceCaps,
} from "../particle-config";

const desktop: DeviceCaps = {
  isMobile: false,
  deviceMemory: 16,
  hardwareConcurrency: 10,
};

const phone: DeviceCaps = {
  isMobile: true,
  deviceMemory: 6,
  hardwareConcurrency: 8,
};

const lowEnd: DeviceCaps = {
  isMobile: true,
  deviceMemory: 4,
  hardwareConcurrency: 4,
};

describe("resolveParticleTier (ADR-2 tiers)", () => {
  it("desktop gets the full experience: 60–100k particles, bloom + DoF", () => {
    const tier = resolveParticleTier(desktop);
    expect(tier.name).toBe("desktop");
    expect(tier.particleCount).toBeGreaterThanOrEqual(60_000);
    expect(tier.particleCount).toBeLessThanOrEqual(100_000);
    expect(tier.bloom).toBe(true);
    expect(tier.depthOfField).toBe(true);
  });

  it("modern mobile gets the live tier: ~10k particles, bloom only", () => {
    const tier = resolveParticleTier(phone);
    expect(tier.name).toBe("mobile");
    expect(tier.particleCount).toBeGreaterThanOrEqual(5_000);
    expect(tier.particleCount).toBeLessThanOrEqual(10_000);
    expect(tier.bloom).toBe(true);
    expect(tier.depthOfField).toBe(false); // ADR-2: bloom only, no DoF
  });

  it("low-end (deviceMemory ≤ 4GB) drops to the floor tier", () => {
    const tier = resolveParticleTier(lowEnd);
    expect(tier.name).toBe("low");
    expect(tier.particleCount).toBeLessThanOrEqual(5_000);
    expect(tier.bloom).toBe(false);
    expect(tier.depthOfField).toBe(false);
  });

  it("treats missing capability data conservatively (unknown ≠ low-end)", () => {
    // deviceMemory is undefined in Safari/Firefox — must not force the floor tier.
    const tier = resolveParticleTier({
      isMobile: false,
      deviceMemory: undefined,
      hardwareConcurrency: undefined,
    });
    expect(tier.name).toBe("desktop");
  });

  it("desktop with very low memory still demotes to low", () => {
    const tier = resolveParticleTier({
      isMobile: false,
      deviceMemory: 2,
      hardwareConcurrency: 2,
    });
    expect(tier.name).toBe("low");
  });
});
