import { describe, expect, it } from "vitest";

import { probeStrengthTarget, resolveProbeEnabled } from "../probe-state";

const FULL = { finePointer: true, canHover: true, reducedMotion: false };

describe("resolveProbeEnabled", () => {
  it("enabled for a fine pointer that can hover with motion allowed", () => {
    expect(resolveProbeEnabled(FULL)).toBe(true);
  });

  it("disabled under reduced motion (probe lift is decorative motion)", () => {
    expect(resolveProbeEnabled({ ...FULL, reducedMotion: true })).toBe(false);
  });

  it("disabled on a coarse pointer (touch — no hover-driven probe)", () => {
    expect(resolveProbeEnabled({ ...FULL, finePointer: false })).toBe(false);
  });

  it("disabled when hover is unavailable", () => {
    expect(resolveProbeEnabled({ ...FULL, canHover: false })).toBe(false);
  });
});

describe("probeStrengthTarget", () => {
  it("is full (1) when enabled", () => {
    expect(probeStrengthTarget(FULL)).toBe(1);
  });

  it("is off (0) when gated out", () => {
    expect(probeStrengthTarget({ ...FULL, reducedMotion: true })).toBe(0);
  });
});
