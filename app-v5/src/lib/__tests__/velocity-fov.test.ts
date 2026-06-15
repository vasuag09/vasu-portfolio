import { describe, expect, it, vi } from "vitest";

import {
  BASE_FOV,
  FOV_VELOCITY_REF,
  MAX_FOV_BOOST,
  stepFov,
  targetFov,
} from "../velocity-fov";

describe("targetFov", () => {
  it("rests at base FOV when velocity is zero", () => {
    expect(targetFov({ velocity: 0, diveBlend: 0, reducedMotion: false })).toBe(
      BASE_FOV,
    );
  });

  it("reaches the +MAX cap at the reference velocity (ADR-9 ≤ +6°)", () => {
    const fov = targetFov({
      velocity: FOV_VELOCITY_REF,
      diveBlend: 0,
      reducedMotion: false,
    });
    expect(fov).toBeCloseTo(BASE_FOV + MAX_FOV_BOOST, 5);
  });

  it("clamps the boost at the cap beyond the reference velocity", () => {
    const fov = targetFov({
      velocity: FOV_VELOCITY_REF * 10,
      diveBlend: 0,
      reducedMotion: false,
    });
    expect(fov).toBe(BASE_FOV + MAX_FOV_BOOST);
  });

  it("treats upward (negative) velocity by magnitude", () => {
    const up = targetFov({
      velocity: -FOV_VELOCITY_REF,
      diveBlend: 0,
      reducedMotion: false,
    });
    expect(up).toBeCloseTo(BASE_FOV + MAX_FOV_BOOST, 5);
  });

  it("is suppressed to base FOV during a dive (ADR-9 camera authority)", () => {
    expect(
      targetFov({ velocity: FOV_VELOCITY_REF, diveBlend: 0.01, reducedMotion: false }),
    ).toBe(BASE_FOV);
  });

  it("is suppressed to base FOV under reduced motion", () => {
    expect(
      targetFov({ velocity: FOV_VELOCITY_REF, diveBlend: 0, reducedMotion: true }),
    ).toBe(BASE_FOV);
  });

  it("falls back to base FOV on a NaN velocity (no poison into projection)", () => {
    expect(targetFov({ velocity: NaN, diveBlend: 0, reducedMotion: false })).toBe(
      BASE_FOV,
    );
  });
});

describe("stepFov", () => {
  it("damps the camera FOV toward the target and refreshes the projection", () => {
    const camera = { fov: BASE_FOV, updateProjectionMatrix: vi.fn() };
    stepFov(camera, BASE_FOV + MAX_FOV_BOOST, 1 / 60);
    expect(camera.fov).toBeGreaterThan(BASE_FOV);
    expect(camera.fov).toBeLessThanOrEqual(BASE_FOV + MAX_FOV_BOOST);
    expect(camera.updateProjectionMatrix).toHaveBeenCalledOnce();
  });

  it("skips the projection refresh when already settled (no per-frame churn)", () => {
    const camera = { fov: BASE_FOV, updateProjectionMatrix: vi.fn() };
    stepFov(camera, BASE_FOV, 1 / 60);
    expect(camera.updateProjectionMatrix).not.toHaveBeenCalled();
  });

  it("converges to the target over many frames", () => {
    const camera = { fov: BASE_FOV, updateProjectionMatrix: vi.fn() };
    for (let i = 0; i < 600; i += 1) {
      stepFov(camera, BASE_FOV + MAX_FOV_BOOST, 1 / 60);
    }
    expect(camera.fov).toBeCloseTo(BASE_FOV + MAX_FOV_BOOST, 2);
  });
});
