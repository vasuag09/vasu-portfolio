import { describe, expect, it } from "vitest";

import {
  resolveHydrationReveal,
  resolveInitialRevealPhase,
} from "../reveal-state";

const VIEWPORT = 800;

const rect = (top: number, height: number, width = 600) => ({
  top,
  bottom: top + height,
  width,
  height,
});

describe("resolveInitialRevealPhase", () => {
  it("renders visible on the server / hydration pass (LCP + no-JS safety)", () => {
    expect(resolveInitialRevealPhase(false)).toBe("ssr");
  });

  it("starts hidden for late mounts so panel reveals animate without a flash", () => {
    expect(resolveInitialRevealPhase(true)).toBe("hidden");
  });
});

describe("resolveHydrationReveal", () => {
  it("keeps content intersecting the first viewport revealed (hero = LCP)", () => {
    expect(resolveHydrationReveal(rect(100, 400), VIEWPORT)).toBe("revealed");
  });

  it("keeps partially-visible content at the fold revealed", () => {
    expect(resolveHydrationReveal(rect(700, 400), VIEWPORT)).toBe("revealed");
  });

  it("keeps content above the viewport revealed (user already scrolled past)", () => {
    expect(resolveHydrationReveal(rect(-500, 300), VIEWPORT)).toBe("revealed");
  });

  it("hides content entirely below the fold so the scroll-in reveal still fires", () => {
    expect(resolveHydrationReveal(rect(VIEWPORT, 400), VIEWPORT)).toBe("hidden");
    expect(resolveHydrationReveal(rect(2400, 400), VIEWPORT)).toBe("hidden");
  });

  it("defers zero-size rects (display:none ancestors) to the observer", () => {
    expect(resolveHydrationReveal(rect(0, 0, 0), VIEWPORT)).toBe("hidden");
  });
});
