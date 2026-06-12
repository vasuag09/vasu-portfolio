import { describe, expect, it } from "vitest";

import { landingOffset } from "../scroll-landing";

describe("landingOffset", () => {
  it("centers a tall section's content band in the viewport", () => {
    // 280svh section @ 900px viewport = 2520px tall; content sits at its
    // vertical center. Landing must scroll PAST the top by (h - vh) / 2.
    expect(landingOffset(2520, 900, "center")).toBe(810);
  });

  it("is zero for align=start sections (hero pins content to the first viewport)", () => {
    expect(landingOffset(1980, 900, "start")).toBe(0);
  });

  it("never returns negative for sections shorter than the viewport", () => {
    expect(landingOffset(600, 900, "center")).toBe(0);
  });

  it("rounds to whole pixels (Lenis target)", () => {
    expect(landingOffset(2521, 900, "center")).toBe(811);
    expect(Number.isInteger(landingOffset(2521, 900, "center"))).toBe(true);
  });
});
