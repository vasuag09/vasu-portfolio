import { describe, expect, it } from "vitest";

import {
  HEADING_VIEWPORT_FRACTION,
  headingScrollOffset,
} from "../scroll-landing";

describe("headingScrollOffset", () => {
  it("rests the heading at the optical headline line (22% from top)", () => {
    expect(headingScrollOffset(900, "center")).toBe(-198);
    expect(headingScrollOffset(900, "center")).toBe(
      -Math.round(900 * HEADING_VIEWPORT_FRACTION),
    );
  });

  it("is zero for align=start sections (hero pins to the section top)", () => {
    expect(headingScrollOffset(900, "start")).toBe(0);
  });

  it("returns whole pixels (Lenis target)", () => {
    expect(Number.isInteger(headingScrollOffset(901, "center"))).toBe(true);
  });
});
