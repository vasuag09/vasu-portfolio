import { describe, expect, it } from "vitest";

import { resolveCursorEnabled, cursorMode } from "../cursor-state";

describe("resolveCursorEnabled", () => {
  it("enables only for fine pointers that can hover, with motion allowed", () => {
    expect(
      resolveCursorEnabled({ finePointer: true, canHover: true, reducedMotion: false }),
    ).toBe(true);
  });

  it("disables on touch (coarse pointer)", () => {
    expect(
      resolveCursorEnabled({ finePointer: false, canHover: false, reducedMotion: false }),
    ).toBe(false);
  });

  it("disables when hover is unavailable (stylus/hybrid edge)", () => {
    expect(
      resolveCursorEnabled({ finePointer: true, canHover: false, reducedMotion: false }),
    ).toBe(false);
  });

  it("disables under prefers-reduced-motion (lerped trailing is motion)", () => {
    expect(
      resolveCursorEnabled({ finePointer: true, canHover: true, reducedMotion: true }),
    ).toBe(false);
  });
});

describe("cursorMode", () => {
  it("is default when idle", () => {
    expect(cursorMode({ overInteractive: false, pressed: false })).toBe("default");
  });

  it("morphs over interactive targets", () => {
    expect(cursorMode({ overInteractive: true, pressed: false })).toBe("interactive");
  });

  it("pressed wins over interactive", () => {
    expect(cursorMode({ overInteractive: true, pressed: true })).toBe("pressed");
    expect(cursorMode({ overInteractive: false, pressed: true })).toBe("pressed");
  });
});
