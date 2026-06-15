import { describe, expect, it } from "vitest";

import {
  BOOT_TIMELINE,
  bootPhaseAt,
  scrambleText,
  shouldPlayBoot,
} from "../boot-state";

describe("shouldPlayBoot", () => {
  it("plays for a first visit with motion allowed", () => {
    expect(shouldPlayBoot({ hasBooted: false, reducedMotion: false })).toBe(true);
  });

  it("skips for return visitors", () => {
    expect(shouldPlayBoot({ hasBooted: true, reducedMotion: false })).toBe(false);
  });

  it("skips under prefers-reduced-motion", () => {
    expect(shouldPlayBoot({ hasBooted: false, reducedMotion: true })).toBe(false);
  });
});

describe("BOOT_TIMELINE", () => {
  it("totals 1.5s or less (PLAN Phase-10 exit criterion)", () => {
    const total =
      BOOT_TIMELINE.scrambleMs + BOOT_TIMELINE.holdMs + BOOT_TIMELINE.fadeMs;
    expect(total).toBeLessThanOrEqual(1500);
  });
});

describe("bootPhaseAt", () => {
  it("maps elapsed time to scramble → hold → fade → done", () => {
    expect(bootPhaseAt(0)).toBe("scramble");
    expect(bootPhaseAt(BOOT_TIMELINE.scrambleMs - 1)).toBe("scramble");
    expect(bootPhaseAt(BOOT_TIMELINE.scrambleMs)).toBe("hold");
    expect(bootPhaseAt(BOOT_TIMELINE.scrambleMs + BOOT_TIMELINE.holdMs)).toBe("fade");
    expect(
      bootPhaseAt(
        BOOT_TIMELINE.scrambleMs + BOOT_TIMELINE.holdMs + BOOT_TIMELINE.fadeMs,
      ),
    ).toBe("done");
  });
});

describe("scrambleText", () => {
  // Deterministic glyph source so assertions are stable.
  const glyph = () => "#";

  it("resolves to the exact target at progress 1", () => {
    expect(scrambleText("VASU AGRAWAL", 1, glyph)).toBe("VASU AGRAWAL");
  });

  it("is fully scrambled at progress 0 (except spaces)", () => {
    expect(scrambleText("VASU", 0, glyph)).toBe("####");
  });

  it("locks characters in from the left as progress advances", () => {
    expect(scrambleText("VASU", 0.5, glyph)).toBe("VA##");
  });

  it("never scrambles spaces (word shape stays readable)", () => {
    expect(scrambleText("VASU AGRAWAL", 0.5, glyph)).toBe("VASU A######");
  });

  it("preserves length at every progress", () => {
    for (const p of [0, 0.25, 0.5, 0.75, 1]) {
      expect(scrambleText("VASU AGRAWAL", p, glyph)).toHaveLength(12);
    }
  });

  it("clamps out-of-range progress", () => {
    expect(scrambleText("VASU", -1, glyph)).toBe("####");
    expect(scrambleText("VASU", 2, glyph)).toBe("VASU");
  });
});
