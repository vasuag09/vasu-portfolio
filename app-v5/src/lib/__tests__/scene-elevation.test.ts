import { describe, expect, it } from "vitest";

import {
  curveSegmentPoints,
  edgeLengthFade,
  pickLabelNodes,
} from "../scene-elevation";

describe("curveSegmentPoints", () => {
  const from: [number, number, number] = [0, 0, 0];
  const to: [number, number, number] = [10, 0, 0];

  it("starts and ends exactly at the endpoints (graph stays truthful)", () => {
    const pts = curveSegmentPoints(from, to, 8);
    expect(pts[0]).toEqual(from);
    expect(pts[pts.length - 1]).toEqual(to);
  });

  it("returns steps+1 points", () => {
    expect(curveSegmentPoints(from, to, 8)).toHaveLength(9);
  });

  it("bows away from the straight line at the midpoint", () => {
    const pts = curveSegmentPoints(from, to, 8);
    const mid = pts[4];
    const offStraight = Math.hypot(mid[1], mid[2]);
    expect(offStraight).toBeGreaterThan(0.3);
  });

  it("is deterministic for identical endpoints (SSR-safe)", () => {
    expect(curveSegmentPoints(from, to, 8)).toEqual(curveSegmentPoints(from, to, 8));
  });
});

describe("edgeLengthFade", () => {
  it("keeps short intra-cluster edges fully visible", () => {
    expect(edgeLengthFade(4)).toBe(1);
  });

  it("fades long cross-section edges to a whisper, never zero", () => {
    expect(edgeLengthFade(30)).toBeCloseTo(0.12, 5);
  });

  it("is monotonically decreasing", () => {
    expect(edgeLengthFade(8)).toBeGreaterThan(edgeLengthFade(14));
    expect(edgeLengthFade(14)).toBeGreaterThan(edgeLengthFade(22));
  });
});

describe("pickLabelNodes", () => {
  it("labels every flagship project", () => {
    const labels = pickLabelNodes();
    const flagshipLabels = labels.filter((l) => l.kind === "project");
    expect(flagshipLabels.length).toBeGreaterThanOrEqual(5);
  });

  it("keeps the set small enough to read (≤14 labels)", () => {
    expect(pickLabelNodes().length).toBeLessThanOrEqual(14);
  });

  it("every label has a position and non-empty text", () => {
    for (const l of pickLabelNodes()) {
      expect(l.text.length).toBeGreaterThan(0);
      expect(l.position).toHaveLength(3);
    }
  });
});
