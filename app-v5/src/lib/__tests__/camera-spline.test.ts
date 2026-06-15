import { describe, expect, it } from "vitest";
import {
  buildCameraSpline,
  sampleCameraPose,
  progressToSplineParam,
} from "../camera-spline";
import { sections } from "@/data/sections-v5";

const ANCHOR_COUNT = sections.length;

describe("buildCameraSpline + sampleCameraPose (ADR-3)", () => {
  const spline = buildCameraSpline(sections);

  it("passes exactly through every anchor rest-pose at t = i/(n-1)", () => {
    sections.forEach((anchor, i) => {
      const t = i / (ANCHOR_COUNT - 1);
      const pose = sampleCameraPose(spline, t);
      expect(pose.position.x).toBeCloseTo(anchor.cameraPos[0], 5);
      expect(pose.position.y).toBeCloseTo(anchor.cameraPos[1], 5);
      expect(pose.position.z).toBeCloseTo(anchor.cameraPos[2], 5);
      expect(pose.target.x).toBeCloseTo(anchor.cameraTarget[0], 5);
      expect(pose.target.y).toBeCloseTo(anchor.cameraTarget[1], 5);
      expect(pose.target.z).toBeCloseTo(anchor.cameraTarget[2], 5);
    });
  });

  it("is continuous across segment joins (no camera teleports)", () => {
    const epsilon = 1e-4;
    for (let i = 1; i < ANCHOR_COUNT - 1; i += 1) {
      const joint = i / (ANCHOR_COUNT - 1);
      const before = sampleCameraPose(spline, joint - epsilon);
      const after = sampleCameraPose(spline, joint + epsilon);
      expect(before.position.distanceTo(after.position)).toBeLessThan(0.05);
      expect(before.target.distanceTo(after.target)).toBeLessThan(0.05);
    }
  });

  it("clamps out-of-range parameters to the end poses", () => {
    const below = sampleCameraPose(spline, -0.5);
    const above = sampleCameraPose(spline, 1.5);
    expect(below.position.z).toBeCloseTo(sections[0].cameraPos[2], 5);
    expect(above.position.z).toBeCloseTo(
      sections[ANCHOR_COUNT - 1].cameraPos[2],
      5,
    );
  });

  it("allocates no objects when sampling into provided outputs", () => {
    const pose = sampleCameraPose(spline, 0.3);
    const again = sampleCameraPose(spline, 0.7, pose);
    expect(again).toBe(pose); // same instance reused for the useFrame hot path
  });
});

describe("progressToSplineParam (scroll → camera mapping)", () => {
  // Section centers in normalized document-scroll space, e.g. measured from DOM.
  const centers = [0.05, 0.28, 0.5, 0.72, 0.95];

  it("rests the camera exactly at anchor i when scroll sits at center i", () => {
    centers.forEach((center, i) => {
      expect(progressToSplineParam(center, centers)).toBeCloseTo(
        i / (centers.length - 1),
        6,
      );
    });
  });

  it("is monotonically non-decreasing in scroll progress", () => {
    let previous = -Infinity;
    for (let p = 0; p <= 1.0001; p += 0.01) {
      const u = progressToSplineParam(p, centers);
      expect(u).toBeGreaterThanOrEqual(previous);
      previous = u;
    }
  });

  it("clamps before the first center and after the last", () => {
    expect(progressToSplineParam(0, centers)).toBe(0);
    expect(progressToSplineParam(1, centers)).toBe(1);
  });
});
