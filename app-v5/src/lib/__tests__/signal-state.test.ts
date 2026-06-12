import { describe, expect, it } from "vitest";

import {
  createArrivalState,
  detectArrival,
  pulseProgress,
} from "../signal-state";

const CENTERS = [0.05, 0.3, 0.55, 0.78, 0.95];

describe("pulseProgress", () => {
  it("leads scroll progress by the lead amount", () => {
    expect(pulseProgress(0.4, 0.06)).toBeCloseTo(0.46);
  });

  it("clamps at 1 near the end of the journey", () => {
    expect(pulseProgress(0.98, 0.06)).toBe(1);
  });

  it("never returns less than the current progress (signal travels ahead)", () => {
    expect(pulseProgress(0, 0.06)).toBeGreaterThanOrEqual(0);
    expect(pulseProgress(0.5, 0)).toBe(0.5);
  });
});

describe("detectArrival", () => {
  it("fires exactly once when progress enters a chapter band", () => {
    let state = createArrivalState();
    let fired: number | null;

    [state, fired] = detectArrival(state, 0.2, CENTERS);
    expect(fired).toBeNull(); // between chapters

    [state, fired] = detectArrival(state, 0.3, CENTERS);
    expect(fired).toBe(1); // arrived at chapter 1

    [state, fired] = detectArrival(state, 0.305, CENTERS);
    expect(fired).toBeNull(); // still inside the band — no refire
  });

  it("re-arms only after leaving the band", () => {
    let state = createArrivalState();
    let fired: number | null;
    [state, fired] = detectArrival(state, 0.3, CENTERS);
    expect(fired).toBe(1);
    [state, fired] = detectArrival(state, 0.31, CENTERS);
    expect(fired).toBeNull();
    // leave band, come back → fires again
    [state, fired] = detectArrival(state, 0.45, CENTERS);
    expect(fired).toBeNull();
    [state, fired] = detectArrival(state, 0.3, CENTERS);
    expect(fired).toBe(1);
  });

  it("fires the chapter being entered when scrolling backwards too", () => {
    let state = createArrivalState();
    let fired: number | null;
    [state, fired] = detectArrival(state, 0.56, CENTERS);
    expect(fired).toBe(2);
    [state, fired] = detectArrival(state, 0.4, CENTERS);
    expect(fired).toBeNull();
    [state, fired] = detectArrival(state, 0.3, CENTERS);
    expect(fired).toBe(1);
  });

  it("fires the initial chapter on first sample at the top of the page", () => {
    let state = createArrivalState();
    let fired: number | null;
    [state, fired] = detectArrival(state, 0.05, CENTERS);
    expect(fired).toBe(0);
  });
});
