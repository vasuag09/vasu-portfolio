import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { useBootSequence } from "../useBootSequence";

describe("useBootSequence", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("starts with empty boot sequence and not booted", () => {
    const { result } = renderHook(() => useBootSequence());
    expect(result.current.bootSequence).toEqual([]);
    expect(result.current.isBooted).toBe(false);
  });

  it("adds boot steps over time", () => {
    const { result } = renderHook(() => useBootSequence());

    // After first step delay (0ms - first fires immediately)
    act(() => vi.advanceTimersByTime(0));
    expect(result.current.bootSequence.length).toBeGreaterThanOrEqual(1);
    expect(result.current.bootSequence[0]).toBe("Initializing kernel...");
  });

  it("completes boot after all steps", () => {
    const { result } = renderHook(() => useBootSequence());

    // 5 steps * 400ms = 2000ms total, but first starts at 0ms
    act(() => vi.advanceTimersByTime(2000));

    expect(result.current.bootSequence).toHaveLength(5);
    expect(result.current.isBooted).toBe(true);
    expect(result.current.bootSequence[4]).toBe("System ready.");
  });

  it("skipBoot immediately completes the sequence", () => {
    const { result } = renderHook(() => useBootSequence());

    act(() => result.current.skipBoot());

    expect(result.current.bootSequence).toHaveLength(5);
    expect(result.current.isBooted).toBe(true);
  });
});
