import { describe, expect, it, vi } from "vitest";

import { BLUR_TITLES, blurTitleAt, createTitleController } from "../tab-title";

describe("blurTitleAt", () => {
  it("returns a configured blur title", () => {
    expect(BLUR_TITLES).toContain(blurTitleAt(0));
  });

  it("rotates through the variants and wraps around", () => {
    expect(blurTitleAt(0)).toBe(BLUR_TITLES[0]);
    expect(blurTitleAt(BLUR_TITLES.length)).toBe(BLUR_TITLES[0]);
  });
});

describe("createTitleController", () => {
  it("swaps to a blur title on blur and restores the original on focus", () => {
    const setTitle = vi.fn();
    const controller = createTitleController("Vasu Agrawal — AI Developer", setTitle);

    controller.onBlur();
    expect(setTitle).toHaveBeenLastCalledWith(BLUR_TITLES[0]);

    controller.onFocus();
    expect(setTitle).toHaveBeenLastCalledWith("Vasu Agrawal — AI Developer");
  });

  it("advances the variant on each successive blur", () => {
    const setTitle = vi.fn();
    const controller = createTitleController("orig", setTitle);

    controller.onBlur();
    controller.onFocus();
    controller.onBlur();

    expect(setTitle).toHaveBeenLastCalledWith(BLUR_TITLES[1 % BLUR_TITLES.length]);
  });

  it("never mutates the original title reference", () => {
    const setTitle = vi.fn();
    const original = "orig";
    const controller = createTitleController(original, setTitle);

    controller.onBlur();
    controller.onFocus();

    expect(original).toBe("orig");
  });
});
