import { describe, expect, it } from "vitest";

import { sectionIndexForKey } from "../nav-keys";

describe("sectionIndexForKey", () => {
  it("maps 1..N to section indices", () => {
    expect(sectionIndexForKey("1", 5)).toBe(0);
    expect(sectionIndexForKey("3", 5)).toBe(2);
    expect(sectionIndexForKey("5", 5)).toBe(4);
  });

  it("rejects keys beyond the section count", () => {
    expect(sectionIndexForKey("6", 5)).toBeNull();
    expect(sectionIndexForKey("9", 5)).toBeNull();
  });

  it("rejects 0 and non-digits", () => {
    expect(sectionIndexForKey("0", 5)).toBeNull();
    expect(sectionIndexForKey("a", 5)).toBeNull();
    expect(sectionIndexForKey("Enter", 5)).toBeNull();
    expect(sectionIndexForKey("!", 5)).toBeNull();
  });

  it("rejects multi-char numeric strings (numpad sequences)", () => {
    expect(sectionIndexForKey("12", 5)).toBeNull();
  });
});
