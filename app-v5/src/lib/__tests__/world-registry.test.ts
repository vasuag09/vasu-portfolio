import { describe, expect, it } from "vitest";

import { getWorldDefinition, WORLD_PROJECT_IDS } from "../world-registry";

const HEX = /^#[0-9a-f]{6}$/i;
const FLAGSHIPS = ["fundlymart", "nm-gpt"] as const;
const PALETTE_ONLY = [
  "ray-serve",
  "insightify",
  "geovision",
  "harness-claude",
  "streamlit-oss",
] as const;

describe("getWorldDefinition", () => {
  it("returns a palette-bearing definition for every dive-able project", () => {
    for (const id of [...FLAGSHIPS, ...PALETTE_ONLY]) {
      const def = getWorldDefinition(id);
      expect(def).not.toBeNull();
      expect(def?.id).toBe(id);
      expect(def?.palette.primary).toMatch(HEX);
      expect(def?.palette.secondary).toMatch(HEX);
      expect(def?.palette.accent).toMatch(HEX);
    }
  });

  it("gives the flagships a lazy motif component", () => {
    for (const id of FLAGSHIPS) {
      expect(getWorldDefinition(id)?.motifComponent).toBeTruthy();
    }
  });

  it("leaves the non-flagship worlds palette-only (no motif yet)", () => {
    for (const id of PALETTE_ONLY) {
      expect(getWorldDefinition(id)?.motifComponent).toBeUndefined();
    }
  });

  it("returns null for no selection or an unknown project", () => {
    expect(getWorldDefinition(null)).toBeNull();
    expect(getWorldDefinition("does-not-exist")).toBeNull();
  });

  it("pre-registers exactly the seven dive-able projects (no silent gaps)", () => {
    expect(WORLD_PROJECT_IDS).toHaveLength(7);
    expect(new Set(WORLD_PROJECT_IDS)).toEqual(
      new Set([...FLAGSHIPS, ...PALETTE_ONLY]),
    );
  });

  it("gives all seven worlds visually distinct primary colours", () => {
    const primaries = [...FLAGSHIPS, ...PALETTE_ONLY].map(
      (id) => getWorldDefinition(id)?.palette.primary,
    );
    expect(new Set(primaries).size).toBe(7);
  });
});
