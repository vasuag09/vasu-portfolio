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
  it("returns a full definition for each flagship world", () => {
    for (const id of FLAGSHIPS) {
      const def = getWorldDefinition(id);
      expect(def).not.toBeNull();
      expect(def?.id).toBe(id);
      expect(def?.palette.primary).toMatch(HEX);
      expect(def?.palette.secondary).toMatch(HEX);
      expect(def?.palette.accent).toMatch(HEX);
    }
  });

  it("returns null for projects without a bespoke world yet (palette-only fallback)", () => {
    for (const id of PALETTE_ONLY) {
      expect(getWorldDefinition(id)).toBeNull();
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

  it("gives the two flagships visually distinct palettes", () => {
    const fundly = getWorldDefinition("fundlymart");
    const nmgpt = getWorldDefinition("nm-gpt");
    expect(fundly?.palette.primary).not.toBe(nmgpt?.palette.primary);
  });
});
