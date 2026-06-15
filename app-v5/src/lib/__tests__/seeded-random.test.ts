import { describe, expect, it } from "vitest";
import { hashString, mulberry32, seededRandom } from "../seeded-random";
import { placeInCluster } from "../scene-layout";
import { sections } from "@/data/sections-v5";
import { projectNodes } from "@/data/projects-v5";
import { skills, edges } from "@/data/skills-graph";
import { projects } from "@/data/projects-v5";

describe("seeded-random (ADR-3 determinism)", () => {
  it("hashString is stable for the same input", () => {
    expect(hashString("fundlymart")).toBe(hashString("fundlymart"));
    expect(hashString("fundlymart")).not.toBe(hashString("nm-gpt"));
  });

  it("mulberry32 yields an identical sequence per seed, in [0, 1)", () => {
    const a = mulberry32(42);
    const b = mulberry32(42);
    for (let i = 0; i < 100; i += 1) {
      const value = a();
      expect(value).toBe(b());
      expect(value).toBeGreaterThanOrEqual(0);
      expect(value).toBeLessThan(1);
    }
  });

  it("seededRandom is keyed on id — SSR and client always agree", () => {
    expect(seededRandom("node-x")()).toBe(seededRandom("node-x")());
  });
});

describe("scene layout (hydration safety)", () => {
  const anchor = sections[1];

  it("placeInCluster is pure: same inputs, same position", () => {
    const a = placeInCluster(anchor, 3, 28, "insightify");
    const b = placeInCluster(anchor, 3, 28, "insightify");
    expect(a).toEqual(b);
  });

  it("rounds to 3 decimals so serialization can never drift", () => {
    const [x, y, z] = placeInCluster(anchor, 7, 28, "geovision");
    [x, y, z].forEach((v) => expect(v).toBe(Math.round(v * 1000) / 1000));
  });
});

describe("data integrity", () => {
  it("every edge references an existing skill and project", () => {
    const projectIds = new Set(projects.map((p) => p.id));
    const skillIds = new Set(skills.map((s) => s.id));
    edges.forEach((edge) => {
      expect(projectIds.has(edge.projectId)).toBe(true);
      expect(skillIds.has(edge.skillId)).toBe(true);
      expect(edge.strength).toBeGreaterThan(0);
      expect(edge.strength).toBeLessThanOrEqual(1);
    });
  });

  it("exactly five flagship scene nodes exist", () => {
    expect(projectNodes.filter((n) => n.flagship)).toHaveLength(5);
  });
});
