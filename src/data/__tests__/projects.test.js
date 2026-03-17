import { describe, it, expect } from "vitest";
import {
  projects,
  tierOrder,
  getSortedProjects,
  getFilteredProjects,
} from "../projects";

describe("projects data", () => {
  it("has 8 projects", () => {
    expect(projects).toHaveLength(8);
  });

  it("every project has required fields", () => {
    projects.forEach((p) => {
      expect(p.id).toBeTypeOf("number");
      expect(p.alias).toBeTypeOf("string");
      expect(p.title).toBeTruthy();
      expect(p.tech).toBeInstanceOf(Array);
      expect(p.tech.length).toBeGreaterThan(0);
      expect(p.description).toBeTruthy();
      expect(p.link).toBeTruthy();
      expect(["LIVE", "RESEARCH", "CODE", "BUILDING"]).toContain(p.status);
      expect(["S", "A", "B"]).toContain(p.tier);
      expect(p.details).toBeTruthy();
      expect(p.details.problem).toBeTruthy();
      expect(p.details.architecture).toBeTruthy();
      expect(p.details.pipeline).toBeTruthy();
      expect(p.details.decisions).toBeTruthy();
      expect(p.details.failures).toBeTruthy();
      expect(p.details.metrics).toBeTruthy();
    });
  });

  it("has unique ids", () => {
    const ids = projects.map((p) => p.id);
    expect(new Set(ids).size).toBe(ids.length);
  });

  it("has unique aliases", () => {
    const aliases = projects.map((p) => p.alias);
    expect(new Set(aliases).size).toBe(aliases.length);
  });

  it("LinkedIn URL is not generic", () => {
    // Ensure LinkedIn links in projects are valid (not just "https://linkedin.com")
    projects.forEach((p) => {
      if (p.link.includes("linkedin")) {
        expect(p.link).not.toBe("https://linkedin.com");
      }
    });
  });
});

describe("getSortedProjects", () => {
  it("sorts S > A > B", () => {
    const sorted = getSortedProjects();
    for (let i = 1; i < sorted.length; i++) {
      expect(tierOrder[sorted[i].tier]).toBeGreaterThanOrEqual(
        tierOrder[sorted[i - 1].tier],
      );
    }
  });
});

describe("getFilteredProjects", () => {
  it("returns all projects when no filter", () => {
    const result = getFilteredProjects(null);
    expect(result).toHaveLength(8);
  });

  it("filters by tech (case insensitive)", () => {
    const result = getFilteredProjects("pytorch");
    expect(result.length).toBeGreaterThan(0);
    result.forEach((p) => {
      expect(
        p.tech.some((t) => t.toLowerCase().includes("pytorch")),
      ).toBe(true);
    });
  });

  it("returns empty for non-existent tech", () => {
    const result = getFilteredProjects("nonexistent-tech-xyz");
    expect(result).toHaveLength(0);
  });
});
