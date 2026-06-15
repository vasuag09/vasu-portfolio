import { describe, expect, it } from "vitest";
import {
  getProjectsForSkill,
  getSkillsForProject,
  buildActivationSet,
} from "../graph-adjacency";

describe("graph adjacency (skills-graph selectors)", () => {
  it("maps a skill to exactly the projects that use it", () => {
    expect(getProjectsForSkill("rag")).toEqual(["nm-gpt"]);
    expect(getProjectsForSkill("agentic-tool-loop")).toEqual([
      "fundlymart",
      "harness-claude",
    ]);
  });

  it("maps a project to all skills wired to it", () => {
    const skills = getSkillsForProject("fundlymart");
    expect(skills).toContain("vercel-ai-sdk");
    expect(skills).toContain("typescript");
    expect(skills).toContain("hexagonal-architecture");
    expect(skills).not.toContain("langchain"); // NM-GPT's, not FundlyMart's
  });

  it("returns empty arrays for unknown ids instead of throwing", () => {
    expect(getProjectsForSkill("not-a-skill")).toEqual([]);
    expect(getSkillsForProject("not-a-project")).toEqual([]);
  });
});

describe("buildActivationSet (drives canvas glow)", () => {
  it("skill hover activates the skill, its projects, and exactly its edges", () => {
    const set = buildActivationSet({ skillId: "rag", projectId: null });
    expect(set.nodeIds.has("rag")).toBe(true);
    expect(set.nodeIds.has("nm-gpt")).toBe(true);
    expect(set.nodeIds.has("fundlymart")).toBe(false);
    // edge indices index into the data-edge list (edges[] order)
    expect(set.edgeIndices.size).toBeGreaterThan(0);
  });

  it("project hover activates the project and all its skills", () => {
    const set = buildActivationSet({ skillId: null, projectId: "nm-gpt" });
    expect(set.nodeIds.has("nm-gpt")).toBe(true);
    expect(set.nodeIds.has("langchain")).toBe(true);
    expect(set.nodeIds.has("faiss")).toBe(true);
    expect(set.nodeIds.has("vercel-ai-sdk")).toBe(false);
  });

  it("empty input produces an empty set", () => {
    const set = buildActivationSet({ skillId: null, projectId: null });
    expect(set.nodeIds.size).toBe(0);
    expect(set.edgeIndices.size).toBe(0);
  });

  it("edge indices all point at edges touching the active ids", () => {
    const set = buildActivationSet({ skillId: "pytorch", projectId: null });
    expect(set.nodeIds.has("geovision")).toBe(true);
    expect(set.nodeIds.has("super-resolution")).toBe(true);
    expect(set.edgeIndices.size).toBe(2); // pytorch has exactly two edges
  });
});
