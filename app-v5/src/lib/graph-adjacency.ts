import { edges } from "@/data/skills-graph";

/**
 * Pure adjacency selectors over the skills graph. Built once at module load
 * from the same data the scene renders — the DOM overlay and the canvas glow
 * can never disagree about what is connected to what.
 */

const projectsBySkill = new Map<string, string[]>();
const skillsByProject = new Map<string, string[]>();
const edgeIndicesBySkill = new Map<string, number[]>();
const edgeIndicesByProject = new Map<string, number[]>();

edges.forEach((edge, index) => {
  append(projectsBySkill, edge.skillId, edge.projectId);
  append(skillsByProject, edge.projectId, edge.skillId);
  append(edgeIndicesBySkill, edge.skillId, index);
  append(edgeIndicesByProject, edge.projectId, index);
});

function append<V>(map: Map<string, V[]>, key: string, value: V): void {
  const list = map.get(key);
  if (list) {
    list.push(value);
  } else {
    map.set(key, [value]);
  }
}

export function getProjectsForSkill(skillId: string): readonly string[] {
  return projectsBySkill.get(skillId) ?? [];
}

export function getSkillsForProject(projectId: string): readonly string[] {
  return skillsByProject.get(projectId) ?? [];
}

export interface ActivationInput {
  skillId: string | null;
  projectId: string | null;
}

export interface ActivationSet {
  /** Node ids (skills + projects) that should glow. */
  nodeIds: ReadonlySet<string>;
  /** Indices into the data-edge list (edges[] order) that should glow. */
  edgeIndices: ReadonlySet<number>;
}

const EMPTY: ActivationSet = {
  nodeIds: new Set(),
  edgeIndices: new Set(),
};

/** Resolve a hover/selection into the exact set of glowing nodes and edges. */
export function buildActivationSet(input: ActivationInput): ActivationSet {
  if (!input.skillId && !input.projectId) return EMPTY;

  const nodeIds = new Set<string>();
  const edgeIndices = new Set<number>();

  if (input.skillId) {
    nodeIds.add(input.skillId);
    getProjectsForSkill(input.skillId).forEach((p) => nodeIds.add(p));
    (edgeIndicesBySkill.get(input.skillId) ?? []).forEach((i) =>
      edgeIndices.add(i),
    );
  }
  if (input.projectId) {
    nodeIds.add(input.projectId);
    getSkillsForProject(input.projectId).forEach((s) => nodeIds.add(s));
    (edgeIndicesByProject.get(input.projectId) ?? []).forEach((i) =>
      edgeIndices.add(i),
    );
  }

  return { nodeIds, edgeIndices };
}
