/**
 * Scene-graph data model — schema per docs/v5/ADR.md (ADR-3).
 * All positions are plain tuples (no three.js dependency in the data layer);
 * Phase 1 builds the camera spline from these anchors.
 */

export type Vec3 = [number, number, number];

export type SectionId =
  | "hero"
  | "projects"
  | "skills"
  | "about"
  | "contact";

export interface SectionAnchor {
  id: SectionId;
  label: string;
  cameraPos: Vec3;
  cameraTarget: Vec3;
}

export type ProjectStatus =
  | "production"
  | "live"
  | "under-review"
  | "active"
  | "archived";

export type ProjectTier = "flagship" | "featured" | "open-source" | "archive";

export interface ProjectNode {
  id: string;
  sectionId: SectionId;
  position: Vec3; // derived: cluster placement + seeded jitter
  color: string; // OKLCH token reference
  scale: number;
  flagship: boolean;
}

/** Case-study content (Phase 5 consumes this; scene nodes derive from it). */
export interface Project {
  id: string;
  title: string;
  tier: ProjectTier;
  status: ProjectStatus;
  oneLiner: string;
  /** The pain that existed before the project (case-study hero block). */
  problem?: string;
  /** What the project does about it — the outcome story. */
  narrative?: string;
  contribution?: string;
  /** Veo clip name in /public/veo (see VEO-BRIEF.md naming convention). */
  clip?: string;
  stack: readonly string[];
  /** Verified metrics only — every claim must survive its own hyperlink. */
  metrics?: readonly string[];
  repoUrl?: string;
  externalUrl?: string;
  /** Honest framing constraints (e.g. Ray PR = "under review", never "merged"). */
  framingNote?: string;
}

export type SkillCategory = "genai" | "ml" | "fullstack" | "lang" | "infra";

export interface SkillNode {
  id: string;
  label: string;
  position: Vec3;
  category: SkillCategory;
  color: string;
}

export interface Edge {
  skillId: string;
  projectId: string;
  /** 0–1: drives connection brightness/width in the graph. */
  strength: number;
}
