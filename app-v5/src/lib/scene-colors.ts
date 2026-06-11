/**
 * Hex approximations of the OKLCH tokens in styles/tokens.css — three.js
 * cannot parse oklch() strings. tokens.css remains the source of truth;
 * update both together (values converted via OKLCH → sRGB).
 */
export const SCENE_COLORS = {
  background: "#05070d", // --bg-base     oklch(8% 0.02 250)
  accent: "#27b173", // --accent          oklch(58% 0.18 150)
  accentBright: "#4fe3a1", // --accent-bright  oklch(72% 0.20 150)
  nodeGenai: "#27b173", // --node-genai
  nodeMl: "#5c8fc2", // --node-ml          oklch(62% 0.10 230)
  nodeFullstack: "#8d8ed6", // --node-fullstack oklch(66% 0.09 280)
  nodeLang: "#9aa7c7", // --node-lang        oklch(70% 0.06 250)
  nodeInfra: "#b89a6a", // --node-infra       oklch(64% 0.08 60)
  nodeProject: "#c3cce0", // --node-project   oklch(85% 0.03 250)
  nodeFlagship: "#4fe3a1", // --node-flagship oklch(72% 0.20 150)
  particleBase: "#3d4a66", // dim field particle, below bloom threshold
} as const;

import type { SkillCategory } from "@/data/types";

export const CATEGORY_HEX: Record<SkillCategory, string> = {
  genai: SCENE_COLORS.nodeGenai,
  ml: SCENE_COLORS.nodeMl,
  fullstack: SCENE_COLORS.nodeFullstack,
  lang: SCENE_COLORS.nodeLang,
  infra: SCENE_COLORS.nodeInfra,
};
