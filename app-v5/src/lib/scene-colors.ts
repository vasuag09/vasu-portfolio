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

/**
 * Per-project World palettes (ADR-10) — the moods the dive cross-fade blends
 * the particle field toward. Only bespoke worlds live here; projects absent
 * from this map fall back to the base scene palette (world-registry returns
 * null). Hex like SCENE_COLORS — three.js can't read oklch(); these are
 * world-only colours, not tokens.css tokens.
 */
export interface WorldPalette {
  primary: string;
  secondary: string;
  accent: string;
}

export const PROJECT_WORLD_COLORS: Record<string, WorldPalette> = {
  // FundlyMart — WhatsApp-native B2B pharmacy commerce: transactional green
  // warmed by a commerce amber, the "money moving" feel.
  fundlymart: {
    primary: "#19a86a",
    secondary: "#e0a23c",
    accent: "#5fe6a6",
  },
  // NM-GPT — RAG over institutional documents: a cool, academic
  // knowledge-blue lifting into citation glow.
  "nm-gpt": {
    primary: "#3f7fd0",
    secondary: "#8b6fd6",
    accent: "#7fb4ff",
  },
};
