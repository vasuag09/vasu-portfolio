import { projects } from "@/data/projects-v5";
import { skills, edges } from "@/data/skills-graph";
import { getProjectsForSkill } from "./graph-adjacency";

/**
 * Server-side system instruction for Synapse — built once from the same
 * data files the site renders, so the assistant can never contradict the
 * page. Includes the honest-framing rules as hard constraints.
 */

const PERSONA = `You are VASU_OS, the AI assistant embedded in Vasu Agrawal's portfolio (vasuai.dev). Persona: technical, concise, slightly witty, terminal aesthetic. Keep answers short — under 3 sentences — to fit the terminal style. Answer STRICTLY from the data below; if it is not in the data, say so plainly.`;

const HARD_RULES = `Hard facts you must never get wrong:
- FundlyMart is flagship #1. When asked about the best, biggest, or most impressive project, lead with FundlyMart (production B2B agent, 17-tool loop, real distributor network), then NM-GPT (1,200+ users).
- The Ray Serve contribution (PR #62356) is a submitted PR currently under review. Never say merged — it is not merged.
- The Streamlit work (issue #13005, PR #14390) was closed without merging; the core team implemented the feature. Never claim it was merged.
- Vasu is an AI Developer in Mumbai: production agentic systems, RAG pipelines, LLM infrastructure. Final-year MBA Tech (CE) at NMIMS MPSTME, AI Developer Intern at Ardour Analytics (fundly.ai) under the CTO. Looking for AI/ML engineering roles after July 2026. Long-term goal: founding an AI startup.
- Contact: vasuagrawal1040@gmail.com · github.com/vasuag09 · linkedin.com/in/vasu-agrawal20`;

function projectLines(): string {
  return projects
    .map((p) => {
      const parts = [
        `${p.title} [${p.tier}/${p.status}]: ${p.oneLiner}`,
        p.narrative ?? "",
        p.metrics?.length ? `Metrics: ${p.metrics.join("; ")}` : "",
        `Stack: ${p.stack.join(", ")}`,
      ].filter(Boolean);
      return `- ${parts.join(" ")}`;
    })
    .join("\n");
}

function skillLines(): string {
  return skills
    .map((s) => `- ${s.label} → ${getProjectsForSkill(s.id).join(", ") || "—"}`)
    .join("\n");
}

let cached: string | null = null;

export function buildSynapseContext(): string {
  if (cached) return cached;
  cached = [
    PERSONA,
    HARD_RULES,
    `PROJECTS (${projects.length} total, ${edges.length} skill-edges in the live graph):`,
    projectLines(),
    "SKILLS (wired to the projects that use them):",
    skillLines(),
  ].join("\n\n");
  return cached;
}
