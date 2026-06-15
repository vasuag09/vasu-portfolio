import type { Edge, SkillCategory, SkillNode } from "./types";
import { getSection } from "./sections-v5";
import { projects } from "./projects-v5";
import { placeInCluster } from "@/lib/scene-layout";

/**
 * Skill → project edge list — source: docs/v5/CONTEXT.md §4.
 * NIDS and Krishnam Ideas edges are stripped (projects removed from v5);
 * skills whose only edges pointed there (e.g. SMOTE) are dropped entirely.
 * Every edge is validated against the project inventory at module load, so a
 * typo fails the build instead of silently rendering a dangling connection.
 */

interface SkillDef {
  id: string;
  label: string;
  category: SkillCategory;
  /** projectId → edge strength (0–1): how central the skill is to the project. */
  uses: Readonly<Record<string, number>>;
}

const skillDefs: readonly SkillDef[] = [
  /* ---- GenAI / LLM (the core story — accent-colored) ---- */
  { id: "vercel-ai-sdk", label: "Vercel AI SDK", category: "genai", uses: { fundlymart: 1 } },
  { id: "agentic-tool-loop", label: "Agentic Tool-Loop Design", category: "genai", uses: { fundlymart: 1, "harness-claude": 0.8 } },
  { id: "function-calling", label: "Tool / Function Calling", category: "genai", uses: { fundlymart: 1 } },
  { id: "multi-step-orchestration", label: "Multi-step LLM Orchestration", category: "genai", uses: { fundlymart: 1, "harness-claude": 0.9 } },
  { id: "mcp", label: "MCP", category: "genai", uses: { fundlymart: 0.7, "harness-claude": 0.9 } },
  { id: "voice-ai", label: "Voice AI (Exotel + Sarvam)", category: "genai", uses: { fundlymart: 0.7 } },
  { id: "langchain", label: "LangChain", category: "genai", uses: { "nm-gpt": 1 } },
  { id: "rag", label: "RAG", category: "genai", uses: { "nm-gpt": 1 } },
  { id: "faiss", label: "FAISS", category: "genai", uses: { "nm-gpt": 1 } },
  { id: "gemini-api", label: "Gemini API", category: "genai", uses: { "nm-gpt": 1, insightify: 0.8 } },
  { id: "prompt-engineering", label: "Prompt Engineering", category: "genai", uses: { fundlymart: 0.9, "nm-gpt": 0.9, insightify: 0.7 } },
  { id: "sse-streaming", label: "SSE / Streaming", category: "genai", uses: { "nm-gpt": 0.8 } },
  { id: "pymupdf", label: "PyMuPDF Ingestion", category: "genai", uses: { "nm-gpt": 0.7 } },

  /* ---- AI/ML models & frameworks ---- */
  { id: "tensorflow", label: "TensorFlow / Keras", category: "ml", uses: { mnist: 1, "flower-recognition": 1, "covid-cxr": 1, "real-estate-dl": 1, "fashion-mnist": 1, heartbeat: 1, "book-writer": 0.8 } },
  { id: "pytorch", label: "PyTorch", category: "ml", uses: { geovision: 1, "super-resolution": 1 } },
  { id: "scikit-learn", label: "Scikit-Learn", category: "ml", uses: { "insurance-fraud": 1, churn: 1, "credit-risk": 1, "airbnb-rio": 0.8, "used-car-price": 1, "medical-insurance-cost": 1 } },
  { id: "xgboost", label: "XGBoost", category: "ml", uses: { "waste-detection": 1, "airbnb-rio": 1 } },
  { id: "opencv", label: "OpenCV", category: "ml", uses: { "traffic-analytics": 1, "roadtraffic-detection": 1 } },
  { id: "transfer-learning", label: "Transfer Learning", category: "ml", uses: { "flower-recognition": 1, "covid-cxr": 1 } },
  { id: "semantic-segmentation", label: "U-Net / DeepLabV3+", category: "ml", uses: { geovision: 1 } },
  { id: "bert", label: "BERT (Hugging Face)", category: "ml", uses: { "airline-sentiment": 1 } },
  { id: "sequence-models", label: "GRU / Sequence Models", category: "ml", uses: { "book-writer": 1 } },
  { id: "explainability", label: "Grad-CAM / Explainability", category: "ml", uses: { "covid-cxr": 1 } },

  /* ---- Full-stack ---- */
  { id: "nextjs", label: "Next.js (App Router)", category: "fullstack", uses: { "nm-gpt": 0.9 } },
  { id: "react", label: "React", category: "fullstack", uses: { "nm-gpt": 0.8, insightify: 1, "job-tracker": 1, "blog-app": 1, "habit-tracker": 1 } },
  { id: "fastapi", label: "FastAPI", category: "fullstack", uses: { "nm-gpt": 1 } },
  { id: "nodejs", label: "Node.js", category: "fullstack", uses: { fundlymart: 1, insightify: 0.9, "job-tracker": 0.7 } },
  { id: "streamlit-fw", label: "Streamlit", category: "fullstack", uses: { mnist: 0.7, "insurance-fraud": 0.7, "covid-cxr": 0.7, "airline-sentiment": 0.7, "nm-gpt": 0.5 } },
  { id: "zod", label: "Zod Validation", category: "fullstack", uses: { fundlymart: 0.9 } },
  { id: "mongodb", label: "MongoDB", category: "fullstack", uses: { insightify: 0.8 } },
  { id: "rest-apis", label: "REST APIs", category: "fullstack", uses: { fundlymart: 0.7, "nm-gpt": 0.7, insightify: 0.7 } },

  /* ---- Languages ---- */
  { id: "python", label: "Python", category: "lang", uses: { "nm-gpt": 1, geovision: 1, "ray-serve": 1, "streamlit-oss": 0.8 } },
  { id: "typescript", label: "TypeScript", category: "lang", uses: { fundlymart: 1, "streamlit-oss": 0.7 } },
  { id: "javascript", label: "JavaScript", category: "lang", uses: { insightify: 0.8, "blog-app": 0.8, "job-tracker": 0.8 } },

  /* ---- Tools & infrastructure ---- */
  { id: "aws-ecs", label: "AWS ECS Fargate", category: "infra", uses: { fundlymart: 0.8 } },
  { id: "whatsapp-api", label: "WhatsApp Business API", category: "infra", uses: { fundlymart: 1 } },
  { id: "hexagonal-architecture", label: "Hexagonal Architecture", category: "infra", uses: { fundlymart: 1 } },
  { id: "geospatial", label: "Rasterio / Geospatial", category: "infra", uses: { geovision: 1, "waste-detection": 1 } },
  { id: "sentinel-2", label: "Sentinel-2", category: "infra", uses: { geovision: 1, "waste-detection": 1 } },
  { id: "vitest", label: "Vitest", category: "infra", uses: { fundlymart: 0.9 } },
  { id: "pytest", label: "Pytest", category: "infra", uses: { "ray-serve": 0.9 } },
  { id: "protobuf", label: "Protobuf / gRPC", category: "infra", uses: { "streamlit-oss": 0.8 } },
] as const;

/* -------------------------------------------------------------- validation */

const knownProjectIds = new Set(projects.map((project) => project.id));

for (const def of skillDefs) {
  for (const projectId of Object.keys(def.uses)) {
    if (!knownProjectIds.has(projectId)) {
      throw new Error(
        `skills-graph: skill "${def.id}" references unknown project "${projectId}"`,
      );
    }
  }
}

/* -------------------------------------------------------------- derivation */

const CATEGORY_COLOR: Record<SkillCategory, string> = {
  genai: "var(--node-genai)",
  ml: "var(--node-ml)",
  fullstack: "var(--node-fullstack)",
  lang: "var(--node-lang)",
  infra: "var(--node-infra)",
};

const skillsAnchor = getSection("skills");

export const skills: readonly SkillNode[] = skillDefs.map((def, index) => ({
  id: def.id,
  label: def.label,
  position: placeInCluster(skillsAnchor, index, skillDefs.length, def.id, {
    radius: 8,
    jitter: 1.0,
    depth: 4,
  }),
  category: def.category,
  color: CATEGORY_COLOR[def.category],
}));

export const edges: readonly Edge[] = skillDefs.flatMap((def) =>
  Object.entries(def.uses).map(([projectId, strength]) => ({
    skillId: def.id,
    projectId,
    strength,
  })),
);

export function getSkill(id: string): SkillNode {
  const skill = skills.find((candidate) => candidate.id === id);
  if (!skill) {
    throw new Error(`Unknown skill id: ${id}`);
  }
  return skill;
}
