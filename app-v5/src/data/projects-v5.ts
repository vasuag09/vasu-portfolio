import type { Project, ProjectNode } from "./types";
import { getSection } from "./sections-v5";
import { placeInCluster } from "@/lib/scene-layout";

/**
 * Project content inventory — source: docs/v5/CONTEXT.md §3 (verified 2026-06-11).
 * Framing rules baked in:
 *  - Ray PR #62356 is OPEN → always "under review", never "merged" (recheck at /ship).
 *  - Streamlit PR #14390 closed unmerged → honest open-source entry, not a flagship.
 *  - NIDS and Krishnam Ideas are excluded from v5 entirely.
 */

/* ---------------------------------------------------------------- flagships */

const flagships: readonly Project[] = [
  {
    id: "fundlymart",
    title: "FundlyMart WhatsApp Bot",
    tier: "flagship",
    status: "production",
    oneLiner:
      "Production B2B pharmacy ordering AI agent — bulk orders between retailers and distributors, entirely over WhatsApp.",
    narrative:
      "India's pharmacy trade runs on WhatsApp. FundlyMart replaces order-by-message chaos with a structured agent handling catalog search, cart, checkout, and order lifecycle in one conversational flow — deployed at Ardour Analytics for a live distributor network. Act two, in progress: a voice calling bot (inbound + outbound AI relationship manager) on Exotel + Sarvam, built on the same conversation engine — one agentic core, two channels.",
    contribution:
      "End-to-end ownership under the CTO: domain model, 17-tool registry, hybrid intent pipeline (keyword matcher ~70% of traffic at <10ms, LLM fallback via generateObject), migration to a manual per-step tool-loop with registry rebuild between steps, and the test suite.",
    stack: [
      "TypeScript",
      "Node.js",
      "Vercel AI SDK",
      "Hexagonal Architecture",
      "WhatsApp Business API",
      "Zod",
      "AWS ECS Fargate",
      "Exotel + Sarvam (voice, in progress)",
    ],
    metrics: [
      "2,314 tests across 61 files",
      "17-tool LLM agent loop",
      "~70% of traffic resolved at <10ms without an LLM call",
    ],
  },
  {
    id: "nm-gpt",
    title: "NM-GPT (CollegeGPT)",
    tier: "flagship",
    status: "live",
    oneLiner:
      "RAG chatbot for NMIMS MPSTME answering student, faculty, and admin queries grounded in official institutional documents.",
    narrative:
      "Campus knowledge lives scattered across PDFs, portals, and WhatsApp groups. NM-GPT centralizes it behind one chatbot with per-response source citations — adopted by 1,200+ students and demoed live to the Dean after a 7-day MVP sprint.",
    contribution:
      "Full-stack solo build: PyMuPDF ingestion → chunking → Gemini embeddings → FAISS, LangChain retrieval chain, FastAPI /chat and /ingest, Next.js streaming UI over SSE.",
    stack: [
      "Python",
      "LangChain",
      "Gemini 2.5 Pro",
      "FAISS",
      "FastAPI",
      "Next.js 14",
      "PyMuPDF",
      "SSE",
    ],
    metrics: ["1,200+ active student users", "7-day MVP → Dean demo"],
    repoUrl: "https://github.com/vasuag09/CollegeGPT",
  },
  {
    id: "ray-serve",
    title: "Ray Serve — Observability PR",
    tier: "flagship",
    status: "under-review",
    oneLiner:
      "Two production observability metrics for Ray Serve's AsyncioRouter — the decoupled routing primitives used in LLM serving (PR #62356).",
    narrative:
      "Ray Serve's new routing primitives shipped without telemetry. This PR adds a selection→dispatch latency histogram (serve_selection_dispatch_gap_ms) and a dropped-slot counter (serve_selections_released_without_dispatch), with unit tests across normal and failure scenarios — real infrastructure code in a 41k-star AI compute repo, in the LLM serving path.",
    contribution:
      "Implemented both metrics against issue #62163 (maintainer-curated good-first-issue), tagged by deployment and application, with failure-scenario unit tests.",
    stack: ["Python", "Ray Serve", "async", "pytest"],
    externalUrl: "https://github.com/ray-project/ray/pull/62356",
    framingNote:
      'PR is OPEN — frame as "submitted PR under review". Never "merged". Re-check status at /ship.',
  },
  {
    id: "insightify",
    title: "Insightify",
    tier: "flagship",
    status: "archived",
    oneLiner:
      "AI-powered resume scorer and job tracker — ATS keyword matching, section-level compatibility scoring, AI feedback loop.",
    narrative:
      "Full-stack AI product: upload a resume, get an ATS compatibility score with section-level diagnostics and AI-driven rewrite suggestions, then track applications on a dashboard.",
    contribution: "Solo build across React frontend, Node API, and Gemini scoring pipeline.",
    stack: ["React", "Node.js", "Gemini API", "MongoDB"],
    repoUrl: "https://github.com/vasuag09/insightify-react",
  },
  {
    id: "geovision",
    title: "GeoVision-LULC",
    tier: "flagship",
    status: "archived",
    oneLiner:
      "Semantic segmentation for Land Use / Land Cover classification on Sentinel-2 multispectral satellite imagery.",
    narrative:
      "Automates LULC classification across urban, agricultural, water, and forest classes with U-Net / DeepLabV3+ on real satellite data — geospatial AI as a deliberately non-standard depth signal.",
    contribution:
      "Full pipeline: data acquisition, multispectral preprocessing, NDVI/NDWI indices, patch-based training, IoU/mAP evaluation.",
    stack: ["Python", "PyTorch", "U-Net / DeepLabV3+", "Sentinel-2", "Rasterio"],
    repoUrl: "https://github.com/vasuag09/GeoVision-LULC",
  },
] as const;

/* ------------------------------------------- featured + open-source entries */

const featured: readonly Project[] = [
  {
    id: "harness-claude",
    title: "harness-claude",
    tier: "featured",
    status: "active",
    oneLiner:
      "A lean full-SDLC harness for Claude Code: Plan → Implement → Verify → Maintain with subagent orchestration, runtime gates, and cross-session memory.",
    narrative:
      "Agentic orchestration tooling at the meta level — this portfolio is itself being built with it.",
    stack: ["Claude Code", "Subagent orchestration", "Plugin architecture"],
    repoUrl: "https://github.com/vasuag09/harness-claude",
  },
  {
    id: "streamlit-oss",
    title: "Streamlit — Issue #13005",
    tier: "open-source",
    status: "archived",
    oneLiner:
      "Cross-layer fix for Streamlit's layout engine spanning Protobuf, Python, and TypeScript, in a codebase used by 1M+ developers.",
    narrative:
      "Identified and implemented integer/pixel gap parameter support across all three layers (PR #14390). All 209 CI tests passed; the feature was ultimately implemented by the core team.",
    stack: ["Python", "TypeScript", "Protobuf"],
    externalUrl: "https://github.com/streamlit/streamlit/issues/13005",
    framingNote: 'Never claim "merged" — PR #14390 was closed; the core team shipped the feature.',
  },
] as const;

/* ------------------------------------------------------------ archive layer */

const archiveEntry = (
  id: string,
  title: string,
  oneLiner: string,
  stack: readonly string[],
  repoUrl?: string,
): Project => ({
  id,
  title,
  tier: "archive",
  status: "archived",
  oneLiner,
  stack,
  ...(repoUrl ? { repoUrl } : {}),
});

const archive: readonly Project[] = [
  archiveEntry(
    "waste-detection",
    "Urban Waste Dump Detection",
    "Satellite detection of illegal waste dumps (Shirpur) — spectral index engineering, multi-temporal compositing, spatial cross-validation.",
    ["Sentinel-2", "XGBoost", "Random Forest", "Rasterio"],
  ),
  archiveEntry(
    "airline-sentiment",
    "US Airline Sentiment Analysis",
    "TF-IDF, Word2Vec, and fine-tuned BERT sentiment pipeline with Streamlit deployment.",
    ["BERT", "Word2Vec", "Streamlit"],
    "https://github.com/vasuag09/Sentiment-Analysis-Of-US-airlines-tweets",
  ),
  archiveEntry("mnist", "MNIST Digit Classifier", "CNN classifier with a Streamlit app.", [
    "TensorFlow",
    "Streamlit",
  ]),
  archiveEntry(
    "flower-recognition",
    "Flower Recognition",
    "ResNet50 transfer-learning classifier.",
    ["TensorFlow", "ResNet50"],
  ),
  archiveEntry(
    "insurance-fraud",
    "Health Insurance Fraud Detection",
    "Classical ML fraud detection with a Streamlit front end.",
    ["Scikit-Learn", "Streamlit"],
  ),
  archiveEntry(
    "super-resolution",
    "Image Super-Resolution",
    "CNN/SRGAN super-resolution on CIFAR-10, evaluated on PSNR/SSIM.",
    ["PyTorch", "SRGAN"],
    "https://github.com/vasuag09/Super-Resolution-CIFAR-10",
  ),
  archiveEntry(
    "covid-cxr",
    "COVID-19 CXR Classification",
    "DenseNet121 chest X-ray classifier with Grad-CAM explainability.",
    ["TensorFlow", "DenseNet121", "Grad-CAM", "Streamlit"],
    "https://github.com/vasuag09/Covid-X-Ray",
  ),
  archiveEntry(
    "traffic-analytics",
    "Intelligent Traffic Analytics",
    "OpenCV + deep-learning traffic analysis system.",
    ["OpenCV", "Deep Learning"],
    "https://github.com/vasuag09/intelligent-traffic-analytics-system",
  ),
  archiveEntry(
    "airbnb-rio",
    "Airbnb Price Optimization (Rio)",
    "XGBoost regression for listing price optimization.",
    ["XGBoost", "Scikit-Learn"],
  ),
  archiveEntry("churn", "Churn Prediction", "Classical ML churn model.", ["Scikit-Learn"]),
  archiveEntry("credit-risk", "Credit Risk Modeling", "Classical ML credit risk model.", [
    "Scikit-Learn",
  ]),
  archiveEntry(
    "fashion-mnist",
    "Fashion MNIST CNN",
    "CNN classifier for Fashion MNIST.",
    ["TensorFlow"],
  ),
  archiveEntry(
    "medical-insurance-cost",
    "Medical Insurance Cost",
    "Regression model for insurance cost prediction.",
    ["Scikit-Learn"],
  ),
  archiveEntry(
    "real-estate-dl",
    "Real Estate Price (DL)",
    "Deep-learning price regression.",
    ["TensorFlow"],
  ),
  archiveEntry(
    "used-car-price",
    "Used Car Price Prediction",
    "Regression model for used car pricing.",
    ["Scikit-Learn"],
  ),
  archiveEntry(
    "roadtraffic-detection",
    "Road Traffic Detection",
    "Background subtraction and car detection on road footage.",
    ["OpenCV"],
  ),
  archiveEntry(
    "book-writer",
    "Book Writer (Shakespeare GRU)",
    "Character-level GRU text generation.",
    ["TensorFlow", "GRU"],
  ),
  archiveEntry(
    "heartbeat",
    "Heartbeat Sound Classification",
    "Audio classification of heartbeat recordings.",
    ["TensorFlow"],
  ),
  archiveEntry(
    "blog-app",
    "Blog App",
    "React blog with React Query data layer.",
    ["React", "React Query"],
  ),
  archiveEntry(
    "job-tracker",
    "Job Tracker App",
    "Full-stack job application tracker.",
    ["React", "Node.js"],
  ),
  archiveEntry(
    "habit-tracker",
    "Habit Tracker App",
    "React habit tracking app.",
    ["React"],
    "https://github.com/vasuag09/habit-tracker-react",
  ),
] as const;

export const projects: readonly Project[] = [...flagships, ...featured, ...archive];

export function getProject(id: string): Project {
  const project = projects.find((candidate) => candidate.id === id);
  if (!project) {
    throw new Error(`Unknown project id: ${id}`);
  }
  return project;
}

/* ----------------------------------------------------- scene-node derivation */

const NODE_SCALE: Record<Project["tier"], number> = {
  flagship: 1.6,
  featured: 1.25,
  "open-source": 1.0,
  archive: 0.55,
};

const NODE_COLOR: Record<Project["tier"], string> = {
  flagship: "var(--node-flagship)",
  featured: "var(--node-genai)",
  "open-source": "var(--node-project)",
  archive: "var(--node-project)",
};

const projectsAnchor = getSection("projects");

/**
 * Deterministic scene nodes (ADR-3): flagship/featured nodes sit on the inner
 * rings of the projects cluster, archive nodes scatter outward — index order
 * in `projects` already encodes that priority.
 */
export const projectNodes: readonly ProjectNode[] = projects.map(
  (project, index) => ({
    id: project.id,
    sectionId: "projects",
    position: placeInCluster(projectsAnchor, index, projects.length, project.id, {
      radius: 7,
      jitter: 1.1,
      depth: 3.5,
    }),
    color: NODE_COLOR[project.tier],
    scale: NODE_SCALE[project.tier],
    flagship: project.tier === "flagship",
  }),
);
