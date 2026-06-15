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
    clip: "fundlymart",
    oneLiner:
      "Production B2B pharmacy ordering AI agent — bulk orders between retailers and distributors, entirely over WhatsApp.",
    problem:
      "India's pharmacy supply chain runs on WhatsApp. Orders arrive as free-text messages, voice notes, and photos. Distributors retype everything by hand, and every retype is an error waiting to ship.",
    narrative:
      "FundlyMart turns that chaos into one structured agent: catalog search, cart, checkout, and order lifecycle in a single conversational flow. Deployed at Ardour Analytics for a live distributor network. Act two is in progress — a voice agent on Exotel + Sarvam handling inbound and outbound calls on the same conversation engine. One agentic core, two channels.",
    contribution:
      "End-to-end ownership under the CTO. Designed the domain model. Built the 17-tool registry. Shipped the hybrid intent pipeline — a keyword matcher resolves ~70% of traffic in under 10ms, the LLM only sees the hard cases. Architected the migration to a manual per-step tool-loop, closed 8 gaps in the migration plan, wrote the test suite.",
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
      "~70% of traffic resolved in <10ms with zero LLM calls",
    ],
  },
  {
    id: "nm-gpt",
    title: "NM-GPT (CollegeGPT)",
    tier: "flagship",
    status: "live",
    clip: "nm-gpt",
    oneLiner:
      "RAG chatbot for NMIMS MPSTME answering student, faculty, and admin queries grounded in official institutional documents.",
    problem:
      "Campus answers live in dozens of PDFs, three portals, and word of mouth. Students ask seniors. Seniors guess. The right answer exists — buried on page 47 of a circular nobody opens.",
    narrative:
      "NM-GPT puts all of it behind one chatbot that cites its sources on every response — zero-hallucination by design, because every answer must point back to an official document. Adopted by 1,200+ students at NMIMS Mumbai. Built as a 7-day MVP sprint that ended in a live demo to the Dean.",
    contribution:
      "Full-stack solo build. PDF ingestion pipeline: PyMuPDF → chunking → Gemini embeddings → FAISS. LangChain retrieval chain with context injection. FastAPI /chat and /ingest endpoints. Next.js streaming UI over SSE.",
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
    metrics: [
      "1,200+ active student users",
      "7-day MVP → live Dean demo",
      "Source citations on every answer",
    ],
    repoUrl: "https://github.com/vasuag09/CollegeGPT",
  },
  {
    id: "ray-serve",
    title: "Ray Serve — Observability PR",
    tier: "flagship",
    status: "under-review",
    oneLiner:
      "Two production observability metrics for Ray Serve's AsyncioRouter — the decoupled routing primitives used in LLM serving (PR #62356).",
    problem:
      "Ray Serve's new decoupled routing primitives shipped blind: no visibility into how long a replica slot waits between selection and dispatch, and no count of slots released without ever dispatching.",
    narrative:
      "This PR adds the missing telemetry: a selection→dispatch latency histogram (serve_selection_dispatch_gap_ms, 11 buckets from 1ms to 5s) and a dropped-slot counter, both tagged by deployment and application. Real infrastructure code in a 41k-star AI compute repo, in the LLM serving path.",
    contribution:
      "Implemented both metrics against issue #62163 — a maintainer-curated good-first-issue in the LLM + Ray Serve area. Unit tests cover normal and failure scenarios, including context exits without dispatch.",
    stack: ["Python", "Ray Serve", "async", "pytest"],
    metrics: [
      "2 production metrics in the LLM serving path",
      "41k-star repo (ray-project/ray)",
      "11-bucket latency histogram, 1ms–5s",
    ],
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
    problem:
      "Most resumes are rejected by ATS keyword filters before a human ever reads them — and applicants get zero feedback about why.",
    narrative:
      "Insightify shows the score before the rejection: upload a resume, get an ATS compatibility score with section-level diagnostics and AI rewrite suggestions, then track every application on one dashboard.",
    contribution:
      "Solo build across the React frontend, Node API, and Gemini scoring pipeline — NLP keyword matching plus section-level scoring logic.",
    stack: ["React", "Node.js", "Gemini API", "MongoDB"],
    metrics: [
      "Section-level ATS scoring",
      "AI rewrite feedback loop",
    ],
    repoUrl: "https://github.com/vasuag09/insightify-react",
  },
  {
    id: "geovision",
    title: "GeoVision-LULC",
    tier: "flagship",
    status: "archived",
    clip: "geovision",
    oneLiner:
      "Semantic segmentation for Land Use / Land Cover classification on Sentinel-2 multispectral satellite imagery.",
    problem:
      "Sentinel-2 photographs the entire planet every five days. Manual land-use mapping takes months per region — the data outruns the humans by orders of magnitude.",
    narrative:
      "GeoVision classifies the land automatically: urban, agricultural, water, and forest zones segmented from raw 13-band multispectral imagery with U-Net and DeepLabV3+. Geospatial AI as a deliberately non-standard depth signal in the portfolio.",
    contribution:
      "Full pipeline, solo: data acquisition, multispectral preprocessing with NDVI/NDWI spectral indices, patch-based training, evaluation on IoU, mAP, and per-class accuracy.",
    stack: ["Python", "PyTorch", "U-Net / DeepLabV3+", "Sentinel-2", "Rasterio"],
    metrics: [
      "13-band Sentinel-2 multispectral input",
      "4 LULC classes, patch-based segmentation",
      "Evaluated on IoU, mAP, per-class accuracy",
    ],
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
