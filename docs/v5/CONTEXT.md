# Vasu Agrawal — Professional Context (v5 content inventory)

> Source: Claude Desktop memory export, 2026-06-11. Items in §8 are unverified.
> GitHub-checkable items are being verified separately (see VERIFICATION notes inline).

---

## 1. Identity & Positioning

**Current role/status:**
4th-year MBA Tech (Computer Engineering) student at Mukesh Patel School of Technology Management & Engineering (NMIMS MPSTME), Mumbai. Concurrently serving as AI Developer Intern at Ardour Analytics Pvt Ltd (fundly.ai), working directly under the CTO on a production B2B AI commerce system.

**One-line positioning:**
Building production LLM systems and agentic AI — from campus chatbots to WhatsApp-native B2B commerce.

**Bio (2–3 sentences):**
I'm an AI Developer based in Mumbai who builds production-grade agentic systems, RAG pipelines, and full-stack AI applications. I've shipped systems used by 1,200+ students, architected a 17-tool LLM agent loop for a live B2B pharmacy ordering platform, and contributed observability code to Ray Serve's production LLM routing layer. My long-term goal is to found an AI startup — the engineering work I do today is deliberate preparation for that.

**Location:** Mumbai, Maharashtra, India

**Looking for:**
- AI/ML Engineering roles at product companies and early-stage startups
- Full-time positions post-graduation (July 2026)
- Specifically interested in agentic systems, LLM infrastructure, and AI product engineering roles
- Long-term: founding an AI startup

---

## 2. Professional Timeline

| Period | Organization | Role | What I did |
|---|---|---|---|
| 2020 | The Lovedale Sr. Sec. School | Student | Class X — 96.8% (CBSE) |
| 2022 | The Lovedale Sr. Sec. School | Student | Class XII — 88% (HSC) |
| 2022–2026 | NMIMS MPSTME, Mumbai | MBA Tech (Computer Engineering) | CGPA 3.71/4. Coursework across ML, deep learning, NLP, DSA, cloud computing, distributed systems, software engineering, and business analytics |
| Mar 2026 | Streamlit (Open Source) | Contributor | Resolved issue #13005 — integer/pixel gap parameter support across proto, Python, and TypeScript layers. All 209 CI tests passed. Navigated full PR workflow independently |
| Apr–Jun 2026 | Ray / Anyscale (Open Source) | Contributor | Contributed observability metrics to `AsyncioRouter` in Ray Serve (PR #62356). Implemented `serve_selection_dispatch_gap_ms` histogram and `serve_selections_released_without_dispatch` counter for the decoupled routing primitives — unit tests across normal and failure scenarios |
| May–Jul 2026 | Ardour Analytics Pvt Ltd (fundly.ai), Mumbai | AI Developer Intern (TIP) — under CTO | Building FundlyMart WhatsApp Bot — B2B pharmacy ordering AI agent in TypeScript, hexagonal architecture, Vercel AI SDK. Designed hybrid intent pipeline (keyword matcher + LLM fallback), architected migration to a full 17-tool LLM tool-loop with manual per-step orchestration, integrated with Spine (Fundly's unified backend). Also conducted voice AI platform evaluation (ElevenLabs vs. Vapi) |

---

## 3. Projects

### 🏆 Flagship Ranking (Top 5)

### [FLAGSHIP #1] FundlyMart WhatsApp Bot
**One-line:** A production B2B pharmacy ordering AI agent that lets pharmacy retailers place bulk orders with distributors over WhatsApp.

**Problem & Outcome:**
India's pharmacy distributors and retailers operate primarily on WhatsApp. FundlyMart replaces fragmented order-by-message chaos with a structured, intelligent agent — handling catalog search, cart management, checkout, and order lifecycle in a single conversational flow. Deployed at Ardour Analytics for a live distributor network. 2,314 tests across 61 test files. Currently being integrated with Spine, Fundly's unified backend system.

**Tech Stack:** TypeScript, Node.js, Vercel AI SDK, Hexagonal Architecture (ports & adapters), WhatsApp Business API (Meta Cloud API), Zod, AWS ECS Fargate

**My Contribution:** End-to-end ownership. Designed the domain model, built the 17-tool registry, implemented the hybrid intent resolution pipeline (keyword matcher handling ~70% of traffic at <10ms, LLM fallback via `generateObject()`), architected the full migration to a manual per-step LLM tool-loop with registry rebuild between steps (to resolve Zod enum validation bugs downstream), closed 8 architectural gaps in the migration plan, and wrote the test suite. Working directly under the CTO.

**Current work (added 2026-06-11):** WhatsApp bot is BUILT; now building a **voice calling
bot** that handles inbound AND outbound calls and acts as an AI relationship manager for
the company. **Stack: Exotel (telephony) + Sarvam (India-focused voice AI)** — NOT
ElevenLabs/Vapi (those were only an earlier evaluation exercise). Architecturally it is
**built on top of the existing WhatsApp conversation engine** — same agentic core, new
voice channel. This extends the FundlyMart case study: text agent (shipped) → voice agent
(in progress) on one shared conversation engine.

**Links:** Not public (proprietary, Ardour Analytics codebase)
**Status:** In progress / production internship work
**Why Flagship #1:** Only project combining production deployment, real users, complex agentic systems design, and a live business context. The most direct signal for "AI Developer" — deployed LLM systems with real tradeoffs, not notebook experiments.

### [FLAGSHIP #2] NM-GPT (CollegeGPT)
**One-line:** A RAG-based AI chatbot for NMIMS MPSTME that answers student, faculty, and admin queries grounded in official institutional documents.

**Problem & Outcome:**
Students waste time hunting for information scattered across PDFs, portals, and WhatsApp groups. NM-GPT centralizes campus knowledge into a single chatbot with zero hallucinations by design. Adopted by 1,200+ students at NMIMS Mumbai. Demoed to the college Dean in a 7-day MVP sprint as a candidate for campus-wide deployment. Source citations appear per response.

**Tech Stack:** Python, LangChain, Gemini 2.5 Pro, Gemini text-embedding-004, FAISS (IndexFlatL2), FastAPI, Next.js 14, PyMuPDF, Server-Sent Events (SSE), Streamlit (MVP)

**My Contribution:** Full-stack solo build. PDF ingestion pipeline (PyMuPDF → chunking → Gemini embeddings → FAISS), LangChain retrieval chain with context injection, FastAPI `/chat` and `/ingest` endpoints, Next.js streaming UI, MVP deployed for the Dean demo in 7 days.

**Links:** GitHub: `github.com/vasuag09/CollegeGPT` (✓ verified 2026-06-11)
**Status:** Live (active users)
**Why Flagship #2:** Quantifiable adoption (1,200+ users), clear narrative arc (7-day MVP → Dean demo → campus deployment), technically rigorous RAG implementation.

### [FLAGSHIP #3] Open Source — Ray Serve (ray-project/ray, PR #62356)
**One-line:** Added two production observability metrics to Ray Serve's AsyncioRouter for the decoupled routing primitives used in LLM serving.

**Problem & Outcome:**
Ray Serve's new decoupled routing primitives (from Phase 1 PR #60865) lacked telemetry — no visibility into selection-dispatch latency or dropped slots. Implemented as part of issue #62163, a maintainer-curated `good-first-issue` in the LLM + Ray Serve area.
- `serve_selection_dispatch_gap_ms` (Histogram) — wall-clock time between `choose_replica()` acquiring a slot and `dispatch()` consuming it. Boundaries: `[1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000]` ms. Tagged by `deployment` and `application`.
- `serve_selections_released_without_dispatch` (Counter) — counts context exits from `choose_replica()` without a matching `dispatch()`.
Unit tests across normal and failure scenarios.

**Tech Stack:** Python, Ray Serve, async programming, pytest
**Links:** PR #62356 on ray-project/ray
**Status:** ⚠ OPEN, NOT merged (✓ verified 2026-06-11). Frame as "submitted PR under review", never "merged contribution". Re-check before launch.
**Why Flagship #3:** Hardest signal to fake — real infrastructure code in one of the highest-profile Python AI compute repos (41k stars), specifically in the LLM serving path.

### [FLAGSHIP #4] Insightify — AI-Powered Resume & Job Tracker
**One-line:** Full-stack AI web app: ATS resume scorer (NLP keyword matching, section-level compatibility scoring), job tracking dashboard, AI-driven feedback loop.

**Tech Stack:** React, Node.js, Gemini API, MongoDB
**Links:** GitHub: `github.com/vasuag09/insightify-react` (✓ verified 2026-06-11)
**Status:** Archived / complete (~9.0/10 portfolio rating)
**Why Flagship #4 (DECIDED 2026-06-11, replaces Streamlit):** AI product with a verified
public repo; reinforces "AI Developer" positioning. Streamlit contribution demoted to the
Open Source section because PR #14390 shows "Closed" on click-through — a flagship slot
must survive recruiter scrutiny.

### Open Source section entry (NOT flagship): Streamlit (Issue #13005)
Honest framing (DECIDED): "Identified and implemented a cross-layer fix for Streamlit's
layout engine spanning Protobuf, Python, and TypeScript (issue #13005, PR #14390) in a
codebase used by 1M+ developers. All 209 CI tests passed; the feature was ultimately
implemented by the core team." Never claim "merged." Sits alongside the Ray Serve PR
("under review") in an Open Source Contributions block.

### [FLAGSHIP #5] GeoVision-LULC
**One-line:** Semantic segmentation pipeline for Land Use Land Cover classification on Sentinel-2 multispectral satellite imagery.

**Problem & Outcome:** Automates LULC classification across urban, agricultural, water, and forest classes using deep learning on real satellite data. Portfolio rating ~8.9/10.

**Tech Stack:** Python, PyTorch, U-Net / DeepLabV3+, Sentinel-2, Rasterio, NDVI/NDWI spectral indices
**My Contribution:** Full pipeline — data acquisition, multispectral preprocessing, patch-based training, architecture implementation, evaluation (IoU, mAP, per-class accuracy).
**Links:** GitHub: `github.com/vasuag09/GeoVision-LULC`
**Status:** Archived
**Why Flagship #5:** Distinguishes the portfolio in a non-standard direction — geospatial AI signals rare technical depth.

### Additional Projects

> EXCLUDED FROM v5 (decided 2026-06-11): **Krishnam Ideas** and **NIDS** are removed from
> the site entirely — strip their skill→project edges when building `skills-graph.ts`.

**harness-claude — Full-SDLC Harness for Claude Code** (ADDED 2026-06-11) — A lean,
full-SDLC harness for Claude Code: Plan → Implement → Verify → Maintain, with subagent
orchestration, runtime gates, and cross-session memory. Installs as an isolated plugin.
GitHub: `github.com/vasuag09/harness-claude` (✓ verified — public, updated 2026-06-10).
Strong "AI Developer" signal: agentic orchestration tooling, meta-level (the portfolio
itself is being built with it — a credible story for the About/case-study copy).
Status: Active. Placement: featured project in the Projects region (non-flagship, but
prominent; candidate for flagship promotion if Vasu wants).

**Insightify — AI-Powered Resume & Job Tracker** (~9.0/10) — ATS resume scorer (NLP keyword matching, section-level ATS scoring), job tracking dashboard, AI feedback loop. Stack: React, Node.js, Gemini API, MongoDB. Status: Archived. (FLAGSHIP #4 — see above.)

**Urban Waste Dump Detection — Sentinel-2, Shirpur** (~8.6/10) — Satellite detection of illegal waste dumps. Spectral index engineering, multi-temporal compositing, XGBoost/RF with spatial cross-validation. Status: Archived.

**US Airline Sentiment Analysis** (~9.3/10) — TF-IDF, Word2Vec, fine-tuned BERT; Streamlit deployment. Status: Archived.

**MNIST Digit Classifier** (~9.2/10) — CNN, Streamlit app. Archived.
**Flower Recognition** (~9.1/10) — ResNet50 transfer learning. Archived.
**Health Insurance Fraud Detection** (~9.0/10) — Scikit-Learn, Streamlit. Archived.
**Image Super-Resolution — CNN/SRGAN** (Jan 2026) — CIFAR-10, PSNR/SSIM. Archived.
**COVID-19 CXR Classification** (Nov–Dec 2025, ~8.2/10) — DenseNet121 + Grad-CAM, Streamlit. Archived.
**Intelligent Traffic Analytics** (~8.7/10) — OpenCV + deep learning. Archived.
**Airbnb Price Optimization (Rio)** (~8.5/10) — XGBoost regression. Archived.

**Other ML/web projects (archived):** Churn Prediction (~8.4), Credit Risk (~8.4), Credit Card Default (~8.3), Fashion MNIST CNN (~8.3), Medical Insurance Cost (~8.2), Real Estate Price DL (~8.1), Used Car Price (~8.1), RoadTraffic BG Subtraction + Car Detection (~7.2–7.5), Book Writer — Shakespeare GRU (~7.4), Heartbeat Sound Classification (~6.4), Blog App with React Query (~8.5), Job Tracker App (~8.7), Habit Tracker App (~8.3).

**Total portfolio: 30+ projects across NLP, computer vision, deep learning, classical ML, remote sensing, full-stack web, and agentic AI.**

---

## 4. Skills (skill → project edge list for the neural graph)

### AI/ML (Models & Frameworks)
| Skill | Projects |
|---|---|
| TensorFlow / Keras | MNIST, Flower Recognition, COVID-19 CXR, Real Estate DL, Fashion MNIST, Heartbeat |
| PyTorch | GeoVision-LULC, SRGAN Super-Resolution |
| Scikit-Learn | NIDS, Health Insurance Fraud, Churn, Credit Risk, Airbnb, Used Car, Medical Insurance |
| XGBoost | NIDS, Urban Waste Detection, Airbnb |
| OpenCV | Intelligent Traffic Analytics, RoadTraffic BG Subtraction |
| ResNet50 / DenseNet121 / Transfer Learning | Flower Recognition, COVID-19 CXR |
| U-Net / DeepLabV3+ | GeoVision-LULC |
| BERT (Hugging Face) | US Airline Sentiment |
| GRU / Sequence Models | Book Writer |
| SMOTE | NIDS |
| Grad-CAM / Explainability | COVID-19 CXR |

### GenAI / LLM
| Skill | Projects |
|---|---|
| Vercel AI SDK | FundlyMart |
| LangChain | NM-GPT |
| RAG | NM-GPT |
| FAISS | NM-GPT |
| Gemini API (generation + embeddings) | NM-GPT, Insightify |
| Prompt Engineering | FundlyMart, NM-GPT, Insightify |
| Agentic Tool-Loop Design | FundlyMart |
| Tool / Function Calling | FundlyMart |
| Zod Schema Validation | FundlyMart, Krishnam Ideas |
| SSE / Streaming | NM-GPT |
| PyMuPDF | NM-GPT |
| Multi-step LLM Orchestration | FundlyMart |
| MCP (Model Context Protocol) | FundlyMart (Spine integration) |

### Full-Stack
| Skill | Projects |
|---|---|
| Next.js 14 (App Router) | NM-GPT, Krishnam Ideas |
| React | NM-GPT, Insightify, Job Tracker, Blog App, vasuai.dev |
| TypeScript | FundlyMart, Krishnam Ideas |
| FastAPI | NM-GPT |
| Node.js | FundlyMart, Insightify |
| Streamlit | MNIST, Health Insurance, COVID-19 CXR, Airline Sentiment, NM-GPT MVP |
| Firebase Auth | Krishnam Ideas |
| Prisma + PostgreSQL | Krishnam Ideas |
| MongoDB | Insightify |
| Cloudinary | Krishnam Ideas |
| Framer Motion | Krishnam Ideas, vasuai.dev |
| shadcn/ui + Radix UI | Krishnam Ideas |
| Zustand | Krishnam Ideas |
| TanStack Table | Krishnam Ideas |
| REST APIs | NM-GPT, FundlyMart, Insightify |

### Languages
Python (all ML/AI, FastAPI, OSS), TypeScript (FundlyMart, Krishnam Ideas), JavaScript (React projects, Streamlit fix), SQL.

### Tools & Infrastructure
AWS ECS Fargate (FundlyMart), WhatsApp Business API (FundlyMart), Git/GitHub (all), Hexagonal Architecture (FundlyMart), Rasterio/geospatial (GeoVision, Waste Detection), Sentinel-2 (GeoVision, Waste Detection), Vitest (Krishnam Ideas, FundlyMart — 2,314 tests), Pytest (Ray contribution), Power BI, Notion, Protobuf/gRPC (Streamlit contribution).

---

## 5. Achievements & Social Proof

### Certifications (LinkedIn-confirmed, June 2026)
| Certification | Issuer | Date |
|---|---|---|
| MathWorks Computer Vision Engineer Professional Certificate | MathWorks (Coursera) | Dec 2025 |
| IBM Machine Learning Professional Certificate | IBM (Coursera) | Nov 2025 |
| Meta React Native | Meta (Coursera) | Sep 2025 |
| IBM Data Science Professional Certificate | IBM (Coursera) | Sep 2025 |
| LLM Engineering: Master AI, LLMs & Agents | Udemy | Aug 2025 |
| Oracle Cloud Infrastructure AI Foundations Associate | Oracle | Aug 2025 |
| React – Complete Guide 2025 (incl. Next.js, Redux) | Udemy | Jul 2025 |
| Machine Learning Specialization | Coursera (DeepLearning.AI) | Jun 2025 |
| Machine Learning A–Z (Python & R) | Udemy | Jun 2025 |
| MySQL Bootcamp | Udemy | Mar 2025 |
| Full-Stack Web Development Bootcamp | Udemy | Mar 2025 |

Also: Kaggle certifications — Python, Pandas, Data Visualization, Intro to ML.

### Publications (Medium — Towards AI)
- "I Caught My LLM Agent Lying Mid-Tool-Call" — production debugging story, LLM tool-loop failure modes (FundlyMart).
- "Gemma 4 12B: The Missing Encoders Are the Point" — architectural analysis.

### Open Source
- Ray/Anyscale — observability metrics for AsyncioRouter in Ray Serve (ray-project/ray, 41k stars), LLM serving path.
- Streamlit — issue #13005 resolved, 209 CI tests passed, repo used by 1M+ developers.

### Campus Impact
- NM-GPT adopted by 1,200+ students at NMIMS Mumbai; live Dean demo.

### Social / Academic
- LinkedIn: 3,203 connections, 3,206 followers (June 2026)
- MBA Tech CGPA 3.71/4 · Class X 96.8% · Class XII 88%
- Active on LinkedIn, Medium (Towards AI), X (@vasuag1040), Substack

---

## 6. Links & Handles

| Platform | Link |
|---|---|
| Portfolio | https://vasuai.dev |
| GitHub | https://github.com/vasuag09 |
| LinkedIn | https://linkedin.com/in/vasu-agrawal20/ |
| Twitter/X | @vasuag1040 |
| HuggingFace | vasu099 |
| Email | vasuagrawal1040@gmail.com |
| Medium (Towards AI) | (confirm exact URL) |
| Substack | (confirm URL) |

---

## 7. Voice & Brand Notes

### Working style
Sprint execution, ship-first. Brutal honesty, zero fluff. Ruthless prioritization. Notion-style structured outputs. Responds to critique that challenges assumptions.

### Content voice
Direct and specific ("Reduced intent classification latency by 40%", not "improved performance"). Practitioner's lens. Holds opinions. Short sentences, active voice, one idea per sentence. Story-driven but data-anchored.
**Banned words:** "excited to announce," "grateful for," "humbled by," "leverage," "synergy," "game-changer," "at the end of the day," "in today's world".

### Content pillars
1. Building in Public (FundlyMart architecture, production failures)
2. AI/ML Technical Insights
3. Student-to-Engineer Gap
4. India AI Ecosystem (WhatsApp-first AI, vernacular, Mumbai startup lens)
5. Founder Mindset

### Target audience
AI/ML hiring managers, CTOs, founders at early-stage startups. Secondary: engineering students building in AI.

### Visual identity notes
Dark navy / electric blue aesthetic. Neural network motifs. Clean, technical, not template. Prefers impact metrics over soft descriptors.

### Positioning shift (v5 goal)
"ML Engineer" (notebooks, model training) → "AI Developer" (production systems, agentic AI, LLM infrastructure). **Lead with FundlyMart and NM-GPT, not the 30 ML experiments.**

---

## 8. Unverified / Needs Confirmation

| Item | Question | Owner |
|---|---|---|
| ~~CGPA~~ | RESOLVED 2026-06-11: display **3.71** | — |
| IBM Generative AI Professional Certificate | In progress / removed? Not on LinkedIn as of June 2026 | Vasu |
| GitHub repo names | ✓ VERIFIED 2026-06-11 (53 public repos). Correct names: `CollegeGPT`, `GeoVision-LULC`, `Network-IDS` (not nids-cic-ids2017), `insightify-react` (not insightify), `Sentiment-Analysis-Of-US-airlines-tweets`, `Covid-X-Ray`, `Super-Resolution-CIFAR-10`, `intelligent-traffic-analytics-system`, `habit-tracker-react`. **No public repo found** for: waste-dump-detection, Fashion MNIST, Heartbeat, Book Writer, Blog App — confirm if private or named differently | Resolved / Vasu |
| Streamlit contribution | ✓ VERIFIED: PR #14390 CLOSED WITHOUT MERGE; issue #13005 still open. Do NOT claim merged. Reframe or drop from flagships | Vasu (framing decision) |
| Ray PR #62356 | ✓ VERIFIED: still OPEN, not merged. Frame as "PR under review". Re-check at launch | Resolved (recheck at /ship) |
| ~~NIDS IEEE paper~~ | MOOT 2026-06-11: NIDS removed from v5 entirely | — |
| ~~Krishnam Ideas~~ | MOOT 2026-06-11: removed from v5 entirely | — |
| FundlyMart | Real transactions vs staging — framing matters. NEW 2026-06-11: WhatsApp bot built; voice calling bot (inbound/outbound, AI relationship manager) in progress — include in case study | Vasu (framing) |
| Medium/Substack URLs | Exact links | Vasu |
| VASU_OS brand system | Palette/typography spec to carry into v5 (defined in v4 codebase — extract during research) | Repo |
| Jio Institute Summer Research Internship | Outcome? | Vasu |
| ~~ElevenLabs vs Vapi evaluation~~ | RESOLVED 2026-06-11: was an evaluation exercise only; the voice bot ships on **Exotel + Sarvam**, built on the WhatsApp conversation engine. Don't feature ElevenLabs/Vapi in the case study | — |
| v4 content inventory | Full export/screenshots before cutover | Task |
| Open to Work status | Display on v5? | Vasu |
