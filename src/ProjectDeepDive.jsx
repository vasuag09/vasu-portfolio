// ProjectDeepDive.jsx
import React from "react";

const ProjectDeepDive = ({ project }) => {
  const insightifyDeepDive = {
    id: "insightify",
    title: "Insightify — AI-Powered Resume & Job Tracker",
    summary:
      "Full-stack MERN product. Resume parsing + AI recommendations + job tracking.",
    repo: "https://github.com/vasuag09/insightify",
    live: "https://insightify-react-frontend.onrender.com/",
    sections: [
      {
        h: "Problem",
        body: "Recruiters and job seekers need fast, structured resume insights. Built an app to parse resumes, surface skills, score matches and track applications.",
      },
      {
        h: "Architecture",
        body: "Frontend: React + Redux. Backend: Node/Express + MongoDB. Parser: Python microservice (spaCy + regex fallbacks). Queue: Redis + BullMQ. Deployment: Docker + Render.",
      },
      {
        h: "Key engineering decisions",
        body: "Decoupled parsing using a job queue to avoid blocking requests. Exposed a websocket progress channel so UI renders streaming updates. Normalized skill taxonomy using fuzzy matching and embedding similarity.",
      },
      {
        h: "Failure modes & fixes",
        body: "PDF OCR fallback failed on scanned resumes — added Tesseract fallback and improved text cleanup. Concurrent uploads caused high memory; limited worker pool and implemented file size caps.",
      },
    ],
    metrics: [
      "Parsing latency (p95) 1.2s",
      "Concurrent safe workers: 4",
      "Resume extraction accuracy: 93% (entity-level)",
    ],
  };

  const covidCxrDeepDive = {
    id: "covid-cxr",
    title: "COVID-19 X-Ray Vision",
    summary:
      "Medical imaging pipeline with DenseNet121, CLAHE preprocessing and Grad-CAM explainability.",
    repo: "https://github.com/vasuag09/Covid-X-Ray",
    live: null,
    sections: [
      {
        h: "Problem",
        body: "Reliable COVID detection on heterogeneous chest X-ray sources with explainability for clinicians.",
      },
      {
        h: "Dataset & preprocessing",
        body: "Merged multiple public datasets, standardized to 512px with aspect-preserving resize, applied CLAHE and consistent normalization. Balanced classes with oversampling + augmentation.",
      },
      {
        h: "Modeling",
        body: "DenseNet121 pretrained backbone fine-tuned with class weights and cyclic LR. Focal loss used on noisy labels to stabilize training.",
      },
      {
        h: "Explainability",
        body: "Grad-CAM on final conv layer with channel normalization and gaussian smoothing. Implemented automated per-case attention sanity checks.",
      },
    ],
    metrics: [
      "Cross-source AUC 0.93",
      "Grad-CAM alignment > 85% (manual audit)",
      "Inference: 70ms/image on GPU",
    ],
  };

  const pdfRagDeepDive = {
    id: "pdf-rag",
    title: "PDF RAG Chatbot",
    summary:
      "Document-level QA for academic notes using semantic chunking, embeddings, and FAISS.",
    repo: null,
    live: null,
    sections: [
      {
        h: "Problem",
        body: "Reliable retrieval + grounded LLM responses over multi-page PDFs with citations.",
      },
      {
        h: "Design",
        body: "Semantic chunking (256 tokens, 50 overlap), embedding store (OpenAI/HF), FAISS HNSW index, LLM prompt templates with citation grounding.",
      },
      {
        h: "Production concerns",
        body: "Chunk freshness, vector store backups, index persistence. Implemented deterministic reranking and hard-citation markers inserted in LLM context.",
      },
    ],
    metrics: [
      "Retrieval recall ↑ 22%",
      "Median query latency ↓ 40% after index tuning",
    ],
  };

  const airlineSentimentDeepDive = {
    id: "airline-sentiment",
    title: "US Airline Sentiment",
    summary:
      "High-precision sentiment pipeline on 14k tweets with a Streamlit dashboard.",
    repo: null,
    live: "https://sentiment-analysis-of-us-airlines-tweets-egszqbuj37bydzk8cypqq.streamlit.app/",
    sections: [
      {
        h: "Problem",
        body: "Real-time sentiment classification for airline tweets with interpretable features.",
      },
      {
        h: "Pipeline",
        body: "Preproc: emoji mapping, URL removal, light stemming. Feature extraction: TF-IDF + custom lexicons. Model: Logistic Regression with class weighting and calibrated probabilities.",
      },
      {
        h: "Deployment",
        body: "Streamlit UI with real-time inference and batch scoring endpoints for CSV uploads.",
      },
    ],
    metrics: [
      "Accuracy 0.87",
      "F1 (positive) 0.82",
      "Streamlit latency ≤ 150ms",
    ],
  };

  const flowerRecognitionDeepDive = {
    id: "flower-recognition",
    title: "Flower Recognition (ResNet50 Transfer Learning)",
    summary:
      "Fine-tuned ResNet50 on small dataset with augmentation and structured evaluation.",
    repo: "https://github.com/vasuag09/Flower-Recognition",
    live: null,
    sections: [
      {
        h: "Problem",
        body: "High accuracy multi-class classification on limited labeled data.",
      },
      {
        h: "Approach",
        body: "Transfer learning, class-aware augmentation, mixup and five-fold cross-validation. Post-training: confusion matrix-driven per-class reweighting for underperforming classes.",
      },
      {
        h: "Results",
        body: "95%+ average accuracy; per-class F1 used to guide data collection for worst-performing classes.",
      },
    ],
    metrics: ["Top-1 accuracy 95.1%", "Per-class F1 min: 0.88"],
  };

  const roadTrafficDeepDive = {
    id: "road-traffic",
    title: "Road Traffic Detection — Background Subtraction & Morphology",
    summary:
      "MATLAB-based foreground segmentation and region analysis for static cameras.",
    repo: null,
    live: null,
    sections: [
      {
        h: "Problem",
        body: "Detect vehicles robustly in static-camera footage with illumination changes and noise.",
      },
      {
        h: "Methods",
        body: "Running median background, CLAHE normalization, morphological open/close, region props-based filtering and tracking by centroid linking.",
      },
      {
        h: "Engineering notes",
        body: "Tuned median window and structuring element sizes per-camera; measured performance by true-pixel ratio and region stability across frames.",
      },
    ],
    metrics: [
      "Frame-level segmentation precision ↑ 14% after tuning",
      "Processing: ~30 FPS on optimized MATLAB pipeline",
    ],
  };

  return (
    <div className="max-w-4xl mx-auto p-6 bg-slate-900 rounded border border-slate-800">
      <header className="mb-6">
        <h1 className="text-3xl font-bold text-white">{project.title}</h1>
        <div className="text-sm text-slate-400 mt-1">{project.summary}</div>
        <div className="flex gap-3 mt-3">
          {project.live && (
            <a
              href={project.live}
              target="_blank"
              rel="noreferrer"
              className="text-emerald-400"
            >
              Live
            </a>
          )}
          {project.repo && (
            <a
              href={project.repo}
              target="_blank"
              rel="noreferrer"
              className="text-slate-300"
            >
              Code
            </a>
          )}
        </div>
      </header>

      {project.sections.map((s, i) => (
        <section key={i} className="mb-6">
          <h3 className="text-emerald-400 font-mono text-sm mb-2">// {s.h}</h3>
          <div className="prose prose-invert text-slate-300">{s.body}</div>
        </section>
      ))}

      {project.metrics && (
        <div className="mt-4 bg-slate-800 p-4 rounded border border-slate-700">
          <h4 className="text-sm font-mono text-emerald-400 mb-2">
            Key Metrics
          </h4>
          <div className="flex gap-4 flex-wrap">
            {project.metrics.map((m, idx) => (
              <div
                key={idx}
                className="bg-slate-900/30 px-3 py-1 rounded text-sm text-white"
              >
                {m}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default ProjectDeepDive;
