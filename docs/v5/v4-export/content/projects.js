export const projects = [
  {
    id: 0,
    alias: "geovision-lulc",
    title: "GeoVision-LULC (Transformer vs CNN Study)",
    tech: [
      "Remote Sensing",
      "SegFormer",
      "DeepLabV3+",
      "Semantic Segmentation",
      "PyTorch",
      "GeoTIFF",
    ],
    description:
      "Research-grade comparative study of CNN vs Vision Transformer architectures for land use segmentation on Sentinel-2 imagery, achieving higher accuracy with 10x model efficiency.",
    link: "#",
    status: "RESEARCH",
    gif: null,
    tier: "S",
    details: {
      problem:
        "Accurate large-scale LULC mapping requires models that balance spatial precision and computational efficiency.",
      architecture:
        "Comparative pipeline: UNet / DeepLabV3+ vs SegFormer (Transformer). Includes hybrid loss (CE + Dice) and mixed precision training.",
      pipeline:
        "Satellite Patch → Augmentation → Model Training → Segmentation Output → GeoTIFF Export → GIS Integration.",
      decisions:
        "Focused on efficiency-performance tradeoff. Used SegFormer for global context modeling and lightweight inference.",
      failures:
        "RGB-only input limits separability of agriculture vs fallow land. Multispectral integration identified as key improvement.",
      metrics:
        "mIoU: 0.461 (+8.7% over DeepLabV3+), 10.6x smaller model size.",
    },
  },

  {
    id: 1,
    alias: "traffic-analytics",
    title: "Intelligent Traffic Analytics System",
    tech: [
      "Computer Vision",
      "YOLO",
      "Multi-Object Tracking",
      "Kalman Filter",
      "Event-Based Analytics",
    ],
    description:
      "Real-time monocular traffic intelligence system for detection, tracking, counting, and relative speed estimation using event-driven logic.",
    link: "https://github.com/vasuag09/intelligent-traffic-analytics-system",
    status: "LIVE",
    tier: "S",
    details: {
      problem: "Inaccurate vehicle counting and tracking in video streams.",
      architecture: "YOLO detection with Kalman Filter tracking.",
      pipeline: "Frame Capture → Detection → Tracking → Speed Estimation → Reporting.",
      decisions: "Used Kalman filters for robustness to occlusion.",
      failures: "Night-time detection remains a challenge.",
      metrics: "95% accuracy in vehicle counts under daylight.",
    },
  },

  {
    id: 2,
    alias: "insightify",
    title: "Insightify (AI Resume + Job Tracker)",
    tech: ["MERN", "NLP", "Microservices", "Resume Parsing"],
    description:
      "Full-stack AI system for resume optimization and job tracking using NLP pipelines and microservice architecture.",
    link: "https://insightify-react-frontend.onrender.com/",
    status: "LIVE",
    tier: "S",
    details: {
      problem: "Inefficient resume screening and job application tracking.",
      architecture: "MERN stack with Python-based NLP microservice.",
      pipeline: "Resume Upload → NLP Parsing → Data Extraction → Analysis → Dashboard.",
      decisions: "Modularized parser to support multiple file formats.",
      failures: "Complex layouts can sometimes break extraction.",
      metrics: "98% extraction accuracy on standard formats.",
    },
  },

  {
    id: 3,
    alias: "cifar-super-resolution",
    title: "Super-Resolution Decoder Study",
    tech: ["Deep Learning", "Computer Vision", "PSNR", "SSIM"],
    description:
      "Controlled experiment analyzing architectural artifacts in super-resolution, exposing limitations of PSNR/SSIM metrics.",
    link: "https://github.com/vasuag09/cifar10-super-resolution-study",
    status: "RESEARCH",
    tier: "S",
    details: {
      problem: "Standard SR metrics don't capture perceptual quality well.",
      architecture: "Variations of SRCNN and ESRGAN backends.",
      pipeline: "Low-Res Data → SR Model → Image Reconstruction → Metric Evaluation → Human Study.",
      decisions: "Focused on artifacts over blunt PSNR scores.",
      failures: "Oversharpening artifacts can be misinterpreted as detail.",
      metrics: "Disclosed a 15% discrepancy between PSNR and MOS.",
    },
  },

  {
    id: 4,
    alias: "covid-xray",
    title: "COVID-19 X-Ray Vision",
    tech: ["CNN", "DenseNet", "Grad-CAM"],
    description:
      "Medical imaging pipeline with explainability to detect COVID-19 while mitigating dataset bias.",
    link: "https://github.com/vasuag09/Covid-X-Ray",
    status: "RESEARCH",
    tier: "A",
    details: {
      problem: "Bias in medical datasets leading to false COVID-19 diagnoses.",
      architecture: "DenseNet backbone with Grad-CAM heatmaps.",
      pipeline: "X-Ray Image → Preprocessing → Classification → Heatmap Generation → Clinical Review.",
      decisions: "Used Grad-CAM to ensure the model focuses on lung features.",
      failures: "Lateral view images required separate handling.",
      metrics: "92% validation accuracy with improved interpretability.",
    },
  },

  {
    id: 5,
    alias: "insurance-fraud",
    title: "Health Insurance Fraud Detection",
    tech: ["Scikit-Learn", "Imbalanced Data", "Streamlit"],
    description:
      "Fraud detection system with recall-optimized pipeline and real-time scoring interface.",
    link: "https://vasuag09-medical-insurance-fraud-app-jlnqby.streamlit.app/",
    status: "LIVE",
    tier: "A",
    details: {
      problem: "Financial losses due to fraudulent health insurance claims.",
      architecture: "Scikit-Learn ensemble with imbalanced learning techniques.",
      pipeline: "Claim Submission → Feature Extraction → Fraud Scoring → Flagging → Investigation.",
      decisions: "Optimized for Recall to catch more fraud cases.",
      failures: "Newly emerging fraud patterns require continuous retraining.",
      metrics: "0.89 Recall score on hidden test set.",
    },
  },

  {
    id: 6,
    alias: "rag-chatbot",
    title: "PDF RAG Chatbot",
    tech: ["LangChain", "FAISS", "LLM"],
    description:
      "Retrieval-augmented QA system for academic documents with citation-backed responses.",
    link: "#",
    status: "BUILDING",
    tier: "A",
    details: {
      problem: "Difficulty in finding specific information in dense PDF documents.",
      architecture: "LangChain with FAISS vector store and OpenAI backends.",
      pipeline: "PDF Ingestion → Chunking → Embedding → Query → Contextual Retrieval → LLM Response.",
      decisions: "Implemented recursive character splitting for better context.",
      failures: "Highly mathematical equations are still hard to retrieve.",
      metrics: "Reduced hallucinations by 40% compared to base LLM GPT-4.",
    },
  },
  {
    id: 7,
    alias: "network-ids",
    title: "Network Intrusion Detection System (CIC-IDS2017)",
    tech: [
      "Cybersecurity ML",
      "Anomaly Detection",
      "Scikit-Learn",
      "Feature Engineering",
      "Imbalanced Data",
    ],
    description:
      "Machine learning-based intrusion detection system trained on CIC-IDS2017 to identify malicious network traffic patterns, with emphasis on high recall and real-world deployment constraints.",
    link: "#",
    status: "RESEARCH",
    tier: "A",
    details: {
      problem:
        "Detecting malicious network activity in highly imbalanced, high-dimensional traffic data while minimizing false negatives.",
      architecture:
        "Supervised ML pipeline with feature engineering, imbalance handling, and ensemble models for intrusion classification.",
      pipeline:
        "Data Cleaning → Feature Engineering → SMOTE → Model Training → Threshold Optimization → Evaluation.",
      decisions:
        "Optimized for recall to reduce undetected attacks. Used ensemble methods for robustness across attack types.",
      failures:
        "High false positives initially impacted usability. Addressed through threshold tuning and feature refinement.",
      metrics:
        "High recall on attack classes with balanced ROC-AUC across categories.",
    },
  },
];

export const tierOrder = {
  S: 0,
  A: 1,
  B: 2,
};

export const getSortedProjects = () => {
  return [...projects].sort((a, b) => {
    return tierOrder[a.tier] - tierOrder[b.tier];
  });
};

export const getFilteredProjects = (techFilter) => {
  const sorted = getSortedProjects();
  if (!techFilter) return sorted;

  const query = techFilter.toLowerCase();
  return sorted.filter((p) =>
    p.tech.some((t) => t.toLowerCase().includes(query)),
  );
};
