import React, { useState, useEffect, useRef } from 'react';
import { 
  Terminal, 
  Cpu, 
  Code2, 
  Briefcase, 
  User, 
  Mail, 
  Github, 
  Linkedin, 
  ExternalLink, 
  ChevronRight,
  Database,
  Brain,
  Layout,
  Award,
  Loader,
  Download,
  Play,
  X,
  Eye,
  BookOpen,
  Filter,
  ArrowLeft,
  GitBranch,
  AlertTriangle,
  CheckCircle2,
  Target,
  FileText,
  Hash,
  Command,
  Zap,
  MessageSquare,
  Activity,
  Monitor,
  Map,
  Server,
  Layers,
  Box,
  RefreshCw,
  Save,
  Globe,
  CheckCircle,
  Sparkles
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const Portfolio = () => {
  const [activeTab, setActiveTab] = useState('overview');
  const [bootSequence, setBootSequence] = useState([]);
  const [isBooted, setIsBooted] = useState(false);
  
  // State for Modal Preview (GIFs)
  const [previewProject, setPreviewProject] = useState(null);
  
  // State for Deep Dive Page
  const [expandedProject, setExpandedProject] = useState(null);
  
  // State for Reading Log
  const [readingLog, setReadingLog] = useState(null);
  
  // State for Terminal Mode
  const [isTerminalOpen, setIsTerminalOpen] = useState(false);
  const [terminalInput, setTerminalInput] = useState('');
  const [terminalHistory, setTerminalHistory] = useState([
    { type: 'system', content: 'Welcome to VASU_OS v4.1.0 (AI Enabled)' },
    { type: 'system', content: 'Type "./help" for commands or just ask me anything about Vasu.' }
  ]);
  const [isAiProcessing, setIsAiProcessing] = useState(false);
  const terminalEndRef = useRef(null);
  
  // State for Retro Mode
  const [isRetro, setIsRetro] = useState(false);
  
  // State for Guided Tour
  const [tourStep, setTourStep] = useState(0);
  
  // State for Pipeline Animation
  const [pipelineStep, setPipelineStep] = useState(0);

  const [filterTech, setFilterTech] = useState(null);

  // Ref to track if boot sequence has already run
  const bootRan = useRef(false);

  // --- API KEY CONFIGURATION ---
  // INSTRUCTIONS: Replace the empty string below with your actual Gemini API Key.
  // In production, use process.env.REACT_APP_GEMINI_API_KEY or import.meta.env.VITE_GEMINI_API_KEY
  const apiKey = import.meta.env.VITE_GEMINI_API_KEY || ""; 

  const profile = {
    name: "VASU AGRAWAL",
    title: "AI/ML ENGINEER & FULL-STACK DEV",
    status: "OPEN_TO_WORK",
    location: "SATNA, MP",
    bio: "AI/ML engineer building real, deployable systems. Specializing in deep learning, model debugging, and full-stack AI integration. I prioritize shipping end-to-end projects over theoretical prototypes."
  };

  const stats = [
    { label: "HIGH-SIGNAL PROJECTS", value: "10" },
    { label: "CGPA", value: "3.82" },
    { label: "PRO CERTIFICATIONS", value: "8" },
    { label: "STACK DEPTH", value: "FULL" }
  ];

  const careerTrajectory = [
    { year: '2023', title: 'ML Fundamentals', desc: 'Python, NumPy, Pandas, Basic Regression', level: 25 },
    { year: '2025', title: 'Deep Learning & CNNs', desc: 'TensorFlow, PyTorch, Computer Vision (Medical)', level: 55 },
    { year: '2024', title: 'Full-Stack AI', desc: 'React, FastAPI, Cloud Deployment, System Design', level: 80 },
    { year: '2025', title: 'LLMs & Agents', desc: 'RAG, LangChain, Vector DBs, Latency Optimization', level: 98 },
  ];

  const projects = [
    {
      id: 1,
      alias: 'insightify',
      title: "Insightify (Full-Stack AI)",
      tech: ["MERN Stack", "Redux", "Resume Parsing", "AI Insights"],
      description: "Full-stack React application with AI-enhanced resume recommendations. Engineered a reusable UI system and secure backend for job tracking.",
      link: "https://insightify-react-frontend.onrender.com/",
      status: "LIVE",
      gif: null,
      details: {
        problem: "Job seekers struggle to optimize resumes for ATS (Applicant Tracking Systems) without feedback. Manual comparison is subjective and slow.",
        architecture: "Decoupled Architecture: React Frontend ↔ Node.js API Gateway ↔ Python NLP Microservice. This separation prevents heavy text processing from blocking the Node.js event loop.",
        pipeline: "1. PDF Parsing (PyPDF2) → 2. Text Cleaning (Regex) → 3. Entity Extraction (Spacy NER) → 4. Vector Similarity (TF-IDF/Cosine) → 5. Scoring Logic.",
        decisions: "Chose Microservices over Monolith to allow independent scaling of the NLP worker. Used Redux for state management to cache analysis results and reduce API calls.",
        failures: "Initial OCR failed on double-column PDF layouts. Solved by implementing heuristic layout analysis before extraction.",
        metrics: "Parses standard resumes in <1.2s. 90% accuracy on keyword extraction compared to manual review."
      }
    },
    {
      id: 2,
      alias: 'covid-xray',
      title: "COVID-19 X-Ray Vision",
      tech: ["CNN", "DenseNet121", "Grad-CAM", "Medical Imaging"],
      description: "End-to-end medical imaging pipeline. Implemented Grad-CAM for explainability and CLAHE normalization to achieve high accuracy on heterogeneous data sources.",
      link: "https://github.com/vasuag09/Covid-X-Ray",
      status: "RESEARCH",
      gif: null,
      details: {
        problem: "Rapid screening of COVID-19 from Chest X-Rays is needed, but deep learning models often learn spurious correlations (e.g., hospital markers) instead of pathology.",
        architecture: "Transfer Learning with DenseNet121 backbone. Added custom classification head (GlobalAvgPool → Dense → Dropout → Softmax).",
        pipeline: "Input Image → CLAHE Normalization (Contrast Enhancement) → Augmentation (Rotation/Zoom) → CNN Feature Extractor → Classification.",
        decisions: "Selected DenseNet121 over ResNet50 for better feature propagation and parameter efficiency. Implemented Grad-CAM to verify the model was actually looking at the lungs.",
        failures: "Model initially overfitted on source-specific artifacts. Fixed by rigorous cross-dataset validation and aggressive augmentation.",
        metrics: "96% Sensitivity, 94% Specificity on test set. Inference time <200ms on standard GPU."
      }
    },
    {
      id: 3,
      alias: 'rag-chatbot',
      title: "PDF RAG Chatbot",
      tech: ["LangChain", "FAISS", "LLM", "Vector DB"],
      description: "Retrieval-Augmented Generation system for academic notes. Capable of retrieving context, analyzing documents, and generating accurate citations.",
      link: "#",
      status: "BUILDING",
      gif: null,
      details: {
        problem: "Students waste time searching through hundreds of pages of academic PDFs for specific concepts.",
        architecture: "RAG Pipeline: PDF Loader → RecursiveCharTextSplitter → OpenAI Embeddings → FAISS Vector Store → LLM Chain.",
        pipeline: "Query → Vector Search (Top-K Retrieval) → Context Injection → LLM Generation → Answer.",
        decisions: "Used FAISS for local vector storage to minimize latency and cost compared to Pinecone. Implemented 'Sources' return to hallucination checks.",
        failures: "Chunking logic initially broke sentences, losing context. Optimized chunk overlap to 200 tokens.",
        metrics: "Retrieval Latency <400ms. Cost per query <$0.01."
      }
    },
    {
      id: 4,
      alias: 'airline-sentiment',
      title: "US Airline Sentiment",
      tech: ["NLP", "TF-IDF", "Logistic Regression", "Streamlit"],
      description: "High-precision sentiment analysis on 14k+ tweets. Deployed as an interactive interface for real-time customer feedback classification.",
      link: "https://sentiment-analysis-of-us-airlines-tweets-egszqbuj37bydzk8cypqq.streamlit.app/",
      status: "LIVE",
      gif: "airline_sentiment.gif",
      details: {
        problem: "Airlines receive thousands of tweets daily. Manual triage is impossible during disruptions.",
        architecture: "Streamlit Frontend ↔ Scikit-Learn Inference Pipeline (Pickled Model).",
        pipeline: "Text Preprocessing (Stopwords/Stemming) → TF-IDF Vectorization (Unigrams+Bigrams) → Logistic Regression Classifier.",
        decisions: "Chose Logistic Regression over BERT for this iteration to prioritize inference speed and interpretability (feature weights).",
        failures: "Sarcasm detection remains a challenge with bag-of-words models. Future work involves LSTM/Transformers.",
        metrics: "82% Accuracy on multiclass classification. <50ms inference time."
      }
    },
    {
      id: 5,
      alias: 'insurance-fraud',
      title: "Health Insurance Fraud",
      tech: ["Scikit-Learn", "Imbalance Handling", "Streamlit", "Feature Eng"],
      description: "Fraud-classification pipeline with engineered features. Deployed interface for real-time risk scoring of insurance claims.",
      link: "https://vasuag09-medical-insurance-fraud-app-jlnqby.streamlit.app/",
      status: "LIVE",
      gif: null,
      details: {
        problem: "Fraudulent claims cost insurers billions. Detecting patterns in tabular data requires robust feature engineering.",
        architecture: "Standard ML Pipeline: Data Ingestion → Preprocessing → Random Forest Classifier.",
        pipeline: "SMOTE (Synthetic Minority Over-sampling) → Feature Scaling → Model Training → Threshold Tuning.",
        decisions: "Used SMOTE to handle severe class imbalance (fraud cases <1%). Optimized for Recall over Precision to minimize missed fraud cases.",
        failures: "High False Positive rate initially. Tuned decision threshold to balance risk/operational load.",
        metrics: "ROC-AUC Score: 0.88. Recall: 0.91."
      }
    },
    {
      id: 6,
      alias: 'flower-recognition',
      title: "Flower Recognition",
      tech: ["PyTorch", "ResNet50", "Transfer Learning", "Augmentation"],
      description: "Fine-tuned ResNet50 achieving >95% accuracy. Implemented structured evaluation with confusion matrices and class-level performance metrics.",
      link: "https://github.com/vasuag09/Flower-Recognition",
      status: "RESEARCH",
      gif: null,
      details: {
        problem: "Classifying 102 flower categories requires fine-grained visual discrimination.",
        architecture: "ResNet50 (Pretrained on ImageNet) → Fine-tuning layers.",
        pipeline: "Resize/CenterCrop → Normalization → ResNet Forward Pass → Softmax.",
        decisions: "Utilized 'One Cycle' learning rate policy for faster convergence. Froze early layers to preserve generic feature detectors.",
        failures: "Confusion between visually similar species (e.g., two types of lilies).",
        metrics: "95.4% Test Accuracy. Training time reduced by 40% via LR scheduling."
      }
    },
    {
      id: 7,
      alias: 'airbnb-pricing',
      title: "Airbnb Price Optimization",
      tech: ["XGBoost", "EDA", "Feature Selection", "Regression"],
      description: "Comprehensive pricing model for Rio de Janeiro rentals. Full exploratory data analysis workflow to identify key value drivers in real estate.",
      link: "https://github.com/vasuag09/airbnb_price_optimization",
      status: "CODE",
      gif: null,
      details: {
        problem: "Hosts struggle to set optimal prices. Underpricing loses revenue; overpricing loses bookings.",
        architecture: "XGBoost Regressor with Grid Search Cross-Validation.",
        pipeline: "EDA (Correlation Heatmaps) → Outlier Removal → Feature Encoding → XGBoost Training.",
        decisions: "Chose XGBoost for its ability to handle non-linear relationships and feature importance outputs.",
        failures: "Location data (Lat/Long) was initially noisy. Clustered locations into neighborhoods for better stability.",
        metrics: "RMSE: 0.42 (Log Price). R-Squared: 0.71."
      }
    },
    {
      id: 8,
      alias: 'churn-prediction',
      title: "Customer Churn Prediction",
      tech: ["Classification", "Pandas", "ROC/AUC", "Business Logic"],
      description: "Production-grade classification workflow for banking data. Handles class imbalance with calibrated probability outputs for risk assessment.",
      link: "https://github.com/vasuag09/churn_modelling_bank",
      status: "CODE",
      gif: null,
      details: {
        problem: "Banks need to identify at-risk customers before they leave to offer retention incentives.",
        architecture: "Ensemble Method (Gradient Boosting + Random Forest).",
        pipeline: "Preprocessing → Feature Selection (RFE) → Model Stacking → Probability Calibration.",
        decisions: "Focus on 'Probability Calibration' so the bank can prioritize high-value at-risk customers.",
        failures: "Demographic bias detected in initial model. Re-weighted samples to ensure fairness.",
        metrics: "Lift Score: 4.2 (Top decile)."
      }
    },
    {
      id: 9,
      alias: 'mnist-digit',
      title: "MNIST Digit Classifier",
      tech: ["TensorFlow", "CNN", "Streamlit", "Computer Vision"],
      description: "Interactive handwritten digit recognition app (~99% accuracy). Real-time inference UI deployed on Streamlit for instant testing.",
      link: "https://mchffzbghhdbt4vzhvvxth.streamlit.app/",
      status: "LIVE",
      gif: "mnist_demo.gif",
      details: {
        problem: "Classic computer vision benchmark to demonstrate end-to-end ML deployment.",
        architecture: "Conv2D (32) → MaxPool → Conv2D (64) → MaxPool → Flatten → Dense (128) → Output.",
        pipeline: "Canvas Input → Grayscale Conversion → Resizing (28x28) → Normalization → Inference.",
        decisions: "Used 'Canvas' library in Streamlit to allow real-time drawing input from users.",
        failures: "Users drawing in corners or with thin lines caused prediction errors. Added centering preprocessing step.",
        metrics: "99.1% Validation Accuracy."
      }
    },
    {
      id: 10,
      alias: 'traffic-detection',
      title: "Road Traffic Detection",
      tech: ["MATLAB", "Computer Vision", "Background Subtraction"],
      description: "Foreground segmentation and vehicle detection system using temporal filtering and morphological operations. Optimized for static camera feeds.",
      link: "#",
      status: "CODE",
      gif: null,
      details: {
        problem: "Counting vehicles in static CCTV feeds without heavy deep learning compute.",
        architecture: "Classical Computer Vision Pipeline (No Neural Nets).",
        pipeline: "Gaussian Mixture Models (GMM) Background Subtraction → Morphological Opening/Closing → Blob Analysis.",
        decisions: "Chose GMM over simple frame differencing to handle changing lighting conditions.",
        failures: "Shadows were initially detected as vehicles. Tuned thresholding parameters to suppress shadows.",
        metrics: "Counting Accuracy: 94% on highway footage."
      }
    }
  ];

  const skills = [
    { category: "GEN AI / LLM", items: ["LangChain", "RAG Pipelines", "HuggingFace", "FAISS", "LLM"] },
    { category: "DEEP LEARNING", items: ["TensorFlow", "PyTorch", "Computer Vision", "Grad-CAM", "CNN"] },
    { category: "FULL STACK", items: ["React", "Node.js", "MongoDB", "Streamlit", "MERN Stack"] },
    { category: "DATA ENG", items: ["Pandas", "SQL", "Data Pipelines", "Model Debugging", "EDA"] }
  ];

  const certifications = [
    "IBM Generative AI Professional Certificate",
    "IBM Machine Learning Professional Certificate",
    "Machine Learning Specialization (Andrew Ng)",
    "Meta React Native Specialization",
    "Full-Stack Web Development Bootcamp",
    "Oracle Cloud Infrastructure 2025 AI Foundations",
    "LLM Engineering: Master AI & Agents"
  ];

  const engineeringLogs = [
    {
      id: 1,
      title: "Debugging Tailwind + PostCSS Build Failures in Vite",
      date: "Nov 19, 2025",
      tags: ["DevOps", "Frontend", "React"],
      readTime: "5 min",
      content: `### The Issue\nWhile deploying the portfolio v3 to Vercel, the build failed with 'PostCSS: configuration file not found'. Locally, everything worked. This is a classic environment drift issue.\n\n### Diagnosis\nVite handles PostCSS internally, but when Tailwind is added via CLI, it creates a 'postcss.config.js'. However, the Vercel build pipeline was expecting 'module.exports' (CommonJS) while my project was set to 'type: module' (ESM).\n\n### The Fix\n1.  Renamed 'postcss.config.js' to 'postcss.config.cjs' to force CommonJS interpretation.\n2.  Explicitly updated 'package.json' to ensure 'autoprefixer' and 'tailwindcss' were in 'devDependencies'.\n\n### Key Takeaway\nAlways verify module resolution strategies (ESM vs CJS) when setting up build tools in 2024. The hybrid ecosystem is still fragile.`
    },
    {
      id: 2,
      title: "Optimizing RAG Pipelines: Beyond Naive Chunking",
      date: "Nov 10, 2025",
      tags: ["GenAI", "LangChain", "Optimization"],
      readTime: "8 min",
      content: `### The Latency Problem\nMy PDF RAG Chatbot was taking 4+ seconds to retrieve context. The bottleneck wasn't the LLM generation—it was the vector search and the context injection size.\n\n### Optimization Strategy\n1.  **Chunking**: Switched from 'CharacterTextSplitter' to 'RecursiveCharacterTextSplitter'. Naive splitting broke sentences mid-thought. Recursive splitting respects paragraph boundaries.\n2.  **Overlap**: Increased overlap from 0 to 200 tokens. This preserved context across chunks, reducing hallucination.\n3.  **FAISS Indexing**: Implemented 'IndexFlatL2' for brute-force accuracy on small datasets (<10k vectors), but prepared 'IVFFlat' for scaling.\n\n### Results\nRetrieval latency dropped to <400ms. Quality of answers improved significantly for multi-paragraph queries.`
    },
    {
      id: 3,
      title: "MATLAB Background Subtraction: Tuning GMMs",
      date: "Oct 25, 2025",
      tags: ["Computer Vision", "MATLAB", "Classical CV"],
      readTime: "6 min",
      content: `### The Shadow Problem\nIn the Road Traffic Detection project, standard background subtraction was counting shadows as cars, inflating the count by ~30%.\n\n### Gaussian Mixture Models (GMM)\nI moved from simple frame differencing to GMM. GMM models each pixel as a mixture of Gaussians, allowing it to handle multimodal backgrounds (e.g., waving trees).\n\n### Morphological Filtering\nEven with GMM, noise persisted. I applied a sequence of morphological operations:\n1.  **Erosion**: To remove small salt-and-pepper noise (birds, leaves).\n2.  **Dilation**: To reconnect fragmented blobs of the same vehicle.\n\n### Outcome\nCounting accuracy on the highway dataset improved from 74% to 94%.`
    },
    {
      id: 4,
      title: "Why DenseNet121 Beats ResNet50 for X-Rays",
      date: "Nov 14, 2025",
      tags: ["Deep Learning", "Medical Imaging", "Research"],
      readTime: "7 min",
      content: `### Feature Propagation\nIn medical imaging, low-level texture features (hazy opacities) are as important as high-level semantic features. ResNet adds features; DenseNet concatenates them.\n\n### The Gradient Flow\nDenseNet121 allows direct connections from any layer to all subsequent layers. This mitigates the vanishing gradient problem more effectively than ResNet's skip connections for this specific dataset size (~5k images).\n\n### Explainability via Grad-CAM\nI visualized the activation maps. ResNet50 often focused on the clavicles (bones), which are irrelevant for COVID detection. DenseNet121 correctly attended to the lower lung lobes, aligning with radiological markers for pneumonia.`
    },
    {
      id: 5,
      title: "Scaling the Insightify Resume Parser",
      date: "Jun 15, 2025",
      tags: ["System Design", "Backend", "MERN"],
      readTime: "6 min",
      content: `### The Blocking Event Loop\nParsing a PDF and running NLP (Spacy) is CPU-intensive. Doing this directly in the Node.js main thread blocked all other API requests, causing the UI to freeze for other users.\n\n### Architecture Shift\n1.  **Decoupling**: Moved the NLP logic to a separate Python microservice (FastAPI).\n2.  **Queueing**: Implemented a simple job queue. The Node backend accepts the upload, pushes a job ID, and returns immediately.\n3.  **Polling**: The React frontend polls the status endpoint every 2 seconds until the Python worker completes the analysis.\n\n### Result\nThe system can now handle concurrent uploads without degrading performance for other users.`
    }
  ];

  // Pipeline Data
  const pipelineStages = [
    { label: "Raw", icon: Database },
    { label: "Clean", icon: Filter },
    { label: "Transform", icon: Layers },
    { label: "Batch", icon: Box },
    { label: "Train", icon: Brain },
    { label: "Eval", icon: Activity },
    { label: "Save", icon: Save },
    { label: "Deploy", icon: Globe }
  ];

  // Boot Sequence
  useEffect(() => {
    if (bootRan.current) return; 
    bootRan.current = true;
    const sequence = ["Initializing kernel...", "Loading neural modules...", "Mounting React frontend...", "Establishing secure uplink...", "System ready."];
    setBootSequence([]); setIsBooted(false);
    const timeouts = []; let delay = 0;
    sequence.forEach((step, index) => {
      const id = setTimeout(() => {
        setBootSequence(prev => [...prev, step]);
        if (index === sequence.length - 1) setIsBooted(true);
      }, delay);
      timeouts.push(id); delay += 400;
    });
    return () => timeouts.forEach(clearTimeout);
  }, []);

  // Pipeline Animator
  useEffect(() => {
    const interval = setInterval(() => {
      setPipelineStep((prev) => (prev + 1) % pipelineStages.length);
    }, 1500); // 1.5s per step
    return () => clearInterval(interval);
  }, []);

  // Scroll to bottom of terminal history
  useEffect(() => {
    if (isTerminalOpen && terminalEndRef.current) {
      terminalEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [terminalHistory, isTerminalOpen]);

  // --- AI INTEGRATION LOGIC ---
  const generateAiResponse = async (userQuery) => {
    setIsAiProcessing(true);
    
    // Check for API Key existence
    if (!apiKey || apiKey === "") {
      setIsAiProcessing(false);
      return "Error: API Key not configured. Please add your Gemini API key to the source code.";
    }

    // Construct context from portfolio data
    const systemContext = `
      You are VASU_OS, the AI assistant for Vasu Agrawal's engineering portfolio.
      Your persona: Technical, concise, slightly witty, "hacker" aesthetic.
      
      Here is Vasu's data:
      - Profile: ${JSON.stringify(profile)}
      - Skills: ${JSON.stringify(skills)}
      - Projects: ${JSON.stringify(projects.map(p => ({ title: p.title, tech: p.tech, description: p.description, details: p.details })))}
      - Engineering Logs: ${JSON.stringify(engineeringLogs.map(l => ({ title: l.title, summary: l.content.substring(0, 100) })))}
      
      Answer the user's question based STRICTLY on this data. 
      If asked about something not in the data, say "Data corrupted or not found in sector 7G."
      Keep answers short (under 3 sentences) to fit the terminal style.
    `;

    try {
      const response = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key=${apiKey}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          contents: [{ parts: [{ text: userQuery }] }],
          systemInstruction: { parts: [{ text: systemContext }] }
        }),
      });

      const data = await response.json();
      
      if (data.error) {
         throw new Error(data.error.message);
      }

      const aiText = data.candidates?.[0]?.content?.parts?.[0]?.text || "Error: Neural link unstable.";
      return aiText;

    } catch (error) {
      // Fallback if API key is missing or call fails
      console.error("AI Error:", error);
      return `Error: AI Core Offline. (${error.message})`;
    } finally {
      setIsAiProcessing(false);
    }
  };

  // Terminal Command Handler
  const handleTerminalCommand = async (e) => {
    if (e.key === 'Enter') {
      const command = terminalInput.trim();
      if (!command) return;

      // Add user command to history immediately
      setTerminalHistory(prev => [...prev, { type: 'user', content: command }]);
      setTerminalInput('');

      // Check for Hardcoded System Commands first
      const args = command.split(' ');
      const cmd = args[0];
      let systemOutput = null;

      switch (cmd) {
        case './help':
          systemOutput = [
            "Available commands:",
            "./list-projects           - List all projects with IDs & Aliases",
            "./open [alias/id]         - Open project deep dive",
            "./stats [alias/id]        - View quick stats for a project",
            "./ship-log                - List recent engineering logs",
            "./whoami                  - Display user context",
            "./clear                   - Clear terminal",
            "./retro                   - Toggle Retro Mode",
            "[Any Question]            - Ask VASU_OS AI anything"
          ];
          break;
        case './clear':
          setTerminalHistory([]);
          return;
        case './whoami':
          systemOutput = ["User: GUEST", "Role: RECRUITER / ENGINEER", "Access: READ_ONLY"];
          break;
        case './retro':
          setIsRetro(prev => !prev);
          systemOutput = [`Retro Mode: ${!isRetro ? 'ON' : 'OFF'}`];
          break;
        case './ls':
        case './list-projects':
          systemOutput = projects.map(p => `[${p.id}] ${p.alias} -> ${p.title} (${p.status})`);
          break;
        case './ship-log':
          systemOutput = engineeringLogs.map(l => `[${l.date}] ${l.title}`);
          break;
        case './project': 
        case './open':
          const target = args[1];
          if (target) {
            let proj;
            if (!isNaN(parseInt(target))) {
              proj = projects.find(p => p.id === parseInt(target));
            } else {
              proj = projects.find(p => p.alias === target);
            }

            if (proj) {
              systemOutput = [`Opening Deep Dive for: ${proj.title}...`];
              setTimeout(() => {
                setExpandedProject(proj);
                setIsTerminalOpen(false);
              }, 800);
            } else {
              systemOutput = [`Error: Project '${target}' not found.`];
            }
          } else {
            systemOutput = ["Error: Missing argument. Usage: ./open insightify"];
          }
          break;
        case './stats':
           const statTarget = args[1];
           if (statTarget) {
             const proj = projects.find(p => p.alias === statTarget || p.id === parseInt(statTarget));
             if (proj && proj.details) {
               systemOutput = [
                 `STATS FOR: ${proj.title}`,
                 `--------------------------------`,
                 `Stack: ${proj.tech.join(', ')}`,
                 `Metrics: ${proj.details.metrics}`,
                 `Status: ${proj.status}`
               ];
             } else {
               systemOutput = [`Error: Project '${statTarget}' not found or has no stats.`];
             }
           } else {
             systemOutput = ["Error: Missing argument. Usage: ./stats covid-xray"];
           }
           break;
        default:
          // If not a system command, treat as AI Query
          systemOutput = null; 
      }

      if (systemOutput) {
        // Handle hardcoded response
        if (Array.isArray(systemOutput)) {
          setTerminalHistory(prev => [...prev, ...systemOutput.map(line => ({ type: 'system', content: line }))]);
        } else {
          setTerminalHistory(prev => [...prev, { type: 'system', content: systemOutput }]);
        }
      } else {
        // Handle AI Response
        const aiResponse = await generateAiResponse(command);
        setTerminalHistory(prev => [...prev, { type: 'ai', content: aiResponse }]);
      }
    }
  };

  // Keyboard Shortcuts (Global)
  useEffect(() => {
    const handleKeyDown = (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        setIsTerminalOpen(prev => !prev);
      }
      if (e.key === 'Escape') {
        setPreviewProject(null);
        setExpandedProject(null);
        setReadingLog(null);
        setFilterTech(null);
        setIsTerminalOpen(false);
        setTourStep(0); 
      }
      if (isTerminalOpen) return; 
      if (e.altKey || e.metaKey || e.ctrlKey) return;
      if (e.key === '1') setActiveTab('overview');
      if (e.key === '2') setActiveTab('projects');
      if (e.key === '3') setActiveTab('skills');
      if (e.key === '4') setActiveTab('blog');
      if (e.key === '5') setActiveTab('about');
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isTerminalOpen]);

  useEffect(() => {
    const hasSeenTour = localStorage.getItem('hasSeenTour');
    if (!hasSeenTour) setTimeout(() => setTourStep(1), 3000);
  }, []);

  const handleTourNext = () => {
    if (tourStep < 5) {
      setTourStep(prev => prev + 1);
      if (tourStep === 1) setActiveTab('projects');
      if (tourStep === 3) setActiveTab('blog');
      if (tourStep === 4) setActiveTab('about');
    } else {
      setTourStep(0);
      localStorage.setItem('hasSeenTour', 'true');
      setActiveTab('overview');
    }
  };

  const handleTechClick = (tech) => { setFilterTech(tech); setActiveTab('projects'); setExpandedProject(null); };
  const filteredProjects = filterTech ? projects.filter(p => p.tech.some(t => t.toLowerCase().includes(filterTech.toLowerCase()))) : projects;

  const NavButton = ({ id, icon: Icon, label, shortcut }) => (
    <button onClick={() => { setActiveTab(id); setExpandedProject(null); setReadingLog(null); }} className={`w-full flex items-center justify-between px-4 py-3 text-sm font-mono transition-all duration-200 border-l-2 cursor-pointer ${activeTab === id && !expandedProject && !readingLog ? 'border-emerald-500 bg-emerald-500/10 text-emerald-400' : 'border-transparent text-slate-400 hover:text-emerald-300 hover:bg-slate-800'}`}>
      <div className="flex items-center gap-3"><Icon size={18} /><span>{label.toUpperCase()}</span></div>
      {shortcut && <span className="text-[10px] opacity-50 border border-slate-700 px-1 rounded hidden lg:block">{shortcut}</span>}
    </button>
  );

  // RENDER PIPELINE ANIMATOR
  const renderPipeline = () => (
    <div className="bg-slate-900/30 border border-slate-800 p-6 rounded-lg mb-8 overflow-x-auto">
      <div className="flex items-center gap-3 mb-6">
        <Server className="text-emerald-500" size={24} />
        <h3 className="text-xl font-bold text-white">MLOps Pipeline Architecture</h3>
      </div>
      <div className="flex items-center justify-between min-w-[800px]">
        {pipelineStages.map((stage, idx) => {
          const isActive = idx === pipelineStep;
          const isPast = idx < pipelineStep;
          const Icon = stage.icon;
          return (
            <div key={idx} className="flex items-center flex-1 last:flex-none relative">
              <div className="flex flex-col items-center gap-2 relative z-10">
                <motion.div animate={{ scale: isActive ? 1.1 : 1, borderColor: isActive ? 'rgb(16 185 129)' : isPast ? 'rgb(16 185 129 / 0.5)' : 'rgb(51 65 85)', backgroundColor: isActive ? 'rgb(6 78 59)' : 'rgb(15 23 42)' }} className={`w-12 h-12 rounded-full border-2 flex items-center justify-center transition-colors duration-300 ${isActive ? 'shadow-[0_0_15px_rgba(16,185,129,0.5)]' : ''}`}>
                  <Icon size={20} className={isActive || isPast ? 'text-emerald-400' : 'text-slate-500'} />
                </motion.div>
                <span className={`text-xs font-mono font-bold ${isActive ? 'text-white' : 'text-slate-500'}`}>{stage.label}</span>
                <div className="h-4">
                  {isActive && <motion.div initial={{ scale: 0 }} animate={{ scale: 1 }} className="flex items-center gap-1 text-[10px] text-emerald-400"><RefreshCw size={10} className="animate-spin" /> PROC</motion.div>}
                  {isPast && <motion.div initial={{ scale: 0 }} animate={{ scale: 1 }} className="text-emerald-500"><CheckCircle2 size={12} /></motion.div>}
                </div>
              </div>
              {idx < pipelineStages.length - 1 && (
                <div className="flex-1 h-[2px] bg-slate-800 mx-2 relative overflow-hidden">
                  <motion.div initial={{ x: '-100%' }} animate={isActive ? { x: '100%' } : { x: isPast ? '100%' : '-100%' }} transition={isActive ? { duration: 1.5, repeat: Infinity, ease: "linear" } : { duration: 0 }} className={`absolute inset-0 bg-gradient-to-r from-transparent via-emerald-500 to-transparent ${isPast ? 'opacity-100 w-full bg-emerald-900' : 'w-1/2'}`} />
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );

  // RENDER TOUR OVERLAY
  const renderTour = () => {
    if (tourStep === 0) return null;
    const tourContent = [
      { title: "Welcome to VASU_OS", text: "This is a high-performance engineering portfolio designed for clarity and speed. Let's take a quick look around." },
      { title: "Project Deployments", text: "Access 10+ deployed AI systems here. Click 'View Analysis' on any card for a deep architectural breakdown." },
      { title: "Developer Terminal", text: "Power user? Press Cmd+K (or Ctrl+K) anytime to open the CLI. It is now AI-POWERED—ask it anything about me." },
      { title: "Engineering Log", text: "I don't just write code; I explain it. Read my long-form debugging logs and system design decisions here." },
      { title: "Ready to Collaborate?", text: "I am open to high-impact roles. Download my CV or initiate communication directly." }
    ];
    return (
      <div className="fixed inset-0 z-[100] bg-black/60 backdrop-blur-sm flex items-center justify-center p-4 cursor-auto">
        <motion.div initial={{ scale: 0.9, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} className="bg-slate-900 border border-emerald-500/50 p-8 rounded-lg max-w-md w-full shadow-2xl shadow-emerald-900/40 relative">
          <div className="absolute top-0 right-0 p-4"><button onClick={() => { setTourStep(0); localStorage.setItem('hasSeenTour', 'true'); }} className="text-slate-500 hover:text-white transition-colors cursor-pointer"><X size={20} /></button></div>
          <div className="w-12 h-12 bg-emerald-900/30 rounded-full flex items-center justify-center mb-6 border border-emerald-500/30 text-emerald-400"><Map size={24} /></div>
          <div className="flex items-center gap-2 mb-2"><span className="text-xs font-bold text-emerald-500 tracking-wider uppercase">Step {tourStep} of 5</span><div className="h-px flex-1 bg-slate-800"></div></div>
          <h2 className="text-2xl font-bold text-white mb-4">{tourContent[tourStep-1].title}</h2>
          <p className="text-slate-400 mb-8 leading-relaxed">{tourContent[tourStep-1].text}</p>
          <button onClick={handleTourNext} className="w-full bg-emerald-600 hover:bg-emerald-500 text-white font-bold py-3 rounded transition-all flex items-center justify-center gap-2 cursor-pointer">{tourStep === 5 ? 'Finish Tour' : 'Next Step'} <ChevronRight size={16} /></button>
        </motion.div>
      </div>
    )
  }

  // RENDER TERMINAL OVERLAY (Updated with AI logic)
  const renderTerminal = () => (
    <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -20 }} className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm cursor-pointer" onClick={() => setIsTerminalOpen(false)}>
      <div className="bg-slate-950 border border-emerald-500/30 rounded-lg w-full max-w-3xl h-[60vh] flex flex-col shadow-2xl shadow-emerald-900/20 overflow-hidden font-mono text-sm cursor-auto" onClick={e => e.stopPropagation()}>
        <div className="bg-slate-900 border-b border-slate-800 p-3 flex items-center justify-between">
          <div className="flex items-center gap-2 text-slate-400">
            <Terminal size={16} />
            <span>vasu_os_terminal</span>
            {isAiProcessing && <span className="text-xs text-emerald-500 animate-pulse flex items-center gap-1"><Sparkles size={10}/> AI PROCESSING</span>}
          </div>
          <div className="flex gap-2"><div className="w-3 h-3 rounded-full bg-yellow-500/50"></div><div className="w-3 h-3 rounded-full bg-green-500/50"></div><button onClick={() => setIsTerminalOpen(false)} className="w-3 h-3 rounded-full bg-red-500/50 hover:bg-red-500 transition-colors cursor-pointer"></button></div>
        </div>
        <div className="flex-1 p-4 overflow-y-auto space-y-2 custom-scrollbar" onClick={() => document.getElementById('terminal-input').focus()}>
          {terminalHistory.map((entry, idx) => (
            <div key={idx} className={`${entry.type === 'user' ? 'text-white' : entry.type === 'ai' ? 'text-emerald-300' : 'text-emerald-400/80'}`}>
              {entry.type === 'user' ? (
                <span className="flex gap-2"><span className="text-emerald-500">➜</span><span>{entry.content}</span></span>
              ) : entry.type === 'ai' ? (
                <div className="pl-5 flex gap-2"><Sparkles size={14} className="mt-0.5 shrink-0 opacity-70" /> <span>{entry.content}</span></div>
              ) : (
                <div className="pl-5 whitespace-pre-wrap">{entry.content}</div>
              )}
            </div>
          ))}
          {isAiProcessing && (
            <div className="pl-5 text-emerald-500/50 animate-pulse">Thinking...</div>
          )}
          <div ref={terminalEndRef} />
        </div>
        <div className="p-3 bg-slate-900 border-t border-slate-800 flex items-center gap-2"><span className="text-emerald-500">➜</span><input id="terminal-input" type="text" value={terminalInput} onChange={(e) => setTerminalInput(e.target.value)} onKeyDown={handleTerminalCommand} autoFocus disabled={isAiProcessing} className="bg-transparent border-none outline-none text-white flex-1 placeholder-slate-600 disabled:opacity-50" placeholder={isAiProcessing ? "VASU_OS is thinking..." : "Type ./help or ask a question..."} /></div>
      </div>
    </motion.div>
  );

  if (expandedProject) {
    return (
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className={`min-h-screen bg-slate-950 text-slate-200 font-sans selection:bg-emerald-500/30 selection:text-emerald-200 relative ${isRetro ? 'retro-mode' : ''}`}>
         {isRetro && <div className="pointer-events-none fixed inset-0 z-[60] overflow-hidden h-full w-full bg-[linear-gradient(rgba(18,16,16,0)_50%,rgba(0,0,0,0.25)_50%),linear-gradient(90deg,rgba(255,0,0,0.06),rgba(0,255,0,0.02),rgba(0,0,255,0.06))] bg-[length:100%_2px,6px_100%] opacity-20"></div>}
         {isTerminalOpen && renderTerminal()}
         <div className="md:hidden p-4 border-b border-slate-800 flex justify-between items-center bg-slate-900/50 backdrop-blur"><div className="font-mono font-bold text-emerald-500">./PROJECT_ANALYSIS</div><div className="text-xs text-slate-500">MODE: DEEP_DIVE</div></div>
        <div className="max-w-7xl mx-auto min-h-screen flex flex-col">
           <div className="p-6 md:p-12 border-b border-slate-800 flex items-center justify-between sticky top-0 bg-slate-950/90 backdrop-blur z-10">
             <button onClick={() => setExpandedProject(null)} className="flex items-center gap-2 text-slate-400 hover:text-emerald-400 transition-colors font-mono text-sm cursor-pointer"><ArrowLeft size={16} /> BACK TO DEPLOYMENTS</button>
             <div className="flex gap-4"><a href={expandedProject.link} target="_blank" rel="noreferrer" className="bg-emerald-600 hover:bg-emerald-500 text-white px-4 py-2 rounded text-sm font-mono flex items-center gap-2 transition-colors cursor-pointer">View Live <ExternalLink size={14} /></a></div>
           </div>
           <motion.div initial={{ y: 20, opacity: 0 }} animate={{ y: 0, opacity: 1 }} transition={{ delay: 0.1 }} className="p-6 md:p-12 grid lg:grid-cols-3 gap-12">
             <div className="lg:col-span-2 space-y-12">
               <div><h1 className="text-3xl md:text-4xl font-bold text-white mb-4">{expandedProject.title}</h1><div className="flex flex-wrap gap-2 mb-8">{expandedProject.tech.map((t, i) => (<span key={i} className="text-xs font-mono text-emerald-400 bg-emerald-900/20 border border-emerald-500/20 px-2 py-1 rounded">{t}</span>))}</div><p className="text-slate-300 leading-relaxed text-lg border-l-4 border-emerald-500 pl-6">{expandedProject.details?.problem}</p></div>
               <div><h3 className="text-xl font-bold text-white flex items-center gap-2 mb-4"><Layout className="text-emerald-500" /> System Architecture</h3><div className="bg-slate-900/50 border border-slate-800 p-6 rounded-lg font-mono text-sm text-slate-400">{expandedProject.details?.architecture}</div></div>
               <div><h3 className="text-xl font-bold text-white flex items-center gap-2 mb-4"><GitBranch className="text-emerald-500" /> Engineering Pipeline</h3><p className="text-slate-400 leading-relaxed">{expandedProject.details?.pipeline}</p></div>
             </div>
             <div className="space-y-8">
               <div className="bg-slate-900/30 border border-slate-800 p-6 rounded-lg"><h3 className="text-sm font-bold text-white flex items-center gap-2 mb-4 uppercase tracking-wider"><Brain size={16} className="text-emerald-500" /> Key Decisions</h3><p className="text-slate-400 text-sm leading-relaxed">{expandedProject.details?.decisions}</p></div>
               <div className="bg-red-900/10 border border-red-500/20 p-6 rounded-lg"><h3 className="text-sm font-bold text-red-400 flex items-center gap-2 mb-4 uppercase tracking-wider"><AlertTriangle size={16} /> Failure Modes</h3><p className="text-slate-400 text-sm leading-relaxed">{expandedProject.details?.failures}</p></div>
               <div className="bg-emerald-900/10 border border-emerald-500/20 p-6 rounded-lg"><h3 className="text-sm font-bold text-emerald-400 flex items-center gap-2 mb-4 uppercase tracking-wider"><Target size={16} /> Performance Metrics</h3><div className="text-slate-300 font-mono text-sm">{expandedProject.details?.metrics}</div></div>
             </div>
           </motion.div>
        </div>
      </motion.div>
    );
  }

  if (readingLog) {
    return (
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className={`min-h-screen bg-slate-950 text-slate-200 font-sans selection:bg-emerald-500/30 selection:text-emerald-200 relative ${isRetro ? 'retro-mode' : ''}`}>
        {isRetro && <div className="pointer-events-none fixed inset-0 z-[60] overflow-hidden h-full w-full bg-[linear-gradient(rgba(18,16,16,0)_50%,rgba(0,0,0,0.25)_50%),linear-gradient(90deg,rgba(255,0,0,0.06),rgba(0,255,0,0.02),rgba(0,0,255,0.06))] bg-[length:100%_2px,6px_100%] opacity-20"></div>}
        {isTerminalOpen && renderTerminal()}
        <div className="max-w-4xl mx-auto min-h-screen flex flex-col">
           <div className="p-6 border-b border-slate-800 flex items-center justify-between sticky top-0 bg-slate-950/90 backdrop-blur z-10">
             <button onClick={() => setReadingLog(null)} className="flex items-center gap-2 text-slate-400 hover:text-emerald-400 transition-colors font-mono text-sm cursor-pointer"><ArrowLeft size={16} /> BACK TO LOGS</button>
           </div>
           <motion.div initial={{ y: 20, opacity: 0 }} animate={{ y: 0, opacity: 1 }} transition={{ delay: 0.1 }} className="p-8 md:p-16">
              <div className="mb-8">
                <div className="flex gap-2 mb-4">{readingLog.tags.map((tag, i) => (<span key={i} className="text-xs font-mono text-emerald-400 bg-emerald-900/20 border border-emerald-500/20 px-2 py-1 rounded">#{tag}</span>))}</div>
                <h1 className="text-3xl md:text-4xl font-bold text-white mb-4">{readingLog.title}</h1>
                <div className="flex items-center gap-4 text-sm text-slate-500 font-mono"><span>{readingLog.date}</span><span>•</span><span>{readingLog.readTime} read</span></div>
              </div>
              <div className="prose prose-invert prose-emerald max-w-none">
                {readingLog.content.split('\n').map((line, i) => {
                  if (line.trim().startsWith('###')) return <h3 key={i} className="text-xl font-bold text-white mt-8 mb-4 flex items-center gap-2"><Hash size={16} className="text-emerald-500" />{line.replace('###', '').trim()}</h3>;
                  if (line.trim().match(/^\d\./)) return <div key={i} className="ml-4 mb-2 text-slate-300 pl-4 border-l border-slate-700">{line.trim()}</div>;
                  return <p key={i} className="text-slate-300 leading-relaxed mb-4">{line}</p>;
                })}
              </div>
           </motion.div>
        </div>
      </motion.div>
    )
  }

  return (
    <div className={`min-h-screen bg-slate-950 text-slate-200 font-sans selection:bg-emerald-500/30 selection:text-emerald-200 relative ${isRetro ? 'retro-mode' : ''}`}>
      
      {/* CRT Overlay */}
      {isRetro && (
         <div className="pointer-events-none fixed inset-0 z-[60] overflow-hidden h-full w-full bg-[linear-gradient(rgba(18,16,16,0)_50%,rgba(0,0,0,0.25)_50%),linear-gradient(90deg,rgba(255,0,0,0.06),rgba(0,255,0,0.02),rgba(0,0,255,0.06))] bg-[length:100%_2px,6px_100%] opacity-20"></div>
      )}
      
      {/* Guided Tour Overlay */}
      <AnimatePresence>
        {tourStep > 0 && renderTour()}
      </AnimatePresence>

      <AnimatePresence>{isTerminalOpen && renderTerminal()}</AnimatePresence>

      <AnimatePresence>
      {previewProject && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/90 backdrop-blur-sm cursor-pointer" onClick={() => setPreviewProject(null)}>
          <motion.div initial={{ scale: 0.9 }} animate={{ scale: 1 }} exit={{ scale: 0.9 }} className="bg-slate-900 border border-slate-700 rounded-lg w-full max-w-5xl overflow-hidden shadow-2xl shadow-black cursor-auto" onClick={e => e.stopPropagation()}>
            <div className="flex justify-between items-center p-4 border-b border-slate-800 bg-slate-900">
              <h3 className="text-lg font-bold text-white flex items-center gap-2"><Play size={16} className="text-emerald-500" /> Demo: {previewProject.title}</h3>
              <button onClick={() => setPreviewProject(null)} className="text-slate-400 hover:text-white transition-colors p-1 hover:bg-slate-800 rounded cursor-pointer"><X size={20} /></button>
            </div>
            <div className="bg-black p-2 flex justify-center items-center min-h-[400px]">
              {previewProject.gif ? <img src={previewProject.gif} alt={`${previewProject.title} Demo`} className="w-full h-auto max-h-[85vh] object-contain rounded" /> : <div className="h-64 flex items-center justify-center text-slate-500 font-mono">Preview not available</div>}
            </div>
          </motion.div>
        </motion.div>
      )}
      </AnimatePresence>

      <div className="md:hidden p-4 border-b border-slate-800 flex justify-between items-center bg-slate-900/50 backdrop-blur">
        <div className="font-mono font-bold text-emerald-500">./{profile.name.replace(' ', '_')}</div>
        <div className="text-xs text-slate-500">V 4.1.0</div>
      </div>

      <div className="flex flex-col md:flex-row max-w-7xl mx-auto min-h-screen">
        <nav className="w-full md:w-64 bg-slate-900/30 border-r border-slate-800 flex flex-col justify-between md:fixed md:h-full z-10">
          <div>
            <div className="p-6 hidden md:block">
              <div className="w-12 h-12 bg-slate-800 rounded border border-slate-700 flex items-center justify-center mb-4"><Terminal className="text-emerald-500" /></div>
              <h1 className="font-bold text-lg tracking-wider">{profile.name}</h1>
              <p className="text-xs text-slate-500 font-mono mt-1">{profile.title}</p>
            </div>
            <div className="flex flex-col gap-1 mt-4">
              <NavButton id="overview" icon={Layout} label="System Overview" shortcut="1" />
              <NavButton id="projects" icon={Code2} label="Deployments" shortcut="2" />
              <NavButton id="skills" icon={Cpu} label="Tech Stack" shortcut="3" />
              <NavButton id="blog" icon={BookOpen} label="Engineering Log" shortcut="4" />
              <NavButton id="about" icon={User} label="About Me" shortcut="5" />
            </div>
            <div className="px-6 mt-6 hidden md:block">
               <button onClick={() => setIsTerminalOpen(true)} className="w-full bg-slate-900 border border-slate-700 p-2 rounded text-xs font-mono text-slate-400 hover:text-emerald-500 hover:border-emerald-500/50 transition-all flex items-center justify-between group cursor-pointer mb-2">
                 <div className="flex items-center gap-2"><Command size={12} /> Terminal</div>
                 <span className="bg-slate-800 px-1.5 py-0.5 rounded text-[10px] group-hover:text-white">⌘K</span>
               </button>
               <button onClick={() => setIsRetro(prev => !prev)} className={`w-full border p-2 rounded text-xs font-mono transition-all flex items-center justify-center gap-2 cursor-pointer ${isRetro ? 'bg-emerald-900/30 border-emerald-500 text-emerald-400' : 'bg-slate-900 border-slate-700 text-slate-400 hover:text-white'}`}>
                 <Monitor size={12} /> {isRetro ? 'Retro Mode: ON' : 'Retro Mode: OFF'}
               </button>
               <button onClick={() => setTourStep(1)} className="w-full border border-slate-700 p-2 rounded text-xs font-mono text-slate-400 hover:text-emerald-500 hover:border-emerald-500/50 transition-all flex items-center justify-center gap-2 cursor-pointer mt-2 bg-slate-900">
                 <Map size={12} /> Start Tour
               </button>
            </div>
          </div>
          <div className="p-6 border-t border-slate-800">
            <div className="flex gap-4 justify-center md:justify-start">
              <a href="https://github.com/vasuag09" target="_blank" rel="noreferrer" className="text-slate-400 hover:text-white transition-colors cursor-pointer"><Github size={20} /></a>
              <a href="https://linkedin.com" target="_blank" rel="noreferrer" className="text-slate-400 hover:text-white transition-colors cursor-pointer"><Linkedin size={20} /></a>
              <a href="mailto:vasuagrawal1040@gmail.com" className="text-slate-400 hover:text-white transition-colors cursor-pointer"><Mail size={20} /></a>
            </div>
          </div>
        </nav>

        <main className="flex-1 md:ml-64 p-6 md:p-12 lg:p-16 overflow-y-auto">
          <header className="flex justify-between items-end mb-12 border-b border-slate-800 pb-4">
            <div>
              <h2 className="text-2xl font-bold text-white mb-1 flex items-center gap-3">
                {activeTab === 'overview' && <Layout className="text-emerald-500" />}
                {activeTab === 'projects' && <Code2 className="text-emerald-500" />}
                {activeTab === 'skills' && <Cpu className="text-emerald-500" />}
                {activeTab === 'blog' && <BookOpen className="text-emerald-500" />}
                {activeTab === 'about' && <User className="text-emerald-500" />}
                {activeTab.toUpperCase()}
              </h2>
              <div className="flex items-center gap-2 text-xs font-mono text-emerald-500/80"><span className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse"></span>SYSTEM ONLINE</div>
            </div>
            <div className="flex items-end gap-6">
              <div className="hidden md:block text-right">
                <a href="https://drive.google.com/file/d/1oPs0eZW7bvuWO032v5NZQs-E1PFO2jp9/view?usp=sharing" target="_blank" rel="noreferrer" className="flex items-center gap-2 text-xs font-mono text-emerald-500 border border-emerald-500/30 bg-emerald-500/5 px-3 py-2 rounded hover:bg-emerald-500/10 transition-all cursor-pointer"><Download size={14} /> DOWNLOAD CV</a>
              </div>
              <div className="hidden md:block text-right"><div className="text-xs text-slate-500 font-mono">CURRENT LOCATION</div><div className="text-sm font-mono">{profile.location}</div></div>
            </div>
          </header>

          <AnimatePresence mode="wait">
            {activeTab === 'overview' && (
              <motion.div key="overview" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -20 }} transition={{ duration: 0.3 }} className="space-y-8">
                <div className="bg-slate-900/50 border border-slate-800 p-8 rounded-lg relative overflow-hidden group">
                  <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity"><Brain size={120} /></div>
                  <div className="font-mono text-emerald-500 text-sm mb-4">
                    {bootSequence.map((log, i) => <div key={i} className="opacity-70"><span className="mr-2 text-slate-600">{`>`}</span>{log}</div>)}
                    {isBooted && <span className="animate-pulse">_</span>}
                  </div>
                  <h1 className="text-3xl md:text-5xl font-bold text-white mb-6 max-w-2xl leading-tight">Build fast. <br/><span className="text-slate-500">Ship models.</span> <br/>Solve problems.</h1>
                  <p className="text-slate-400 max-w-xl leading-relaxed mb-8">{profile.bio}</p>
                  <div className="flex gap-4">
                    <button onClick={() => setActiveTab('projects')} className="bg-emerald-600 hover:bg-emerald-500 text-white font-mono px-6 py-3 rounded flex items-center gap-2 transition-all active:scale-95 cursor-pointer">VIEW SHIPMENTS <ChevronRight size={16} /></button>
                    <a href="https://drive.google.com/file/d/1oPs0eZW7bvuWO032v5NZQs-E1PFO2jp9/view?usp=sharing" target="_blank" rel="noreferrer" className="border border-slate-600 hover:border-white text-slate-300 hover:text-white font-mono px-6 py-3 rounded flex items-center gap-2 transition-all cursor-pointer">DOWNLOAD CV <Download size={16} /></a>
                  </div>
                </div>
                
                {/* MLOps Pipeline Animator - NEW FEATURE */}
                {renderPipeline()}

                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  {stats.map((stat, idx) => (
                    <div key={idx} className="bg-slate-900/30 border border-slate-800 p-4 rounded hover:border-emerald-500/30 transition-colors">
                      <div className="text-xs text-slate-500 font-mono mb-1">{stat.label}</div>
                      <div className="text-2xl font-bold text-white">{stat.value}</div>
                    </div>
                  ))}
                </div>
              </motion.div>
            )}

            {/* ... (Projects, Skills, Blog, About tabs are unchanged) ... */}
            {activeTab === 'projects' && (
              <motion.div key="projects" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -20 }} transition={{ duration: 0.3 }} className="space-y-6">
                {filterTech && (
                  <div className="flex items-center gap-4 bg-emerald-900/20 border border-emerald-500/30 p-3 rounded mb-4 text-sm">
                    <Filter size={16} className="text-emerald-500" />
                    <span className="text-slate-300">Filtering by: <span className="text-white font-bold">{filterTech}</span></span>
                    <button onClick={() => setFilterTech(null)} className="ml-auto text-xs bg-slate-800 hover:bg-slate-700 px-2 py-1 rounded text-slate-300 cursor-pointer">Clear Filter</button>
                  </div>
                )}
                <div className="grid md:grid-cols-2 xl:grid-cols-3 gap-6">
                  {filteredProjects.length > 0 ? filteredProjects.map((project) => (
                    <motion.div layout key={project.id} whileHover={{ scale: 1.02 }} className="group bg-slate-900/30 border border-slate-800 p-6 rounded-lg hover:border-emerald-500/50 transition-all hover:bg-slate-900/50 flex flex-col">
                      <div className="flex justify-between items-start mb-4 gap-4">
                        <div className="flex items-start gap-2 min-w-0"><Briefcase size={16} className="text-emerald-500 mt-1 shrink-0" /><h3 className="text-lg font-bold text-white group-hover:text-emerald-400 transition-colors leading-tight break-words">{project.title}</h3></div>
                        <span className={`text-[10px] font-mono px-2 py-1 rounded border flex items-center gap-2 shrink-0 ${project.status === 'LIVE' ? 'bg-emerald-500/10 border-emerald-500/50 text-emerald-400' : project.status === 'RESEARCH' ? 'bg-blue-500/10 border-blue-500/50 text-blue-400' : project.status === 'CODE' ? 'bg-purple-500/10 border-purple-500/50 text-purple-400' : 'bg-yellow-500/10 border-yellow-500/50 text-yellow-400'}`}>{project.status === 'BUILDING' && <Loader size={10} className="animate-spin" />}{project.status}</span>
                      </div>
                      <p className="text-slate-400 text-sm leading-relaxed mb-6 flex-grow">{project.description}</p>
                      <div className="flex flex-wrap gap-2 mb-6">{project.tech.map((t, i) => (<button key={i} onClick={(e) => { e.stopPropagation(); setFilterTech(t); }} className={`text-xs font-mono px-2 py-1 rounded transition-colors cursor-pointer ${filterTech === t ? 'bg-emerald-500 text-white' : 'bg-slate-800 text-slate-300 hover:bg-slate-700'}`}>{t}</button>))}</div>
                      <div className="mt-auto pt-4 border-t border-slate-800/50 flex items-center gap-4">
                        <button onClick={() => setExpandedProject(project)} className="flex items-center gap-2 text-sm font-medium text-emerald-500 hover:text-emerald-400 transition-colors cursor-pointer"><BookOpen size={14} /> View Analysis</button>
                        {project.gif && <button onClick={() => setPreviewProject(project)} className="flex items-center gap-2 text-sm font-medium text-slate-300 hover:text-white transition-colors ml-auto cursor-pointer"><Eye size={14} /> Watch Demo</button>}
                        {!project.gif && project.link !== '#' && <a href={project.link} target="_blank" rel="noreferrer" className="flex items-center gap-2 text-sm font-medium text-slate-300 hover:text-white transition-colors ml-auto cursor-pointer"><ExternalLink size={14} /> Source</a>}
                      </div>
                    </motion.div>
                  )) : <div className="col-span-full flex flex-col items-center justify-center py-12 text-slate-500"><Code2 size={48} className="mb-4 opacity-20" /><p>No projects found with tech stack: "{filterTech}"</p><button onClick={() => setFilterTech(null)} className="mt-4 text-emerald-500 hover:underline cursor-pointer">Clear Filters</button></div>}
                </div>
              </motion.div>
            )}

            {activeTab === 'skills' && (
              <motion.div key="skills" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -20 }} transition={{ duration: 0.3 }} className="space-y-6">
                <div className="bg-slate-900/30 border border-slate-800 rounded-lg p-6 mb-8">
                   <div className="flex items-center gap-3 mb-6"><Activity className="text-emerald-500" size={24} /><h3 className="text-xl font-bold text-white">Competence Trajectory</h3></div>
                   <div className="relative border-l-2 border-slate-800 ml-4 space-y-8 pb-2">
                      {careerTrajectory.map((milestone, idx) => (
                         <div key={idx} className="relative pl-8">
                            <div className="absolute -left-[9px] top-0 w-4 h-4 rounded-full bg-slate-900 border-2 border-emerald-500"></div>
                            <div className="flex flex-col sm:flex-row sm:items-center justify-between mb-1">
                               <span className="text-emerald-400 font-bold font-mono text-sm">{milestone.year}</span>
                               <div className="hidden sm:block flex-1 mx-4 h-px bg-slate-800"></div>
                               <div className="flex items-center gap-2 text-xs text-slate-500 font-mono"><span>LEVEL: {milestone.level}%</span></div>
                            </div>
                            <h4 className="text-lg font-bold text-white mb-1">{milestone.title}</h4>
                            <p className="text-slate-400 text-sm">{milestone.desc}</p>
                         </div>
                      ))}
                   </div>
                </div>

                <div className="grid md:grid-cols-2 gap-6">
                  {skills.map((skillGroup, idx) => (
                    <div key={idx} className="bg-slate-900/30 border border-slate-800 rounded-lg p-6">
                      <h3 className="text-emerald-500 font-mono text-sm mb-6 border-b border-slate-800 pb-2">// {skillGroup.category}</h3>
                      <div className="grid grid-cols-2 gap-4">
                        {skillGroup.items.map((skill, sIdx) => (
                          <button key={sIdx} onClick={() => handleTechClick(skill)} className="flex items-center gap-3 group text-left hover:bg-slate-800/50 p-2 rounded -ml-2 transition-all cursor-pointer">
                            <div className="w-2 h-2 bg-slate-700 rounded-sm group-hover:bg-emerald-500 transition-colors"></div>
                            <span className="text-slate-300 group-hover:text-white font-mono text-sm">{skill}</span>
                          </button>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
                <div className="bg-gradient-to-r from-slate-900 to-slate-900/50 border border-slate-800 rounded-lg p-8">
                  <div className="flex items-center gap-3 mb-6"><Award className="text-emerald-500" size={24} /><h3 className="text-xl font-bold text-white">Professional Certifications</h3></div>
                  <div className="grid md:grid-cols-2 gap-4">
                    {certifications.map((cert, idx) => (
                      <div key={idx} className="flex items-start gap-3 text-slate-300 hover:text-white transition-colors">
                         <ChevronRight className="text-emerald-500 mt-0.5 shrink-0" size={16} />
                         <span className="font-mono text-sm">{cert}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </motion.div>
            )}

            {activeTab === 'blog' && (
              <motion.div key="blog" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -20 }} transition={{ duration: 0.3 }} className="max-w-4xl space-y-6">
                <div className="flex items-center justify-between mb-8"><h2 className="text-2xl font-bold text-white flex items-center gap-3"><BookOpen className="text-emerald-500" /> Engineering Log</h2><span className="text-xs font-mono text-slate-500">LATEST ENTRIES</span></div>
                <div className="grid gap-6">
                  {engineeringLogs.map((post, idx) => (
                    <motion.div key={idx} whileHover={{ scale: 1.01 }} className="bg-slate-900/30 border border-slate-800 p-6 rounded-lg hover:border-emerald-500/30 transition-all group">
                      <div className="flex justify-between items-start mb-2"><h3 className="text-xl font-bold text-white group-hover:text-emerald-400 transition-colors">{post.title}</h3><span className="text-xs font-mono text-slate-500 whitespace-nowrap ml-4">{post.date}</span></div>
                      <div className="flex gap-2 mb-4">{post.tags.map((tag, i) => (<span key={i} className="text-[10px] bg-slate-800 text-slate-400 px-2 py-0.5 rounded font-mono">#{tag}</span>))}</div>
                      <button onClick={() => setReadingLog(post)} className="text-emerald-500 text-sm font-mono hover:underline flex items-center gap-2 cursor-pointer">Read Analysis <ChevronRight size={14} /></button>
                    </motion.div>
                  ))}
                </div>
              </motion.div>
            )}

            {activeTab === 'about' && (
              <motion.div key="about" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -20 }} transition={{ duration: 0.3 }} className="max-w-3xl space-y-6">
                <div className="font-mono text-sm space-y-6 text-slate-300">
                  <div className="bg-slate-950 border border-slate-800 p-6 rounded">
                    <p className="mb-4 text-emerald-500">{`> cat about_me.txt`}</p>
                    <p className="mb-4 leading-relaxed">I am an AI/ML engineer and full-stack developer focused on building real, deployable systems. I specialize in deep learning, model debugging, computer vision, classical ML pipelines, and building AI-powered web applications.</p>
                    <p className="mb-4 leading-relaxed">I work across the stack—Python, ML, TensorFlow, React/Next.js, Node.js—and prioritize shipping end-to-end projects that demonstrate clear engineering depth and practical value.</p>
                    <p className="text-slate-500"># machine_learning # full_stack # systems_engineering</p>
                  </div>
                  <div className="bg-emerald-900/10 border border-emerald-500/30 p-8 rounded-lg relative overflow-hidden">
                    <div className="absolute top-0 right-0 p-4 opacity-5"><Zap size={100} /></div>
                    <div className="relative z-10">
                      <h3 className="text-xl font-bold text-white mb-6 flex items-center gap-2"><div className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse"></div>OPEN FOR COLLABORATION</h3>
                      <div className="grid md:grid-cols-2 gap-8 mb-8">
                        <div>
                          <h4 className="text-emerald-400 text-xs font-bold uppercase tracking-wider mb-3">Roles I'm Targeting</h4>
                          <ul className="space-y-2">
                            <li className="flex items-center gap-2 text-slate-300 text-sm"><CheckCircle2 size={14} className="text-emerald-500" /> AI/ML Engineer</li>
                            <li className="flex items-center gap-2 text-slate-300 text-sm"><CheckCircle2 size={14} className="text-emerald-500" /> Full-Stack Developer</li>
                            <li className="flex items-center gap-2 text-slate-300 text-sm"><CheckCircle2 size={14} className="text-emerald-500" /> Technical Co-Founder</li>
                          </ul>
                        </div>
                        <div>
                          <h4 className="text-emerald-400 text-xs font-bold uppercase tracking-wider mb-3">Services & Expertise</h4>
                          <ul className="space-y-2">
                            <li className="flex items-center gap-2 text-slate-300 text-sm"><CheckCircle2 size={14} className="text-emerald-500" /> MVP Development (0 to 1)</li>
                            <li className="flex items-center gap-2 text-slate-300 text-sm"><CheckCircle2 size={14} className="text-emerald-500" /> RAG Pipeline Design</li>
                            <li className="flex items-center gap-2 text-slate-300 text-sm"><CheckCircle2 size={14} className="text-emerald-500" /> Model Fine-Tuning & Deployment</li>
                          </ul>
                        </div>
                      </div>
                      <a href="mailto:vasuagrawal1040@gmail.com" className="inline-flex items-center gap-2 bg-emerald-600 hover:bg-emerald-500 text-white font-mono px-6 py-3 rounded transition-all active:scale-95 cursor-pointer"><MessageSquare size={18} /> INITIATE COMMUNICATION</a>
                    </div>
                  </div>
                  <div className="border-l-2 border-slate-800 pl-6 py-2">
                    <h4 className="text-white font-bold mb-2">Education</h4>
                    <ul className="list-none space-y-2 text-slate-400">
                      <li><span className="text-emerald-400 font-bold">MBA Tech - Computer Engineering</span><br/>SVKM's NMIMS (Mukesh Patel School)<br/><span className="text-xs text-slate-500">2023 - Present | CGPA: 3.82/4</span></li>
                    </ul>
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </main>
      </div>
    </div>
  );
};

export default Portfolio;