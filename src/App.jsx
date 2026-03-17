import React, { Suspense, lazy, useEffect } from "react";
import { Routes, Route } from "react-router-dom";
import { AnimatePresence } from "framer-motion";

// Layout
import Sidebar from "./components/layout/Sidebar";
import Header from "./components/layout/Header";
import MobileNav from "./components/layout/MobileNav";

// UI
import SkipToContent from "./components/ui/SkipToContent";
import { ErrorBoundary } from "./components/ui/ErrorBoundary";

// Effects
import RetroOverlay from "./components/effects/RetroOverlay";
import ParticleField from "./components/effects/ParticleField";

// Modals
import GifPreview from "./components/modals/GifPreview";
import GuidedTour from "./components/modals/GuidedTour";

// Hooks
import { useKeyboardShortcuts } from "./hooks/useKeyboardShortcuts";

// Context
import { UIProvider } from "./context/UIProvider";
import { useUI } from "./hooks/useUI";

// Lazy-loaded components
const TerminalModal = lazy(() => import("./components/modals/Terminal"));
const ProjectDeepDive = lazy(() => import("./components/views/ProjectDeepDive"));
const BlogReader = lazy(() => import("./components/views/BlogReader"));

// Lazy-loaded sections
const Overview = lazy(() => import("./components/sections/Overview"));
const Projects = lazy(() => import("./components/sections/Projects"));
const Skills = lazy(() => import("./components/sections/Skills"));
const Blog = lazy(() => import("./components/sections/Blog"));
const About = lazy(() => import("./components/sections/About"));

function LoadingFallback() {
  return (
    <div className="flex items-center justify-center py-20">
      <div className="text-emerald-500 font-mono text-sm animate-pulse flex items-center gap-2">
        <span className="w-2 h-2 bg-emerald-500 rounded-full animate-ping"></span>
        LOADING MODULE...
      </div>
    </div>
  );
}

function AppContent() {
  const {
    isRetro,
    isTerminalOpen,
    setIsTerminalOpen,
    previewProject,
    setPreviewProject,
    tourStep,
    setTourStep,
    terminal,
    setIsMobileMenuOpen,
  } = useUI();

  // Keyboard shortcuts
  useKeyboardShortcuts({
    isTerminalOpen,
    setIsTerminalOpen,
    setPreviewProject,
    setIsMobileMenuOpen,
  });

  // Tour auto-start
  useEffect(() => {
    const hasSeenTour = localStorage.getItem("hasSeenTour");
    if (!hasSeenTour) {
      const timer = setTimeout(() => setTourStep(1), 3000);
      return () => clearTimeout(timer);
    }
  }, [setTourStep]);

  return (
    <div
      className={`min-h-screen bg-slate-950 text-slate-200 font-sans selection:bg-emerald-500/30 selection:text-emerald-200 relative transition-colors duration-500 ${
        isRetro ? "retro-mode" : ""
      }`}
    >
      <SkipToContent />
      <ParticleField />

      {isRetro && <RetroOverlay />}

      <AnimatePresence>
        {tourStep > 0 && <GuidedTour tourStep={tourStep} setTourStep={setTourStep} />}
      </AnimatePresence>

      <AnimatePresence>
        {isTerminalOpen && (
          <Suspense fallback={null}>
            <TerminalModal
              {...terminal}
              onClose={() => setIsTerminalOpen(false)}
            />
          </Suspense>
        )}
      </AnimatePresence>

      <AnimatePresence>
        {previewProject && (
          <GifPreview
            project={previewProject}
            onClose={() => setPreviewProject(null)}
          />
        )}
      </AnimatePresence>

      <MobileNav />

      <div className="flex flex-col md:flex-row max-w-7xl mx-auto min-h-screen">
        <Sidebar />

        <main
          id="main-content"
          className="flex-1 md:ml-64 p-6 md:p-12 lg:p-16 overflow-y-auto relative z-[1]"
        >
          <Header />
          <Routes>
            <Route path="/" element={<Suspense fallback={<LoadingFallback />}><Overview /></Suspense>} />
            <Route path="/projects" element={<Suspense fallback={<LoadingFallback />}><Projects /></Suspense>} />
            <Route path="/projects/:alias" element={<Suspense fallback={<LoadingFallback />}><ProjectDeepDive /></Suspense>} />
            <Route path="/skills" element={<Suspense fallback={<LoadingFallback />}><Skills /></Suspense>} />
            <Route path="/blog" element={<Suspense fallback={<LoadingFallback />}><Blog /></Suspense>} />
            <Route path="/blog/:slug" element={<Suspense fallback={<LoadingFallback />}><BlogReader /></Suspense>} />
            <Route path="/about" element={<Suspense fallback={<LoadingFallback />}><About /></Suspense>} />
            <Route path="*" element={<Suspense fallback={<LoadingFallback />}><Overview /></Suspense>} />
          </Routes>
        </main>
      </div>
    </div>
  );
}

export default function App() {
  return (
    <ErrorBoundary>
      <UIProvider>
        <AppContent />
      </UIProvider>
    </ErrorBoundary>
  );
}
