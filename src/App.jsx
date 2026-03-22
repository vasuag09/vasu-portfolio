import React, { Suspense, lazy, useEffect } from "react";
import { Routes, Route, useLocation } from "react-router-dom";
import { AnimatePresence } from "framer-motion";

// Layout
import LayerNav from "./components/layout/LayerNav";
import { getActiveLayer } from "./utils/layers";
import Header from "./components/layout/Header";
import MobileNav from "./components/layout/MobileNav";

// UI
import SkipToContent from "./components/ui/SkipToContent";
import ScrollProgress from "./components/ui/ScrollProgress";
import { ErrorBoundary } from "./components/ui/ErrorBoundary";

// Canvas
import NeuralNetwork3D from "./components/canvas/NeuralNetwork3D";

// Effects
import CustomCursor from "./components/effects/CustomCursor";

// Synapse
import SynapseButton from "./components/synapse/SynapseButton";

// Modals
import GifPreview from "./components/modals/GifPreview";
import GuidedTour from "./components/modals/GuidedTour";

// Hooks
import { useKeyboardShortcuts } from "./hooks/useKeyboardShortcuts";

// Context
import { UIProvider } from "./context/UIProvider";
import { TerminalProvider } from "./context/TerminalProvider";
import { useUI } from "./hooks/useUI";
import { useTerminalContext } from "./hooks/useTerminalContext";

// Lazy-loaded components
const SynapsePanel = lazy(() => import("./components/synapse/SynapsePanel"));
const ProjectDeepDive = lazy(() => import("./components/views/ProjectDeepDive"));
// Lazy-loaded sections
const Hero = lazy(() => import("./components/sections/Hero"));
const Projects = lazy(() => import("./components/sections/Projects"));
const Skills = lazy(() => import("./components/sections/Skills"));
const Research = lazy(() => import("./components/sections/Research"));
const About = lazy(() => import("./components/sections/About"));
const NotFound = lazy(() => import("./components/sections/NotFound"));

function LoadingFallback() {
  return (
    <div className="flex items-center justify-center py-20">
      <div className="flex items-center gap-3 text-cyan-500/60 font-mono text-sm">
        <div className="w-2 h-2 rounded-full bg-cyan-500/60 animate-neural-pulse" />
        Loading layer...
      </div>
    </div>
  );
}

function AppContent() {
  const location = useLocation();
  const activeLayer = getActiveLayer(location.pathname);

  const {
    isTerminalOpen,
    setIsTerminalOpen,
    previewProject,
    setPreviewProject,
    tourStep,
    setTourStep,
    setIsMobileMenuOpen,
  } = useUI();

  const terminal = useTerminalContext();

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
    <div className="min-h-screen bg-[var(--bg-void)] text-[var(--text-primary)] relative">
      <CustomCursor />
      <SkipToContent />
      <ScrollProgress />

      {/* 3D Neural network background */}
      <NeuralNetwork3D activeLayer={activeLayer} />

      {/* Guided tour */}
      <AnimatePresence>
        {tourStep > 0 && <GuidedTour tourStep={tourStep} setTourStep={setTourStep} />}
      </AnimatePresence>

      {/* Synapse AI panel */}
      <AnimatePresence>
        {isTerminalOpen && (
          <Suspense fallback={null}>
            <SynapsePanel
              {...terminal}
              onClose={() => setIsTerminalOpen(false)}
            />
          </Suspense>
        )}
      </AnimatePresence>

      {/* GIF preview */}
      <AnimatePresence>
        {previewProject && (
          <GifPreview
            project={previewProject}
            onClose={() => setPreviewProject(null)}
          />
        )}
      </AnimatePresence>

      {/* Synapse floating button (hidden when panel is open) */}
      {!isTerminalOpen && (
        <SynapseButton onClick={() => setIsTerminalOpen(true)} />
      )}

      {/* Desktop layer navigation */}
      <LayerNav />

      {/* Main content */}
      <main
        id="main-content"
        className="relative z-[1] md:ml-20 px-6 md:px-12 lg:px-16 pt-6 md:pt-12 lg:pt-16 pb-20 md:pb-8 max-w-6xl"
      >
        <Header />
        <Routes>
          <Route path="/" element={<Suspense fallback={<LoadingFallback />}><Hero /></Suspense>} />
          <Route path="/projects" element={<Suspense fallback={<LoadingFallback />}><Projects /></Suspense>} />
          <Route path="/projects/:alias" element={<Suspense fallback={<LoadingFallback />}><ProjectDeepDive /></Suspense>} />
          <Route path="/skills" element={<Suspense fallback={<LoadingFallback />}><Skills /></Suspense>} />
          <Route path="/research" element={<Suspense fallback={<LoadingFallback />}><Research /></Suspense>} />
          <Route path="/about" element={<Suspense fallback={<LoadingFallback />}><About /></Suspense>} />
          <Route path="*" element={<Suspense fallback={<LoadingFallback />}><NotFound /></Suspense>} />
        </Routes>
      </main>

      {/* Mobile navigation */}
      <MobileNav />
    </div>
  );
}

export default function App() {
  return (
    <ErrorBoundary>
      <UIProvider>
        <TerminalProvider>
          <AppContent />
        </TerminalProvider>
      </UIProvider>
    </ErrorBoundary>
  );
}
