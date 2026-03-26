import React, { Suspense, lazy, useEffect } from "react";
import { Routes, Route, useLocation } from "react-router-dom";
import { AnimatePresence } from "framer-motion";

// Layout
import LayerNav from "./components/layout/LayerNav";
import { getActiveLayer } from "./data/navigation";
import Header from "./components/layout/Header";
import MobileNav from "./components/layout/MobileNav";
import Footer from "./components/layout/Footer";
import StatusBar from "./components/layout/StatusBar";

// UI
import SkipToContent from "./components/ui/SkipToContent";
import ScrollProgress from "./components/ui/ScrollProgress";
import { ErrorBoundary } from "./components/ui/ErrorBoundary";

// Canvas
import NeuralNetwork3D from "./components/canvas/NeuralNetwork3D";

// Effects
import CustomCursor from "./components/effects/CustomCursor";
import PageTransition from "./components/effects/PageTransition";

// Synapse
import SynapseButton from "./components/synapse/SynapseButton";

// Modals
import GifPreview from "./components/modals/GifPreview";
import GuidedTour from "./components/modals/GuidedTour";
import KeyboardShortcuts from "./components/modals/KeyboardShortcuts";

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
const BlogPost = lazy(() => import("./components/views/BlogPost"));

// Lazy-loaded sections
const Hero = lazy(() => import("./components/sections/Hero"));
const Projects = lazy(() => import("./components/sections/Projects"));
const Skills = lazy(() => import("./components/sections/Skills"));
const Research = lazy(() => import("./components/sections/Research"));
const About = lazy(() => import("./components/sections/About"));
const Blog = lazy(() => import("./components/sections/Blog"));
const NotFound = lazy(() => import("./components/sections/NotFound"));

function LoadingFallback() {
  return (
    <div className="space-y-6 py-8 animate-pulse">
      <div className="h-8 w-48 bg-[rgba(0,212,255,0.04)] rounded-lg" />
      <div className="h-4 w-80 bg-[rgba(0,212,255,0.03)] rounded" />
      <div className="grid md:grid-cols-2 gap-4 mt-8">
        <div className="h-48 bg-[rgba(0,212,255,0.03)] rounded-xl border border-[rgba(0,212,255,0.04)]" />
        <div className="h-48 bg-[rgba(0,212,255,0.03)] rounded-xl border border-[rgba(0,212,255,0.04)]" />
        <div className="h-48 bg-[rgba(0,212,255,0.03)] rounded-xl border border-[rgba(0,212,255,0.04)]" />
        <div className="h-48 bg-[rgba(0,212,255,0.03)] rounded-xl border border-[rgba(0,212,255,0.04)]" />
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
    isShortcutsOpen,
    setIsShortcutsOpen,
    reducedEffects,
  } = useUI();

  const terminal = useTerminalContext();

  useKeyboardShortcuts({
    isTerminalOpen,
    setIsTerminalOpen,
    setPreviewProject,
    setIsMobileMenuOpen,
    setIsShortcutsOpen,
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

      {/* OS-style status bar */}
      <StatusBar />

      {/* 3D Neural network background */}
      {!reducedEffects && <NeuralNetwork3D activeLayer={activeLayer} />}

      {/* Guided tour */}
      <AnimatePresence>
        {tourStep > 0 && <GuidedTour tourStep={tourStep} setTourStep={setTourStep} />}
      </AnimatePresence>

      {/* Keyboard shortcuts overlay */}
      <KeyboardShortcuts
        isOpen={isShortcutsOpen}
        onClose={() => setIsShortcutsOpen(false)}
      />

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

      {/* Synapse floating button */}
      {!isTerminalOpen && (
        <SynapseButton onClick={() => setIsTerminalOpen(true)} />
      )}

      {/* Desktop layer navigation */}
      <LayerNav />

      {/* Main content — offset for status bar on desktop (h-7 = 28px) */}
      <main
        id="main-content"
        className="relative z-[1] md:ml-20 px-6 md:px-12 lg:px-16 pt-6 md:pt-16 lg:pt-20 pb-20 md:pb-8 max-w-6xl"
      >
        <Header />
        <PageTransition>
          <Routes location={location}>
            <Route path="/" element={<Suspense fallback={<LoadingFallback />}><Hero /></Suspense>} />
            <Route path="/projects" element={<Suspense fallback={<LoadingFallback />}><Projects /></Suspense>} />
            <Route path="/projects/:alias" element={<Suspense fallback={<LoadingFallback />}><ProjectDeepDive /></Suspense>} />
            <Route path="/skills" element={<Suspense fallback={<LoadingFallback />}><Skills /></Suspense>} />
            <Route path="/research" element={<Suspense fallback={<LoadingFallback />}><Research /></Suspense>} />
            <Route path="/about" element={<Suspense fallback={<LoadingFallback />}><About /></Suspense>} />
            <Route path="/blog" element={<Suspense fallback={<LoadingFallback />}><Blog /></Suspense>} />
            <Route path="/blog/:slug" element={<Suspense fallback={<LoadingFallback />}><BlogPost /></Suspense>} />
            <Route path="*" element={<Suspense fallback={<LoadingFallback />}><NotFound /></Suspense>} />
          </Routes>
        </PageTransition>
        <Footer />
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
