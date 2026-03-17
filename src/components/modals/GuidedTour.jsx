import React, { useRef, useEffect } from "react";
import { motion } from "framer-motion";
import { useNavigate } from "react-router-dom";
import { Map, ChevronRight, X } from "lucide-react";
import { tourSteps, tourTabMapping } from "../../data/tour";
import { useFocusTrap } from "../../hooks/useFocusTrap";

export default function GuidedTour({ tourStep, setTourStep }) {
  const navigate = useNavigate();
  const modalRef = useRef(null);

  useFocusTrap(modalRef, () => setTourStep(0), tourStep > 0);

  useEffect(() => {
    if (tourStep > 0) {
      modalRef.current?.querySelector("button")?.focus();
    }
  }, [tourStep]);

  const handleNext = () => {
    if (tourStep < 5) {
      setTourStep((prev) => prev + 1);
      // Navigate to the relevant tab
      const nextStep = tourStep + 1;
      if (tourTabMapping[nextStep]) {
        navigate(`/${tourTabMapping[nextStep]}`);
      }
    } else {
      setTourStep(0);
      localStorage.setItem("hasSeenTour", "true");
      navigate("/");
    }
  };

  const handleClose = () => {
    setTourStep(0);
    localStorage.setItem("hasSeenTour", "true");
  };

  return (
    <div
      className="fixed inset-0 z-[100] bg-black/60 backdrop-blur-sm flex items-center justify-center p-4 cursor-auto"
      role="dialog"
      aria-modal="true"
      aria-label="Guided tour"
    >
      <motion.div
        ref={modalRef}
        initial={{ scale: 0.9, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        className="bg-slate-900 border border-emerald-500/50 p-8 rounded-lg max-w-md w-full shadow-2xl shadow-emerald-900/40 relative"
      >
        <div className="absolute top-0 right-0 p-4">
          <button
            onClick={handleClose}
            aria-label="Close tour"
            className="text-slate-500 hover:text-white transition-colors"
          >
            <X size={20} />
          </button>
        </div>
        <div className="w-12 h-12 bg-emerald-900/30 rounded-full flex items-center justify-center mb-6 border border-emerald-500/30 text-emerald-400">
          <Map size={24} />
        </div>
        <div className="flex items-center gap-2 mb-2">
          <span className="text-xs font-bold text-emerald-500 tracking-wider uppercase">
            Step {tourStep} of 5
          </span>
          <div className="h-px flex-1 bg-slate-800"></div>
        </div>
        <h2 className="text-2xl font-bold text-white mb-4">
          {tourSteps[tourStep - 1].title}
        </h2>
        <p className="text-slate-400 mb-8 leading-relaxed">
          {tourSteps[tourStep - 1].text}
        </p>
        <button
          onClick={handleNext}
          className="w-full bg-emerald-600 hover:bg-emerald-500 text-white font-bold py-3 rounded transition-all flex items-center justify-center gap-2"
        >
          {tourStep === 5 ? "Finish Tour" : "Next Step"}{" "}
          <ChevronRight size={16} />
        </button>
      </motion.div>
    </div>
  );
}
