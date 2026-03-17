import React, { useRef, useEffect } from "react";
import { motion } from "framer-motion";
import { Play, X } from "lucide-react";
import { useFocusTrap } from "../../hooks/useFocusTrap";

export default function GifPreview({ project, onClose }) {
  const modalRef = useRef(null);
  useFocusTrap(modalRef, onClose, !!project);

  useEffect(() => {
    if (project) {
      modalRef.current?.querySelector("button")?.focus();
    }
  }, [project]);

  if (!project) return null;

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/90 backdrop-blur-sm cursor-pointer"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-label={`Demo preview for ${project.title}`}
    >
      <motion.div
        ref={modalRef}
        initial={{ scale: 0.9 }}
        animate={{ scale: 1 }}
        exit={{ scale: 0.9 }}
        className="bg-slate-900 border border-slate-700 rounded-lg w-full max-w-5xl overflow-hidden shadow-2xl shadow-black cursor-auto"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex justify-between items-center p-4 border-b border-slate-800 bg-slate-900">
          <h3 className="text-lg font-bold text-white flex items-center gap-2">
            <Play size={16} className="text-emerald-500" /> Demo: {project.title}
          </h3>
          <button
            onClick={onClose}
            aria-label="Close preview"
            className="text-slate-400 hover:text-white transition-colors p-1 hover:bg-slate-800 rounded cursor-pointer"
          >
            <X size={20} />
          </button>
        </div>
        <div className="bg-black p-2 flex justify-center items-center min-h-[400px]">
          {project.gif ? (
            <img
              src={project.gif}
              alt={`${project.title} Demo`}
              className="w-full h-auto max-h-[85vh] object-contain rounded"
            />
          ) : (
            <div className="h-64 flex items-center justify-center text-slate-500 font-mono">
              Preview not available
            </div>
          )}
        </div>
      </motion.div>
    </motion.div>
  );
}
