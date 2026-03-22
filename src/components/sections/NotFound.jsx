import React from "react";
import { motion } from "framer-motion";
import { AlertTriangle, Home, ArrowLeft } from "lucide-react";
import { Link, useNavigate } from "react-router-dom";
import { useDocumentTitle } from "../../hooks/useDocumentTitle";

export default function NotFound() {
  useDocumentTitle("404 — Node Not Found");
  const navigate = useNavigate();

  return (
    <section className="flex items-center justify-center min-h-[60vh]">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center space-y-6 max-w-md"
      >
        <div className="flex justify-center">
          <div className="w-20 h-20 rounded-full bg-red-500/5 border border-red-500/20 flex items-center justify-center">
            <AlertTriangle size={36} className="text-red-400/70" />
          </div>
        </div>

        <div className="space-y-2">
          <h1
            className="text-4xl font-bold text-red-400/80"
            style={{ fontFamily: "var(--font-display)" }}
          >
            404
          </h1>
          <p className="text-sm font-mono text-slate-500">NODE_NOT_FOUND</p>
          <p className="text-sm text-slate-600">
            This node does not exist in the neural network.
          </p>
        </div>

        <div className="glass-card-static p-4 text-left font-mono text-xs text-slate-600 space-y-1">
          <p className="text-red-400/60">$ trace --route</p>
          <p>[WARN] Route resolution failed</p>
          <p>[ERR] No matching node in any layer</p>
          <p>[INFO] Navigate to a known layer</p>
        </div>

        <div className="flex gap-3 justify-center">
          <button
            onClick={() => navigate(-1)}
            className="border border-slate-700 hover:border-slate-500 text-slate-400 hover:text-white font-mono px-5 py-2.5 rounded-lg flex items-center gap-2 transition-all cursor-pointer text-sm"
          >
            <ArrowLeft size={14} /> BACK
          </button>
          <Link
            to="/"
            className="bg-gradient-to-r from-cyan-600 to-cyan-500 hover:from-cyan-500 hover:to-cyan-400 text-white font-mono px-5 py-2.5 rounded-lg flex items-center gap-2 transition-all cursor-pointer text-sm shadow-[0_0_15px_rgba(0,212,255,0.15)]"
          >
            <Home size={14} /> INPUT LAYER
          </Link>
        </div>
      </motion.div>
    </section>
  );
}
