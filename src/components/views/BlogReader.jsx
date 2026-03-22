import React from "react";
import { motion } from "framer-motion";
import { useParams, useNavigate } from "react-router-dom";
import { ArrowLeft } from "lucide-react";
import { engineeringLogs } from "../../data/blog-posts";
import { useDocumentTitle } from "../../hooks/useDocumentTitle";
import { renderMarkdown } from "../../utils/markdown";
import Waveform from "../canvas/Waveform";

export default function BlogReader() {
  const { slug } = useParams();
  const navigate = useNavigate();
  const post = engineeringLogs.find((l) => l.slug === slug);

  useDocumentTitle(post?.title || "Signal Not Found");

  if (!post) {
    return (
      <div className="flex items-center justify-center min-h-[50vh]">
        <div className="text-center">
          <h2
            className="text-2xl font-bold text-white mb-4"
            style={{ fontFamily: "var(--font-display)" }}
          >
            SIGNAL NOT FOUND
          </h2>
          <button
            onClick={() => navigate("/blog")}
            className="text-cyan-500 hover:text-cyan-400 font-mono cursor-pointer text-sm"
          >
            Back to Signal Propagation
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-4xl pb-20 md:pb-8">
      <button
        onClick={() => navigate("/blog")}
        className="flex items-center gap-2 text-slate-500 hover:text-cyan-400 transition-colors font-mono text-sm cursor-pointer mb-8"
      >
        <ArrowLeft size={16} /> Back to Signals
      </button>

      <motion.div
        initial={{ y: 20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        className="space-y-6"
      >
        {/* Waveform decoration */}
        <div className="opacity-30">
          <Waveform seed={post.title} />
        </div>

        {/* Header */}
        <div>
          <div className="flex gap-2 mb-4 flex-wrap">
            {post.tags.map((tag, i) => (
              <span
                key={i}
                className="text-[10px] font-mono text-cyan-500/50 bg-[rgba(0,212,255,0.04)] border border-[rgba(0,212,255,0.08)] px-2 py-0.5 rounded-full"
              >
                #{tag}
              </span>
            ))}
          </div>
          <h1
            className="text-3xl md:text-4xl font-bold text-white mb-4"
            style={{ fontFamily: "var(--font-display)" }}
          >
            {post.title}
          </h1>
          <div className="flex items-center gap-4 text-sm text-slate-600 font-mono">
            <span>{post.date}</span>
            <span className="text-slate-700">·</span>
            <span>{post.readTime} read</span>
          </div>
        </div>

        {/* Content */}
        <div className="glass-card-static p-6 md:p-8">
          <div className="prose prose-invert max-w-none text-slate-400 leading-relaxed [&_h3]:text-white [&_h3]:font-semibold [&_h3]:mt-6 [&_h3]:mb-3 [&_strong]:text-slate-200 [&_code]:text-cyan-400 [&_code]:bg-[rgba(0,212,255,0.06)] [&_code]:px-1.5 [&_code]:py-0.5 [&_code]:rounded [&_a]:text-cyan-400 [&_a]:underline [&_a]:decoration-cyan-500/30 [&_a:hover]:decoration-cyan-400">
            {renderMarkdown(post.content)}
          </div>
        </div>
      </motion.div>
    </div>
  );
}
