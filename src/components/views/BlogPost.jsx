import React from "react";
import { motion } from "framer-motion";
import { useParams, useNavigate, Link } from "react-router-dom";
import { ArrowLeft, Clock, Tag } from "lucide-react";
import { engineeringLogs } from "../../data/blog-posts";
import { useDocumentTitle } from "../../hooks/useDocumentTitle";
import { renderMarkdown } from "../../utils/markdown";
import Waveform from "../canvas/Waveform";

/**
 * Full blog post reader view with markdown rendering.
 */
export default function BlogPost() {
  const { slug } = useParams();
  const navigate = useNavigate();
  const post = engineeringLogs.find((p) => p.slug === slug);

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
            Back to Signal Log
          </button>
        </div>
      </div>
    );
  }

  return (
    <motion.article
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="max-w-3xl pb-20 md:pb-8"
    >
      {/* Back navigation */}
      <button
        onClick={() => navigate("/blog")}
        className="flex items-center gap-2 text-slate-500 hover:text-cyan-400 transition-colors font-mono text-sm cursor-pointer mb-8"
      >
        <ArrowLeft size={16} /> Back to Signal Log
      </button>

      {/* Waveform header */}
      <div className="glass-card-static overflow-hidden mb-6">
        <div className="px-6 pt-6 pb-2 opacity-40">
          <Waveform seed={post.title} />
        </div>
        <div className="px-6 pb-6">
          {/* Meta */}
          <div className="flex items-center gap-4 mb-4">
            <span className="text-[10px] font-mono text-cyan-500/50 tracking-wider">
              SIGNAL #{post.id}
            </span>
            <span className="flex items-center gap-1 text-[10px] font-mono text-slate-600">
              <Clock size={10} /> {post.date}
            </span>
            <span className="flex items-center gap-1 text-[10px] font-mono text-slate-600">
              <Clock size={10} /> {post.readTime}
            </span>
          </div>

          {/* Title */}
          <h1
            className="text-2xl md:text-3xl font-bold text-white mb-4"
            style={{ fontFamily: "var(--font-display)" }}
          >
            {post.title}
          </h1>

          {/* Tags */}
          <div className="flex gap-2 flex-wrap">
            {post.tags.map((tag, i) => (
              <span
                key={i}
                className="text-[10px] bg-[rgba(0,212,255,0.04)] border border-[rgba(0,212,255,0.08)] text-slate-500 px-2 py-0.5 rounded-full font-mono flex items-center gap-1"
              >
                <Tag size={8} />
                {tag}
              </span>
            ))}
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="glass-card-static p-6 md:p-8">
        <div className="prose prose-invert prose-sm max-w-none space-y-4 text-slate-400 leading-relaxed">
          {renderMarkdown(post.content)}
        </div>
      </div>

      {/* Navigation */}
      <div className="flex justify-between mt-6">
        <Link
          to="/blog"
          className="text-xs font-mono text-cyan-500/50 hover:text-cyan-400 transition-colors"
        >
          ← ALL SIGNALS
        </Link>
      </div>
    </motion.article>
  );
}
