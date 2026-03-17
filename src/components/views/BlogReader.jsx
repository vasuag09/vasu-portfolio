import React from "react";
import { motion } from "framer-motion";
import { useParams, useNavigate } from "react-router-dom";
import { ArrowLeft } from "lucide-react";
import { engineeringLogs } from "../../data/blog-posts";
import { useDocumentTitle } from "../../hooks/useDocumentTitle";
import { renderMarkdown } from "../../utils/markdown";

export default function BlogReader() {
  const { slug } = useParams();
  const navigate = useNavigate();
  const post = engineeringLogs.find((l) => l.slug === slug);

  useDocumentTitle(post?.title || "Post Not Found");

  if (!post) {
    return (
      <div className="min-h-screen bg-slate-950 flex items-center justify-center">
        <div className="text-center">
          <h2 className="text-2xl font-bold text-white mb-4 font-mono">
            POST NOT FOUND
          </h2>
          <button
            onClick={() => navigate("/blog")}
            className="text-emerald-500 hover:underline font-mono cursor-pointer"
          >
            Back to Engineering Log
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto flex flex-col">
      <div className="flex items-center justify-between mb-8">
        <button
          onClick={() => navigate("/blog")}
          className="flex items-center gap-2 text-slate-400 hover:text-emerald-400 transition-colors font-mono text-sm cursor-pointer"
        >
          <ArrowLeft size={16} /> BACK TO LOGS
        </button>
      </div>
      <motion.div
        initial={{ y: 20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ delay: 0.1 }}
        className="space-y-8"
      >
        <div className="mb-8">
          <div className="flex gap-2 mb-4">
            {post.tags.map((tag, i) => (
              <span
                key={i}
                className="text-xs font-mono text-emerald-400 bg-emerald-900/20 border border-emerald-500/20 px-2 py-1 rounded"
              >
                #{tag}
              </span>
            ))}
          </div>
          <h1 className="text-3xl md:text-4xl font-bold text-white mb-4">
            {post.title}
          </h1>
          <div className="flex items-center gap-4 text-sm text-slate-500 font-mono">
            <span>{post.date}</span>
            <span>•</span>
            <span>{post.readTime} read</span>
          </div>
        </div>
        <div className="prose prose-invert prose-emerald max-w-none">
          {renderMarkdown(post.content)}
        </div>
      </motion.div>
    </div>
  );
}
