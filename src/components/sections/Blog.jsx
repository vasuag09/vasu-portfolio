import React from "react";
import { motion } from "framer-motion";
import { useNavigate } from "react-router-dom";
import { BookOpen, ChevronRight } from "lucide-react";
import { engineeringLogs } from "../../data/blog-posts";
import { useDocumentTitle } from "../../hooks/useDocumentTitle";

import SectionWrapper from "../layout/SectionWrapper";

export default function Blog() {
  useDocumentTitle("Engineering Log");
  const navigate = useNavigate();

  return (
    <SectionWrapper id="blog" className="max-w-4xl space-y-6">
      <div className="flex items-center justify-between mb-8">
        <h2 className="text-2xl font-bold text-white flex items-center gap-3">
          <BookOpen className="text-emerald-500" /> Engineering Log
        </h2>
        <span className="text-xs font-mono text-slate-500">
          LATEST ENTRIES
        </span>
      </div>
      <div className="grid gap-6">
        {engineeringLogs.map((post, idx) => (
          <motion.div
            key={post.id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: idx * 0.08, duration: 0.3 }}
            whileHover={{ scale: 1.01 }}
            className="bg-slate-900/30 border border-slate-800 p-6 rounded-lg hover:border-emerald-500/30 transition-all group"
          >
            <div className="flex justify-between items-start mb-2">
              <h3 className="text-xl font-bold text-white group-hover:text-emerald-400 transition-colors">
                {post.title}
              </h3>
              <span className="text-xs font-mono text-slate-500 whitespace-nowrap ml-4">
                {post.date}
              </span>
            </div>
            <div className="flex gap-2 mb-4">
              {post.tags.map((tag, i) => (
                <span
                  key={i}
                  className="text-[11px] bg-slate-800 text-slate-400 px-2 py-0.5 rounded font-mono"
                >
                  #{tag}
                </span>
              ))}
            </div>
            <button
              onClick={() => navigate(`/blog/${post.slug}`)}
              className="text-emerald-500 text-sm font-mono hover:underline flex items-center gap-2 cursor-pointer"
            >
              Read Analysis <ChevronRight size={14} />
            </button>
          </motion.div>
        ))}
      </div>
    </SectionWrapper>
  );
}
