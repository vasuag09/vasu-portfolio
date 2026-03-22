import React from "react";
import { motion } from "framer-motion";
import { useNavigate } from "react-router-dom";
import { ArrowUpRight } from "lucide-react";
import { engineeringLogs } from "../../data/blog-posts";
import { useDocumentTitle } from "../../hooks/useDocumentTitle";
import Waveform from "../canvas/Waveform";

export default function Blog() {
  useDocumentTitle("Signal Propagation");
  const navigate = useNavigate();

  return (
    <section id="blog" className="max-w-4xl space-y-5 pb-20 md:pb-8">
      {engineeringLogs.map((post, idx) => (
        <motion.button
          key={post.id}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: idx * 0.08 }}
          whileHover={{ y: -2 }}
          onClick={() => navigate(`/blog/${post.slug}`)}
          className="glass-card w-full text-left p-0 overflow-hidden group cursor-pointer block"
        >
          {/* Waveform header */}
          <div className="px-5 pt-4 pb-1 opacity-40 group-hover:opacity-70 transition-opacity">
            <Waveform seed={post.title} />
          </div>

          <div className="px-5 pb-5">
            {/* Signal label + date */}
            <div className="flex items-center justify-between mb-2">
              <span className="text-[10px] font-mono text-cyan-500/50 tracking-wider">
                SIGNAL #{post.id}
              </span>
              <span className="text-[10px] font-mono text-slate-600">
                {post.date}
              </span>
            </div>

            {/* Title */}
            <h3
              className="text-lg font-semibold text-white group-hover:text-cyan-200 transition-colors mb-3"
              style={{ fontFamily: "var(--font-display)" }}
            >
              {post.title}
            </h3>

            {/* Tags */}
            <div className="flex items-center justify-between">
              <div className="flex gap-2 flex-wrap">
                {post.tags.map((tag, i) => (
                  <span
                    key={i}
                    className="text-[10px] bg-[rgba(0,212,255,0.04)] border border-[rgba(0,212,255,0.08)] text-slate-500 px-2 py-0.5 rounded-full font-mono"
                  >
                    #{tag}
                  </span>
                ))}
              </div>
              <span className="text-xs font-mono text-cyan-500/50 group-hover:text-cyan-400/80 flex items-center gap-1 transition-colors shrink-0 ml-4">
                Decode <ArrowUpRight size={12} />
              </span>
            </div>
          </div>
        </motion.button>
      ))}
    </section>
  );
}
