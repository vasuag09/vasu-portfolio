import React from "react";
import { Cpu, Github } from "lucide-react";

/**
 * Subtle footer with copyright and tech stack.
 */
export default function Footer() {
  const year = new Date().getFullYear();

  return (
    <footer className="border-t border-[rgba(0,212,255,0.04)] mt-16 pt-6 pb-24 md:pb-8">
      <div className="flex flex-col md:flex-row items-center justify-between gap-3 text-[10px] font-mono text-slate-600">
        <div className="flex items-center gap-2">
          <Cpu size={10} className="text-cyan-500/30" />
          <span>© {year} VASU_OS v2.0 · Built with React + Three.js</span>
        </div>
        <div className="flex items-center gap-4">
          <a
            href="https://github.com/vasuag09/vasu-portfolio"
            target="_blank"
            rel="noreferrer"
            className="flex items-center gap-1 hover:text-cyan-400/60 transition-colors"
          >
            <Github size={10} />
            Source
          </a>
          <span className="text-slate-700">·</span>
          <span>Last updated {year}</span>
        </div>
      </div>
    </footer>
  );
}
