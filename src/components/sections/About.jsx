import React from "react";
import {
  Zap,
  CheckCircle2,
  MessageSquare,
} from "lucide-react";
import { useDocumentTitle } from "../../hooks/useDocumentTitle";

import SectionWrapper from "../layout/SectionWrapper";

export default function About() {
  useDocumentTitle("About Me");

  return (
    <SectionWrapper id="about" className="max-w-3xl space-y-6">
      <div className="font-mono text-sm space-y-6 text-slate-300">
        <div className="bg-slate-950 border border-slate-800 p-6 rounded">
          <p className="mb-4 text-emerald-500">{`> cat about_me.txt`}</p>
          <p className="mb-4 leading-relaxed">
            I am an AI/ML engineer and full-stack developer focused on building
            real, deployable systems. I specialize in deep learning, model
            debugging, computer vision, classical ML pipelines, and building
            AI-powered web applications.
          </p>
          <p className="mb-4 leading-relaxed">
            I work across the stack—Python, ML, TensorFlow, React/Next.js,
            Node.js—and prioritize shipping end-to-end projects that demonstrate
            clear engineering depth and practical value.
          </p>
          <p className="text-slate-500">
            # machine_learning # full_stack # systems_engineering
          </p>
        </div>

        <div className="bg-emerald-900/10 border border-emerald-500/30 p-8 rounded-lg relative overflow-hidden">
          <div className="absolute top-0 right-0 p-4 opacity-5">
            <Zap size={100} />
          </div>
          <div className="relative z-10">
            <h3 className="text-xl font-bold text-white mb-6 flex items-center gap-2">
              <div className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse"></div>
              OPEN FOR COLLABORATION
            </h3>
            <div className="grid md:grid-cols-2 gap-8 mb-8">
              <div>
                <h4 className="text-emerald-400 text-xs font-bold uppercase tracking-wider mb-3">
                  Roles I&apos;m Targeting
                </h4>
                <ul className="space-y-2">
                  {["AI/ML Engineer", "Full-Stack Developer", "Technical Co-Founder"].map(
                    (role) => (
                      <li
                        key={role}
                        className="flex items-center gap-2 text-slate-300 text-sm"
                      >
                        <CheckCircle2
                          size={14}
                          className="text-emerald-500"
                        />{" "}
                        {role}
                      </li>
                    ),
                  )}
                </ul>
              </div>
              <div>
                <h4 className="text-emerald-400 text-xs font-bold uppercase tracking-wider mb-3">
                  Services & Expertise
                </h4>
                <ul className="space-y-2">
                  {[
                    "MVP Development (0 to 1)",
                    "RAG Pipeline Design",
                    "Model Fine-Tuning & Deployment",
                  ].map((service) => (
                    <li
                      key={service}
                      className="flex items-center gap-2 text-slate-300 text-sm"
                    >
                      <CheckCircle2
                        size={14}
                        className="text-emerald-500"
                      />{" "}
                      {service}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
            <a
              href="mailto:vasuagrawal1040@gmail.com"
              className="inline-flex items-center gap-2 bg-emerald-600 hover:bg-emerald-500 text-white font-mono px-6 py-3 rounded transition-all active:scale-95 cursor-pointer"
            >
              <MessageSquare size={18} /> INITIATE COMMUNICATION
            </a>
          </div>
        </div>

        <div className="border-l-2 border-slate-800 pl-6 py-2">
          <h4 className="text-white font-bold mb-2">Education</h4>
          <ul className="list-none space-y-2 text-slate-400">
            <li>
              <span className="text-emerald-400 font-bold">
                MBA Tech - Computer Engineering
              </span>
              <br />
              SVKM&apos;s NMIMS (Mukesh Patel School)
              <br />
              <span className="text-xs text-slate-500">
                2023 - Present | CGPA: 3.82/4
              </span>
            </li>
          </ul>
        </div>
      </div>
    </SectionWrapper>
  );
}
