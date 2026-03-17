import React from "react";
import { motion } from "framer-motion";
import { ChevronRight, Download } from "lucide-react";
import { Link } from "react-router-dom";
import { profile, stats } from "../../data/profile";
import { useBootSequence } from "../../hooks/useBootSequence";
import { useDocumentTitle } from "../../hooks/useDocumentTitle";
import { CV_LINK } from "../../data/constants";
import BootSequence from "../effects/BootSequence";
import Pipeline from "./Pipeline";
import GlowOrb from "../effects/GlowOrb";

import SectionWrapper from "../layout/SectionWrapper";

export default function Overview() {
  useDocumentTitle("System Overview");
  const { bootSequence, isBooted, skipBoot } = useBootSequence();

  return (
    <SectionWrapper id="overview" className="space-y-8">
      <div className="relative">
        <GlowOrb className="w-96 h-96 -top-20 -right-20" />
        <BootSequence
          bootSequence={bootSequence}
          isBooted={isBooted}
          skipBoot={skipBoot}
        />
        <div className="relative z-10 px-8 pb-8 mt-6">
          <p className="text-slate-400 max-w-xl leading-relaxed mb-8">
            {profile.bio}
          </p>
          <div className="flex gap-4">
            <Link
              to="/projects"
              className="bg-emerald-600 hover:bg-emerald-500 text-white font-mono px-6 py-3 rounded flex items-center gap-2 transition-all active:scale-95 cursor-pointer group"
            >
              VIEW SHIPMENTS{" "}
              <ChevronRight
                size={16}
                className="group-hover:translate-x-1 transition-transform"
              />
            </Link>
            <a
              href={CV_LINK}
              target="_blank"
              rel="noreferrer"
              className="border border-slate-600 hover:border-white text-slate-300 hover:text-white font-mono px-6 py-3 rounded flex items-center gap-2 transition-all cursor-pointer"
            >
              DOWNLOAD CV <Download size={16} />
            </a>
          </div>
        </div>
      </div>

      <Pipeline />

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {stats.map((stat, idx) => (
          <motion.div
            key={idx}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 * idx, duration: 0.4 }}
            whileHover={{ borderColor: "rgba(16, 185, 129, 0.3)", y: -2 }}
            className="bg-slate-900/30 border border-slate-800 p-4 rounded hover:border-emerald-500/30 transition-colors"
          >
            <div className="text-xs text-slate-500 font-mono mb-1">
              {stat.label}
            </div>
            <div className="text-2xl font-bold text-white">{stat.value}</div>
          </motion.div>
        ))}
      </div>
    </SectionWrapper>
  );
}
