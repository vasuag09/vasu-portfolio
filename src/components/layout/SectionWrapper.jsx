import React from "react";
import { motion } from "framer-motion";

/**
 * SectionWrapper provides a consistent entrance and exit animation for all sections.
 */
export default function SectionWrapper({ children, className = "", id }) {
  return (
    <motion.div
      key={id}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      transition={{ duration: 0.4, ease: "easeOut" }}
      className={className}
    >
      {children}
    </motion.div>
  );
}
