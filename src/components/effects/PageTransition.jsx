import React from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useLocation } from "react-router-dom";

/**
 * Cinematic page transitions — wraps route content with
 * smooth enter/exit animations. Uses location key for AnimatePresence.
 */

const pageVariants = {
  initial: {
    opacity: 0,
    y: 20,
    filter: "blur(8px)",
  },
  enter: {
    opacity: 1,
    y: 0,
    filter: "blur(0px)",
    transition: {
      duration: 0.5,
      ease: [0.23, 1, 0.32, 1],
      staggerChildren: 0.06,
    },
  },
  exit: {
    opacity: 0,
    y: -10,
    filter: "blur(4px)",
    transition: {
      duration: 0.3,
      ease: [0.23, 1, 0.32, 1],
    },
  },
};

export default function PageTransition({ children }) {
  const location = useLocation();

  return (
    <AnimatePresence mode="wait">
      <motion.div
        key={location.pathname}
        variants={pageVariants}
        initial="initial"
        animate="enter"
        exit="exit"
      >
        {children}
      </motion.div>
    </AnimatePresence>
  );
}
