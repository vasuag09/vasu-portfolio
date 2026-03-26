import React from "react";
import { motion, AnimatePresence } from "framer-motion";
import { X, Keyboard } from "lucide-react";
import { NAVIGATION_ITEMS } from "../../data/navigation";

const SHORTCUTS = [
  { keys: ["⌘", "K"], desc: "Toggle Synapse AI" },
  { keys: ["Esc"], desc: "Close overlays / Go back" },
  { keys: ["?"], desc: "Toggle this shortcuts panel" },
  ...NAVIGATION_ITEMS.map((item) => ({
    keys: [item.shortcut],
    desc: `Navigate to ${item.label}`,
  })),
];

/**
 * Keyboard shortcuts overlay — toggled with '?'
 */
export default function KeyboardShortcuts({ isOpen, onClose }) {
  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm"
          onClick={onClose}
        >
          <motion.div
            initial={{ scale: 0.95, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.95, opacity: 0 }}
            className="bg-[rgba(6,8,15,0.95)] backdrop-blur-xl border border-[rgba(0,212,255,0.1)] rounded-2xl p-6 w-full max-w-sm shadow-[0_0_60px_rgba(0,212,255,0.05)]"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center gap-3">
                <Keyboard size={16} className="text-cyan-400/70" />
                <h3
                  className="text-sm font-semibold text-white"
                  style={{ fontFamily: "var(--font-display)" }}
                >
                  Keyboard Shortcuts
                </h3>
              </div>
              <button
                onClick={onClose}
                className="w-8 h-8 rounded-lg flex items-center justify-center text-slate-500 hover:text-white hover:bg-[rgba(255,255,255,0.05)] transition-all cursor-pointer"
                aria-label="Close shortcuts"
              >
                <X size={16} />
              </button>
            </div>

            <div className="space-y-2">
              {SHORTCUTS.map((shortcut, idx) => (
                <div
                  key={idx}
                  className="flex items-center justify-between py-1.5"
                >
                  <span className="text-sm text-slate-400">{shortcut.desc}</span>
                  <div className="flex gap-1">
                    {shortcut.keys.map((key, i) => (
                      <kbd
                        key={i}
                        className="px-2 py-0.5 text-[10px] font-mono bg-[rgba(255,255,255,0.04)] border border-[rgba(255,255,255,0.08)] rounded text-slate-300 min-w-[24px] text-center"
                      >
                        {key}
                      </kbd>
                    ))}
                  </div>
                </div>
              ))}
            </div>

            <p className="text-[10px] font-mono text-slate-700 mt-6 text-center">
              Press <kbd className="px-1.5 py-0.5 bg-[rgba(255,255,255,0.04)] border border-[rgba(255,255,255,0.06)] rounded text-slate-500">?</kbd> to toggle
            </p>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
