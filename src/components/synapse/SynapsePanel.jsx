import React, { useRef, useEffect } from "react";
import { motion } from "framer-motion";
import { X, Send } from "lucide-react";
import { useFocusTrap } from "../../hooks/useFocusTrap";
import MiniNetwork from "../canvas/MiniNetwork";

const SUGGESTIONS = [
  "What's Vasu's best project?",
  "What tech stack does he use?",
  "Tell me about his ML experience",
  "What makes him unique?",
];

/**
 * Synapse AI chat panel — slide-up conversational interface
 * that replaces the old terminal modal.
 */
export default function SynapsePanel({
  terminalInput,
  setTerminalInput,
  terminalHistory,
  isAiProcessing,
  terminalEndRef,
  handleCommand,
  submitQuery,
  onClose,
}) {
  const inputRef = useRef(null);
  const panelRef = useRef(null);

  useFocusTrap(panelRef, onClose);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const onSubmit = (e) => {
    e.preventDefault();
    handleCommand({ key: "Enter" });
  };

  const askSuggestion = (question) => {
    submitQuery(question);
  };

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 z-50 flex items-end md:items-center justify-center md:justify-end p-0 md:p-8 bg-black/60 backdrop-blur-sm"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-label="Synapse AI assistant"
    >
      <motion.div
        ref={panelRef}
        initial={{ y: 40, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        exit={{ y: 40, opacity: 0 }}
        transition={{ type: "spring", stiffness: 300, damping: 30 }}
        className="w-full md:w-[420px] h-[75vh] md:h-[70vh] md:max-h-[600px] bg-[rgba(6,8,15,0.95)] backdrop-blur-xl border border-[rgba(0,212,255,0.1)] md:rounded-2xl rounded-t-2xl flex flex-col shadow-[0_0_60px_rgba(0,212,255,0.05)] overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-[rgba(0,212,255,0.06)]">
          <div className="flex items-center gap-3">
            <div className="w-3 h-3 rounded-full bg-gradient-to-br from-cyan-400 to-purple-500 shadow-[0_0_8px_rgba(0,212,255,0.4)] animate-neural-breathe" />
            <div>
              <h3
                className="text-sm font-semibold text-white"
                style={{ fontFamily: "var(--font-display)" }}
              >
                Synapse
              </h3>
              <p className="text-[10px] font-mono text-cyan-500/50">
                Neural Intelligence Link
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            aria-label="Close Synapse"
            className="w-8 h-8 rounded-lg flex items-center justify-center text-slate-500 hover:text-white hover:bg-[rgba(255,255,255,0.05)] transition-all cursor-pointer"
          >
            <X size={16} />
          </button>
        </div>

        {/* Messages */}
        <div
          className="flex-1 overflow-y-auto px-5 py-4 space-y-4 custom-scrollbar"
          onClick={() => inputRef.current?.focus()}
        >
          {terminalHistory.map((entry, idx) => (
            <div key={idx} className="animate-float-up" style={{ animationDelay: `${idx * 0.05}s` }}>
              {entry.type === "user" ? (
                <div className="flex justify-end">
                  <div className="bg-[rgba(0,212,255,0.08)] border border-[rgba(0,212,255,0.12)] rounded-2xl rounded-br-sm px-4 py-2.5 max-w-[85%]">
                    <p className="text-sm text-white">{entry.content}</p>
                  </div>
                </div>
              ) : (
                <div className="flex gap-3">
                  <div className="w-2 h-2 rounded-full bg-cyan-500/40 mt-2.5 shrink-0" />
                  <div className="bg-[rgba(255,255,255,0.02)] border border-[rgba(255,255,255,0.04)] rounded-2xl rounded-bl-sm px-4 py-2.5 max-w-[85%]">
                    <p className="text-sm text-slate-300 leading-relaxed">
                      {entry.content}
                    </p>
                  </div>
                </div>
              )}
            </div>
          ))}

          {/* Processing indicator */}
          {isAiProcessing && (
            <div className="flex gap-3 items-center" role="status">
              <div className="w-2 h-2 rounded-full bg-cyan-500/40 shrink-0" />
              <div className="bg-[rgba(255,255,255,0.02)] border border-[rgba(255,255,255,0.04)] rounded-2xl px-4 py-3">
                <MiniNetwork active />
                <p className="text-[10px] font-mono text-cyan-500/40 mt-1">
                  Processing signal...
                </p>
              </div>
            </div>
          )}

          {/* Suggestions (show only when history is just the welcome messages) */}
          {terminalHistory.length <= 2 && !isAiProcessing && (
            <div className="space-y-2 pt-2">
              <p className="text-[10px] font-mono text-slate-600 tracking-wider">
                PRE-TRAINED PROMPTS
              </p>
              {SUGGESTIONS.map((q, idx) => (
                <button
                  key={idx}
                  onClick={() => askSuggestion(q)}
                  className="block w-full text-left text-sm text-slate-500 hover:text-cyan-300 px-3 py-2 rounded-lg hover:bg-[rgba(0,212,255,0.04)] transition-all cursor-pointer font-mono border border-transparent hover:border-[rgba(0,212,255,0.08)]"
                >
                  {q}
                </button>
              ))}
            </div>
          )}

          <div ref={terminalEndRef} />
        </div>

        {/* Input */}
        <form
          onSubmit={onSubmit}
          className="px-5 py-4 border-t border-[rgba(0,212,255,0.06)] flex items-center gap-3"
        >
          <input
            ref={inputRef}
            type="text"
            value={terminalInput}
            onChange={(e) => setTerminalInput(e.target.value)}
            disabled={isAiProcessing}
            aria-label="Ask Synapse"
            className="flex-1 bg-[rgba(255,255,255,0.03)] border border-[rgba(255,255,255,0.06)] rounded-xl px-4 py-2.5 text-sm text-white placeholder-slate-600 outline-none focus:border-cyan-500/30 transition-colors disabled:opacity-50"
            placeholder={
              isAiProcessing ? "Processing..." : "Ask anything..."
            }
          />
          <button
            type="submit"
            disabled={isAiProcessing || !terminalInput.trim()}
            className="w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-500/20 to-purple-500/20 border border-cyan-500/20 flex items-center justify-center text-cyan-400 hover:from-cyan-500/30 hover:to-purple-500/30 disabled:opacity-30 transition-all cursor-pointer disabled:cursor-not-allowed"
            aria-label="Send message"
          >
            <Send size={16} />
          </button>
        </form>
      </motion.div>
    </motion.div>
  );
}
