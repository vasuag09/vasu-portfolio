import React, { useEffect, useRef } from "react";
import { motion } from "framer-motion";
import { Terminal as TerminalIcon, Sparkles } from "lucide-react";
import { useFocusTrap } from "../../hooks/useFocusTrap";

export default function TerminalModal({
  terminalInput,
  setTerminalInput,
  terminalHistory,
  isAiProcessing,
  terminalEndRef,
  handleCommand,
  onClose,
}) {
  const inputRef = useRef(null);
  const modalRef = useRef(null);

  useFocusTrap(modalRef, onClose);

  // Auto-focus input on mount
  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const onFormSubmit = (e) => {
    e.preventDefault();
    handleCommand({ key: "Enter" });
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm cursor-pointer"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-label="Developer terminal"
    >
      <div
        ref={modalRef}
        className="bg-slate-950 border border-emerald-500/30 rounded-lg w-full max-w-3xl h-[60vh] flex flex-col shadow-2xl shadow-emerald-900/20 overflow-hidden font-mono text-sm cursor-auto"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="bg-slate-900 border-b border-slate-800 p-3 flex items-center justify-between">
          <div className="flex items-center gap-2 text-slate-400">
            <TerminalIcon size={16} aria-hidden="true" />
            <span>vasu_os_terminal</span>
            {isAiProcessing && (
              <span 
                className="text-xs text-emerald-500 animate-pulse flex items-center gap-1"
                role="status"
                aria-live="polite"
              >
                <Sparkles size={10} aria-hidden="true" /> AI PROCESSING
              </span>
            )}
          </div>
          <div className="flex gap-2">
            <div className="w-3 h-3 rounded-full bg-yellow-500/50" aria-hidden="true"></div>
            <div className="w-3 h-3 rounded-full bg-green-500/50" aria-hidden="true"></div>
            <button
              onClick={onClose}
              aria-label="Close terminal"
              className="w-3 h-3 rounded-full bg-red-500/50 hover:bg-red-500 transition-colors cursor-pointer focus:outline-none focus:ring-1 focus:ring-red-500"
            ></button>
          </div>
        </div>
        
        <div
          className="flex-1 p-4 overflow-y-auto space-y-2 custom-scrollbar"
          onClick={() => inputRef.current?.focus()}
          aria-live="polite"
          aria-relevant="additions"
        >
          {terminalHistory.map((entry, idx) => (
            <div
              key={idx}
              className={`${
                entry.type === "user"
                  ? "text-white"
                  : entry.type === "ai"
                    ? "text-emerald-300"
                    : "text-emerald-400/80"
              }`}
            >
              {entry.type === "user" ? (
                <div className="flex gap-2">
                  <span className="text-emerald-500" aria-hidden="true">➜</span>
                  <span className="sr-only">User input:</span>
                  <span>{entry.content}</span>
                </div>
              ) : entry.type === "ai" ? (
                <div className="pl-5 flex gap-2">
                  <Sparkles size={14} className="mt-0.5 shrink-0 opacity-70" aria-hidden="true" />{" "}
                  <span className="sr-only">AI response:</span>
                  <span>{entry.content}</span>
                </div>
              ) : (
                <div className="pl-5 whitespace-pre-wrap">{entry.content}</div>
              )}
            </div>
          ))}
          {isAiProcessing && (
            <div className="pl-5 text-emerald-500/50 animate-pulse" role="status">
              Thinking...
            </div>
          )}
          <div ref={terminalEndRef} />
        </div>

        <form 
          onSubmit={onFormSubmit}
          className="p-3 bg-slate-900 border-t border-slate-800 flex items-center gap-2"
        >
          <span className="text-emerald-500" aria-hidden="true">➜</span>
          <input
            ref={inputRef}
            type="text"
            value={terminalInput}
            onChange={(e) => setTerminalInput(e.target.value)}
            disabled={isAiProcessing}
            aria-label="Terminal input"
            className="bg-transparent border-none outline-none text-white flex-1 placeholder-slate-600 disabled:opacity-50"
            placeholder={
              isAiProcessing
                ? "VASU_OS is thinking..."
                : "Type ./help or ask a question..."
            }
          />
        </form>
      </div>
    </motion.div>
  );
}
