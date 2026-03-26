import React, { useState } from "react";
import { motion } from "framer-motion";
import { Send, Loader2 } from "lucide-react";

/**
 * Contact form — uses Formspree (or can be swapped for any endpoint).
 * Falls back to a mailto link if no endpoint is configured.
 */

const FORMSPREE_ENDPOINT = "https://formspree.io/f/mvzvodzv"; // Replace with your Formspree endpoint

export default function ContactForm() {
  const [formState, setFormState] = useState({ name: "", email: "", message: "" });
  const [status, setStatus] = useState("idle"); // idle | sending | sent | error

  const handleSubmit = async (e) => {
    e.preventDefault();
    setStatus("sending");

    try {
      const res = await fetch(FORMSPREE_ENDPOINT, {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "application/json" },
        body: JSON.stringify(formState),
      });

      if (res.ok) {
        setStatus("sent");
        setFormState({ name: "", email: "", message: "" });
      } else {
        setStatus("error");
      }
    } catch {
      setStatus("error");
    }
  };

  if (status === "sent") {
    return (
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        className="glass-card-static p-6 text-center"
      >
        <div className="w-12 h-12 rounded-full bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center mx-auto mb-4">
          <Send size={18} className="text-emerald-400" />
        </div>
        <h4 className="text-sm font-semibold text-white mb-2" style={{ fontFamily: "var(--font-display)" }}>
          Signal Transmitted
        </h4>
        <p className="text-xs text-slate-500 font-mono">
          Your message has been sent. I'll respond within 24 hours.
        </p>
      </motion.div>
    );
  }

  return (
    <div className="glass-card-static p-6">
      <h4 className="text-[10px] font-mono text-cyan-500/60 tracking-[0.2em] mb-4">
        SEND SIGNAL
      </h4>
      <form onSubmit={handleSubmit} className="space-y-3">
        <div className="grid md:grid-cols-2 gap-3">
          <input
            type="text"
            placeholder="Name"
            required
            value={formState.name}
            onChange={(e) => setFormState((s) => ({ ...s, name: e.target.value }))}
            className="bg-[rgba(255,255,255,0.03)] border border-[rgba(255,255,255,0.06)] rounded-lg px-4 py-2.5 text-sm text-white placeholder-slate-600 outline-none focus:border-cyan-500/30 transition-colors w-full"
          />
          <input
            type="email"
            placeholder="Email"
            required
            value={formState.email}
            onChange={(e) => setFormState((s) => ({ ...s, email: e.target.value }))}
            className="bg-[rgba(255,255,255,0.03)] border border-[rgba(255,255,255,0.06)] rounded-lg px-4 py-2.5 text-sm text-white placeholder-slate-600 outline-none focus:border-cyan-500/30 transition-colors w-full"
          />
        </div>
        <textarea
          placeholder="Your message..."
          required
          rows={4}
          value={formState.message}
          onChange={(e) => setFormState((s) => ({ ...s, message: e.target.value }))}
          className="bg-[rgba(255,255,255,0.03)] border border-[rgba(255,255,255,0.06)] rounded-lg px-4 py-2.5 text-sm text-white placeholder-slate-600 outline-none focus:border-cyan-500/30 transition-colors w-full resize-none"
        />
        <div className="flex items-center justify-between">
          {status === "error" && (
            <p className="text-[10px] font-mono text-red-400/70">
              Transmission failed — try email instead
            </p>
          )}
          <button
            type="submit"
            disabled={status === "sending"}
            className="ml-auto flex items-center gap-2 bg-gradient-to-r from-cyan-600 to-cyan-500 hover:from-cyan-500 hover:to-cyan-400 text-white font-mono px-5 py-2.5 rounded-lg transition-all cursor-pointer text-sm disabled:opacity-50 disabled:cursor-not-allowed shadow-[0_0_20px_rgba(0,212,255,0.15)]"
          >
            {status === "sending" ? <Loader2 size={14} className="animate-spin" /> : <Send size={14} />}
            {status === "sending" ? "SENDING..." : "TRANSMIT"}
          </button>
        </div>
      </form>
    </div>
  );
}
