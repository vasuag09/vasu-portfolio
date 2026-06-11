"use client";

import { setGraphState } from "@/lib/graph-store";

/** DOM trigger for the Synapse terminal (the canvas node is pointer-inert). */
export function SynapseTrigger({ className }: { className?: string }) {
  return (
    <button
      type="button"
      onClick={() => setGraphState({ synapseOpen: true })}
      aria-haspopup="dialog"
      className={`group inline-flex cursor-pointer items-center gap-2 rounded border px-4 py-2 text-[length:var(--text-sm)] transition-colors duration-[var(--duration-fast)] hover:border-[var(--border-active)] ${className ?? ""}`}
      style={{ borderColor: "var(--border)", color: "var(--accent-bright)" }}
    >
      <span
        aria-hidden="true"
        className="block h-2 w-2 rounded-full transition-shadow duration-[var(--duration-fast)] group-hover:shadow-[0_0_8px_var(--accent-glow)]"
        style={{ background: "var(--accent)" }}
      />
      boot synapse
    </button>
  );
}
