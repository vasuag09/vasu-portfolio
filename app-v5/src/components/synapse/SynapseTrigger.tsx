"use client";

import { setGraphState } from "@/lib/graph-store";

interface SynapseTriggerProps {
  /** Primary = the signature CTA treatment (hero/contact). */
  primary?: boolean;
  className?: string;
}

/** DOM trigger for the Synapse terminal (the canvas node is pointer-inert). */
export function SynapseTrigger({ primary = false, className }: SynapseTriggerProps) {
  return (
    <button
      type="button"
      onClick={() => setGraphState({ synapseOpen: true })}
      aria-haspopup="dialog"
      className={`group inline-flex cursor-pointer items-center rounded border transition-colors duration-[var(--duration-fast)] hover:border-[var(--border-active)] ${
        primary
          ? "gap-3 px-6 py-3 text-[length:var(--text-base)] tracking-[var(--tracking-wide)] uppercase"
          : "gap-2 px-4 py-2 text-[length:var(--text-sm)]"
      } ${className ?? ""}`}
      style={{
        borderColor: primary ? "var(--border-active)" : "var(--border)",
        color: "var(--accent-bright)",
        animation: primary ? "cta-pulse 2.6s var(--ease-in-out-soft) infinite" : undefined,
        background: primary ? "oklch(58% 0.18 150 / 0.07)" : undefined,
      }}
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
