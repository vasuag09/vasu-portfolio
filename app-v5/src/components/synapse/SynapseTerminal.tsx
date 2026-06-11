"use client";

import { useEffect, useRef, useState } from "react";
import { useSynapse } from "@/hooks/useSynapse";
import { setGraphState } from "@/lib/graph-store";
import { afterNextPaint } from "@/lib/after-paint";
import { useGraphState } from "@/hooks/useGraphState";
import { useScrollLock } from "@/hooks/useScrollLock";

/**
 * Synapse terminal (Phase 6) — ported from v4 SynapsePanel, restyled to the
 * v5 token system (mono, phosphor green, no gradient chrome). Modal dialog:
 * Esc or backdrop closes, Tab is trapped inside, input autofocuses, history
 * autoscrolls. The Gemini call goes through /api/synapse — no key here.
 */

const SUGGESTIONS = [
  "What's Vasu's best project?",
  "Tell me about the 17-tool agent loop",
  "What's his open source work?",
  "What stack does he ship with?",
];

export function SynapseTerminal() {
  const { synapseOpen } = useGraphState();
  const { messages, isProcessing, send } = useSynapse();
  const [input, setInput] = useState("");
  const panelRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const endRef = useRef<HTMLDivElement>(null);
  const restoreFocusRef = useRef<HTMLElement | null>(null);

  // Deferred past the next paint: unmount + scroll-unlock relayout must not
  // land inside the interaction (Phase-8 INP finding, see after-paint.ts).
  const close = () => afterNextPaint(() => setGraphState({ synapseOpen: false }));

  // Freeze the page while the terminal is open (wheel over the terminal
  // must scroll the history, not the document behind it).
  useScrollLock(synapseOpen);

  // Focus management + Esc + Tab trap.
  useEffect(() => {
    if (!synapseOpen) return;
    restoreFocusRef.current = document.activeElement as HTMLElement;
    inputRef.current?.focus();

    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        close();
        return;
      }
      if (event.key !== "Tab") return;
      const focusables = panelRef.current?.querySelectorAll<HTMLElement>(
        "button, input, [href]",
      );
      if (!focusables?.length) return;
      const first = focusables[0];
      const last = focusables[focusables.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("keydown", onKey);
      restoreFocusRef.current?.focus();
    };
  }, [synapseOpen]);

  // Autoscroll to the latest message.
  useEffect(() => {
    endRef.current?.scrollIntoView({ block: "end" });
  }, [messages, isProcessing]);

  if (!synapseOpen) return null;

  const showSuggestions = messages.length <= 1 && !isProcessing;

  const onSubmit = (event: React.FormEvent) => {
    event.preventDefault();
    void send(input);
    setInput("");
  };

  return (
    <div
      className="fixed inset-0 flex items-end justify-center p-0 md:items-center md:justify-end md:p-8"
      style={{ zIndex: "var(--z-overlay)", background: "oklch(8% 0.02 250 / 0.7)" }}
      onClick={close}
      role="presentation"
    >
      <div
        ref={panelRef}
        role="dialog"
        aria-modal="true"
        aria-label="Synapse AI terminal"
        onClick={(event) => event.stopPropagation()}
        className="flex h-[75svh] w-full flex-col border md:h-[70vh] md:max-h-[600px] md:w-[440px]"
        style={{ background: "var(--bg-elevated)", borderColor: "var(--border)" }}
      >
        {/* Header */}
        <div
          className="flex items-center justify-between border-b px-5 py-3"
          style={{ borderColor: "var(--border)" }}
        >
          <div className="flex items-center gap-3">
            <span
              aria-hidden="true"
              className="block h-2.5 w-2.5 rounded-full"
              style={{
                background: "var(--accent)",
                boxShadow: "0 0 10px var(--accent-glow)",
              }}
            />
            <div>
              <h2 className="text-[length:var(--text-sm)] font-bold">SYNAPSE</h2>
              <p
                className="text-[length:var(--text-xs)] tracking-[var(--tracking-wide)]"
                style={{ color: "var(--text-faint)" }}
              >
                neural intelligence link
              </p>
            </div>
          </div>
          <button
            type="button"
            onClick={close}
            aria-label="Close Synapse"
            className="cursor-pointer rounded border px-2 py-0.5 text-[length:var(--text-sm)] transition-colors duration-[var(--duration-fast)] hover:border-[var(--border-active)]"
            style={{ borderColor: "var(--border)", color: "var(--text-muted)" }}
          >
            esc
          </button>
        </div>

        {/* History */}
        <div data-lenis-prevent className="flex-1 overflow-y-auto overscroll-contain px-5 py-4">
          <ul className="flex flex-col gap-3" role="list">
            {messages.map((message, index) => (
              <li key={index} className="flex gap-3">
                <span
                  className="shrink-0 select-none text-[length:var(--text-sm)]"
                  style={{
                    color:
                      message.role === "user"
                        ? "var(--accent-bright)"
                        : "var(--text-faint)",
                  }}
                >
                  {message.role === "user" ? ">" : "::"}
                </span>
                <p
                  className="text-[length:var(--text-sm)] leading-[var(--leading-body)]"
                  style={{
                    color:
                      message.role === "user" ? "var(--text)" : "var(--text-muted)",
                  }}
                >
                  {message.text}
                </p>
              </li>
            ))}
          </ul>

          {isProcessing ? (
            <p
              role="status"
              className="mt-3 animate-pulse text-[length:var(--text-xs)] tracking-[var(--tracking-wide)]"
              style={{ color: "var(--accent)" }}
            >
              processing signal…
            </p>
          ) : null}

          {showSuggestions ? (
            <div className="mt-4">
              <p
                className="text-[length:var(--text-xs)] tracking-[var(--tracking-wide)] uppercase"
                style={{ color: "var(--text-faint)" }}
              >
                pre-trained prompts
              </p>
              <ul className="mt-2 flex flex-col gap-1" role="list">
                {SUGGESTIONS.map((question) => (
                  <li key={question}>
                    <button
                      type="button"
                      onClick={() => void send(question)}
                      className="cursor-pointer text-left text-[length:var(--text-sm)] underline-offset-4 hover:underline"
                      style={{ color: "var(--text-muted)" }}
                    >
                      {question}
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          ) : null}
          <div ref={endRef} />
        </div>

        {/* Input */}
        <form
          onSubmit={onSubmit}
          className="flex items-center gap-3 border-t px-5 py-3"
          style={{ borderColor: "var(--border)" }}
        >
          <span aria-hidden="true" style={{ color: "var(--accent)" }}>
            &gt;
          </span>
          <input
            ref={inputRef}
            type="text"
            value={input}
            onChange={(event) => setInput(event.target.value)}
            disabled={isProcessing}
            aria-label="Ask Synapse"
            placeholder={isProcessing ? "processing…" : "ask anything…"}
            maxLength={2000}
            className="flex-1 bg-transparent outline-none text-[length:var(--text-sm)] disabled:opacity-50"
            style={{ color: "var(--text)" }}
          />
          <button
            type="submit"
            disabled={isProcessing || !input.trim()}
            aria-label="Send message"
            className="cursor-pointer rounded border px-3 py-1 text-[length:var(--text-sm)] transition-colors duration-[var(--duration-fast)] hover:border-[var(--border-active)] disabled:cursor-not-allowed disabled:opacity-40"
            style={{ borderColor: "var(--border)", color: "var(--accent-bright)" }}
          >
            send
          </button>
        </form>
      </div>
    </div>
  );
}
