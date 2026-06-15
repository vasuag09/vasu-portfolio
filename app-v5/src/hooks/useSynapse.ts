"use client";

import { useCallback, useState } from "react";
import type { SynapseMessage } from "@/lib/synapse-validation";

const WELCOME: SynapseMessage = {
  role: "model",
  text: "Neural link established. Ask me anything about Vasu's work — projects, stack, the production stories.",
};

/** Chat state + transport for the Synapse terminal. */
export function useSynapse() {
  const [messages, setMessages] = useState<SynapseMessage[]>([WELCOME]);
  const [isProcessing, setIsProcessing] = useState(false);

  const send = useCallback(
    async (question: string) => {
      const trimmed = question.trim();
      if (!trimmed || isProcessing) return;

      const history = [...messages, { role: "user" as const, text: trimmed }];
      setMessages(history);
      setIsProcessing(true);
      try {
        const response = await fetch("/api/synapse", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          // Keep the wire payload inside the server's history cap.
          body: JSON.stringify({ messages: history.slice(-12) }),
        });
        const data = (await response.json()) as {
          answer?: string;
          error?: string;
        };
        const reply = response.ok
          ? (data.answer ?? "Error: neural link unstable.")
          : `Error: ${data.error ?? "AI core offline."}`;
        setMessages([...history, { role: "model", text: reply }]);
      } catch {
        setMessages([
          ...history,
          { role: "model", text: "Error: AI core offline. Try again shortly." },
        ]);
      } finally {
        setIsProcessing(false);
      }
    },
    [messages, isProcessing],
  );

  return { messages, isProcessing, send };
}
