import { useState, useRef, useEffect, useCallback } from "react";
import { processCommand, generateAiResponse } from "../utils/terminal-commands";

/**
 * Manages terminal state: history, input, AI processing, and command handling.
 */
export function useTerminal({ setExpandedProject, setIsTerminalOpen, isRetro, setIsRetro }) {
  const [terminalInput, setTerminalInput] = useState("");
  const [terminalHistory, setTerminalHistory] = useState([
    { type: "system", content: "Welcome to VASU_OS v4.2.0 (AI Enabled)" },
    {
      type: "system",
      content: 'Type "./help" for commands or just ask me anything about Vasu.',
    },
  ]);
  const [isAiProcessing, setIsAiProcessing] = useState(false);
  const terminalEndRef = useRef(null);

  const apiKey = import.meta.env.VITE_GEMINI_API_KEY || "";

  useEffect(() => {
    if (terminalEndRef.current) {
      terminalEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [terminalHistory]);

  const handleCommand = useCallback(
    async (e) => {
      if (e.key !== "Enter") return;

      const command = terminalInput.trim();
      if (!command) return;

      setTerminalHistory((prev) => [
        ...prev,
        { type: "user", content: command },
      ]);
      setTerminalInput("");

      // Handle ./clear specially
      if (command === "./clear") {
        setTerminalHistory([]);
        return;
      }

      const output = processCommand(command, {
        setExpandedProject,
        setIsTerminalOpen,
        isRetro,
        setIsRetro,
      });

      if (output) {
        setTerminalHistory((prev) => [
          ...prev,
          ...output.map((line) => ({ type: "system", content: line })),
        ]);
      } else {
        // AI query
        setIsAiProcessing(true);
        try {
          const aiResponse = await generateAiResponse(command, apiKey);
          setTerminalHistory((prev) => [
            ...prev,
            { type: "ai", content: aiResponse },
          ]);
        } finally {
          setIsAiProcessing(false);
        }
      }
    },
    [terminalInput, setExpandedProject, setIsTerminalOpen, isRetro, setIsRetro, apiKey],
  );

  return {
    terminalInput,
    setTerminalInput,
    terminalHistory,
    isAiProcessing,
    terminalEndRef,
    handleCommand,
  };
}
