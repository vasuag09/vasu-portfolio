import { useState, useRef, useEffect, useCallback } from "react";
import { processCommand, generateAiResponse } from "../utils/terminal-commands";

/**
 * Manages terminal/Synapse state: history, input, AI processing, and command handling.
 */
export function useTerminal({ setExpandedProject, setIsTerminalOpen, isRetro, setIsRetro }) {
  const [terminalInput, setTerminalInput] = useState("");
  const [terminalHistory, setTerminalHistory] = useState([
    {
      type: "system",
      content: "Synapse online. Ask me anything about Vasu's work, projects, or skills.",
    },
  ]);
  const [isAiProcessing, setIsAiProcessing] = useState(false);
  const terminalEndRef = useRef(null);

  useEffect(() => {
    if (terminalEndRef.current) {
      terminalEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [terminalHistory]);

  const processInput = useCallback(
    async (command) => {
      if (!command.trim()) return;

      setTerminalHistory((prev) => [
        ...prev,
        { type: "user", content: command },
      ]);

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
        setIsAiProcessing(true);
        try {
          const aiResponse = await generateAiResponse(command);
          setTerminalHistory((prev) => [
            ...prev,
            { type: "ai", content: aiResponse },
          ]);
        } finally {
          setIsAiProcessing(false);
        }
      }
    },
    [setExpandedProject, setIsTerminalOpen, isRetro, setIsRetro],
  );

  const handleCommand = useCallback(
    (e) => {
      if (e.key !== "Enter") return;
      const command = terminalInput.trim();
      if (!command) return;
      setTerminalInput("");
      processInput(command);
    },
    [terminalInput, processInput],
  );

  // Direct submit for suggestion chips (bypasses input state)
  const submitQuery = useCallback(
    (query) => {
      setTerminalInput("");
      processInput(query);
    },
    [processInput],
  );

  return {
    terminalInput,
    setTerminalInput,
    terminalHistory,
    isAiProcessing,
    terminalEndRef,
    handleCommand,
    submitQuery,
  };
}
