import React, { useMemo } from "react";
import { useNavigate } from "react-router-dom";
import { useTerminal } from "../hooks/useTerminal";
import { TerminalContext } from "./TerminalContext";
import { useUI } from "../hooks/useUI";

/**
 * TerminalProvider manages terminal-specific state separately from UI state.
 * This prevents terminal keystrokes from re-rendering the entire app tree.
 */
export const TerminalProvider = ({ children }) => {
  const navigate = useNavigate();
  const { isRetro, setIsRetro, setIsTerminalOpen } = useUI();

  const terminal = useTerminal({
    setExpandedProject: (project) => {
      if (project) navigate(`/projects/${project.alias}`);
    },
    setIsTerminalOpen,
    isRetro,
    setIsRetro,
  });

  const value = useMemo(() => terminal, [terminal]);

  return <TerminalContext.Provider value={value}>{children}</TerminalContext.Provider>;
};
