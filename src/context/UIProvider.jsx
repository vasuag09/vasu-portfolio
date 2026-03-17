import React, { useState, useMemo } from "react";
import { useNavigate } from "react-router-dom";
import { useTerminal } from "../hooks/useTerminal";
import { UIContext } from "./UIContext.js";

/**
 * UIProvider manages the global UI state for the application.
 */
export const UIProvider = ({ children }) => {
  const navigate = useNavigate();
  // Global UI state
  const [isRetro, setIsRetro] = useState(false);
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const [isTerminalOpen, setIsTerminalOpen] = useState(false);
  const [previewProject, setPreviewProject] = useState(null);
  const [tourStep, setTourStep] = useState(0);

  // Terminal state management
  const terminal = useTerminal({
    setExpandedProject: (project) => {
      if (project) navigate(`/projects/${project.alias}`);
    },
    setIsTerminalOpen,
    isRetro,
    setIsRetro,
  });

  const value = useMemo(() => ({
    isRetro,
    setIsRetro,
    isMobileMenuOpen,
    setIsMobileMenuOpen,
    isTerminalOpen,
    setIsTerminalOpen,
    previewProject,
    setPreviewProject,
    tourStep,
    setTourStep,
    terminal,
  }), [isRetro, isMobileMenuOpen, isTerminalOpen, previewProject, tourStep, terminal]);

  return <UIContext.Provider value={value}>{children}</UIContext.Provider>;
};
