import React, { useState, useMemo } from "react";
import { UIContext } from "./UIContext.js";

/**
 * UIProvider manages global UI state (modals, retro mode, tour).
 * Terminal state lives in TerminalProvider to avoid re-rendering the
 * entire app tree on every terminal keystroke.
 */
export const UIProvider = ({ children }) => {
  const [isRetro, setIsRetro] = useState(false);
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const [isTerminalOpen, setIsTerminalOpen] = useState(false);
  const [previewProject, setPreviewProject] = useState(null);
  const [tourStep, setTourStep] = useState(0);

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
  }), [isRetro, isMobileMenuOpen, isTerminalOpen, previewProject, tourStep]);

  return <UIContext.Provider value={value}>{children}</UIContext.Provider>;
};
