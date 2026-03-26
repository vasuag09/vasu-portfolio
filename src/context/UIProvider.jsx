import React, { useState, useMemo, useCallback } from "react";
import { UIContext } from "./UIContext.js";

/**
 * UIProvider manages global UI state (modals, retro mode, tour, effects).
 * Terminal state lives in TerminalProvider to avoid re-rendering the
 * entire app tree on every terminal keystroke.
 */
export const UIProvider = ({ children }) => {
  const [isRetro, setIsRetro] = useState(false);
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const [isTerminalOpen, setIsTerminalOpen] = useState(false);
  const [previewProject, setPreviewProject] = useState(null);
  const [tourStep, setTourStep] = useState(0);
  const [isShortcutsOpen, setIsShortcutsOpen] = useState(false);
  const [soundEnabled, setSoundEnabled] = useState(
    () => typeof window !== "undefined" && localStorage.getItem("soundEnabled") === "true"
  );
  const [reducedEffects, setReducedEffects] = useState(
    () => typeof window !== "undefined" && localStorage.getItem("reducedEffects") === "true"
  );

  // Persist reduced effects preference
  const toggleReducedEffects = useCallback((updater) => {
    setReducedEffects((prev) => {
      const next = typeof updater === "function" ? updater(prev) : updater;
      localStorage.setItem("reducedEffects", String(next));
      return next;
    });
  }, []);

  const toggleSound = useCallback((updater) => {
    setSoundEnabled((prev) => {
      const next = typeof updater === "function" ? updater(prev) : updater;
      localStorage.setItem("soundEnabled", String(next));
      return next;
    });
  }, []);

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
    isShortcutsOpen,
    setIsShortcutsOpen,
    reducedEffects,
    setReducedEffects: toggleReducedEffects,
    soundEnabled,
    setSoundEnabled: toggleSound,
  }), [isRetro, isMobileMenuOpen, isTerminalOpen, previewProject, tourStep, isShortcutsOpen, reducedEffects, toggleReducedEffects, soundEnabled, toggleSound]);

  return <UIContext.Provider value={value}>{children}</UIContext.Provider>;
};
