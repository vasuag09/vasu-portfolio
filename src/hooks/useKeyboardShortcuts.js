import { useEffect } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { SHORTCUT_ROUTES } from "../data/navigation";

/**
 * Global keyboard shortcuts:
 *  - Cmd/Ctrl+K: toggle terminal
 *  - Escape: close overlays
 *  - 1-6: navigate tabs (when terminal is closed)
 *  - ?: show keyboard shortcuts overlay
 */
export function useKeyboardShortcuts({
  isTerminalOpen,
  setIsTerminalOpen,
  setPreviewProject,
  setFilterTech,
  setIsMobileMenuOpen,
  setIsShortcutsOpen,
}) {
  const navigate = useNavigate();
  const location = useLocation();

  useEffect(() => {
    const handleKeyDown = (e) => {
      // Cmd/Ctrl + K => toggle terminal
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault();
        setIsTerminalOpen((prev) => !prev);
        return;
      }

      // ? => show keyboard shortcuts (only when not typing in input)
      if (e.key === "?" && !e.target.closest("input, textarea, [contenteditable]")) {
        e.preventDefault();
        setIsShortcutsOpen?.((prev) => !prev);
        return;
      }

      // Escape => close overlays / navigate back
      if (e.key === "Escape") {
        setPreviewProject(null);
        setFilterTech?.(null);
        setIsTerminalOpen(false);
        setIsMobileMenuOpen?.(false);
        setIsShortcutsOpen?.(false);

        // If on a sub-view, navigate back
        if (location.pathname.startsWith("/projects/")) {
          navigate("/projects");
        }
        if (location.pathname.startsWith("/blog/")) {
          navigate("/blog");
        }
        return;
      }

      // Number keys for tab navigation (only when terminal is closed)
      if (isTerminalOpen) return;
      if (e.altKey || e.metaKey || e.ctrlKey) return;
      if (e.target.closest("input, textarea, [contenteditable]")) return;

      if (SHORTCUT_ROUTES[e.key]) {
        setIsMobileMenuOpen?.(false);
        navigate(SHORTCUT_ROUTES[e.key]);
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [
    isTerminalOpen,
    navigate,
    location.pathname,
    setIsTerminalOpen,
    setPreviewProject,
    setFilterTech,
    setIsMobileMenuOpen,
    setIsShortcutsOpen,
  ]);
}
