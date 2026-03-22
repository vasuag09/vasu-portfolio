import { useEffect } from "react";
import { useNavigate, useLocation } from "react-router-dom";

/**
 * Global keyboard shortcuts:
 *  - Cmd/Ctrl+K: toggle terminal
 *  - Escape: close overlays
 *  - 1-4: navigate tabs (when terminal is closed)
 */
export function useKeyboardShortcuts({
  isTerminalOpen,
  setIsTerminalOpen,
  setPreviewProject,
  setFilterTech,
  setIsMobileMenuOpen,
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

      // Escape => close overlays / navigate back
      if (e.key === "Escape") {
        setPreviewProject(null);
        setFilterTech?.(null);
        setIsTerminalOpen(false);
        setIsMobileMenuOpen?.(false);

        // If on a sub-view, navigate back
        if (location.pathname.startsWith("/projects/")) {
          navigate("/projects");
        }
        return;
      }

      // Number keys for tab navigation (only when terminal is closed)
      if (isTerminalOpen) return;
      if (e.altKey || e.metaKey || e.ctrlKey) return;

      const routes = { "1": "/", "2": "/projects", "3": "/skills", "4": "/research", "5": "/about" };
      if (routes[e.key]) {
        setIsMobileMenuOpen?.(false);
        navigate(routes[e.key]);
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
  ]);
}
