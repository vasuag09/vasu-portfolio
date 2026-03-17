import { useEffect } from "react";

/**
 * Sets the document title dynamically based on the current view.
 */
export function useDocumentTitle(title) {
  useEffect(() => {
    const base = "VASU_OS";
    document.title = title ? `${title} | ${base}` : base;
    return () => {
      document.title = base;
    };
  }, [title]);
}
