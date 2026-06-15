"use client";

import { useEffect } from "react";

import { createTitleController } from "@/lib/tab-title";

/**
 * Mounts the tab-blur title easter egg. DOM-only (no canvas), so it sits
 * beside CustomCursor in the page root. Captures the real title on mount,
 * swaps it on window blur, restores on focus, and restores on unmount.
 */
export function TabTitleEffect() {
  useEffect(() => {
    const original = document.title;
    const controller = createTitleController(original, (title) => {
      document.title = title;
    });

    const onBlur = () => controller.onBlur();
    const onFocus = () => controller.onFocus();
    window.addEventListener("blur", onBlur);
    window.addEventListener("focus", onFocus);

    return () => {
      window.removeEventListener("blur", onBlur);
      window.removeEventListener("focus", onFocus);
      document.title = original;
    };
  }, []);

  return null;
}
