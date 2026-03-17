import React from "react";

export default function SkipToContent() {
  return (
    <a
      href="#main-content"
      className="sr-only focus:not-sr-only focus:fixed focus:top-4 focus:left-4 focus:z-[200] focus:bg-emerald-600 focus:text-white focus:px-4 focus:py-2 focus:rounded focus:text-sm focus:font-mono"
    >
      Skip to main content
    </a>
  );
}
