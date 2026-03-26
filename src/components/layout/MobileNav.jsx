import React from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { NAVIGATION_ITEMS, getActiveLayer } from "../../data/navigation";

/**
 * Mobile navigation — horizontal node chain at the bottom of the screen.
 * Uses consolidated navigation data (no local duplication).
 */

// Only show main nav items in mobile bar (exclude blog)
const MOBILE_ITEMS = NAVIGATION_ITEMS.filter((item) => item.path !== "/blog");

export default function MobileNav() {
  const location = useLocation();
  const navigate = useNavigate();
  const activeIdx = MOBILE_ITEMS.findIndex((item) => {
    if (item.path === "/") return location.pathname === "/";
    return location.pathname.startsWith(item.path);
  });

  return (
    <nav
      className="fixed bottom-0 left-0 right-0 z-40 md:hidden bg-[rgba(6,8,15,0.9)] backdrop-blur-lg border-t border-[rgba(0,212,255,0.08)]"
      aria-label="Mobile navigation"
    >
      <div className="flex items-center justify-around py-3 px-4">
        {MOBILE_ITEMS.map((layer, idx) => {
          const isActive = activeIdx === idx;
          const Icon = layer.icon;

          return (
            <React.Fragment key={layer.path}>
              {idx > 0 && (
                <div
                  className={`flex-1 h-px mx-1 ${
                    (activeIdx >= idx - 1 && activeIdx <= idx)
                      ? "bg-gradient-to-r from-cyan-500/30 to-purple-500/30"
                      : "bg-slate-700/20"
                  }`}
                />
              )}

              <button
                onClick={() => navigate(layer.path)}
                className="relative flex flex-col items-center gap-1 cursor-pointer"
                aria-label={`Navigate to ${layer.label}`}
                aria-current={isActive ? "page" : undefined}
              >
                <div
                  className={`w-3 h-3 rounded-full transition-all duration-300 ${
                    isActive
                      ? "bg-cyan-400 shadow-[0_0_10px_rgba(0,212,255,0.5)]"
                      : "bg-slate-600"
                  }`}
                />
                <span
                  className={`text-[9px] font-mono ${
                    isActive ? "text-cyan-400" : "text-slate-500"
                  }`}
                >
                  {layer.shortLabel}
                </span>
              </button>
            </React.Fragment>
          );
        })}
      </div>
    </nav>
  );
}
