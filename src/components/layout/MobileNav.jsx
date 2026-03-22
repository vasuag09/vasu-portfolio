import React from "react";
import { useLocation, useNavigate } from "react-router-dom";

/**
 * Mobile navigation — horizontal node chain at the bottom of the screen.
 */

const LAYERS = [
  { path: "/", label: "IN" },
  { path: "/projects", label: "H1" },
  { path: "/skills", label: "H2" },
  { path: "/research", label: "H3" },
  { path: "/about", label: "OUT" },
];

function getActiveIndex(pathname) {
  if (pathname === "/") return 0;
  if (pathname.startsWith("/projects")) return 1;
  if (pathname.startsWith("/skills")) return 2;
  if (pathname.startsWith("/research")) return 3;
  if (pathname.startsWith("/about")) return 4;
  return -1;
}

export default function MobileNav() {
  const location = useLocation();
  const navigate = useNavigate();
  const activeIdx = getActiveIndex(location.pathname);

  return (
    <nav
      className="fixed bottom-0 left-0 right-0 z-40 md:hidden bg-[rgba(6,8,15,0.9)] backdrop-blur-lg border-t border-[rgba(0,212,255,0.08)]"
      aria-label="Mobile navigation"
    >
      <div className="flex items-center justify-around py-3 px-4">
        {LAYERS.map((layer, idx) => {
          const isActive = activeIdx === idx;

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
                  className={`text-[10px] font-mono ${
                    isActive ? "text-cyan-400" : "text-slate-500"
                  }`}
                >
                  {layer.label}
                </span>
              </button>
            </React.Fragment>
          );
        })}
      </div>
    </nav>
  );
}
