import React from "react";

export default function RetroOverlay() {
  return (
    <div className="pointer-events-none fixed inset-0 z-[60] overflow-hidden h-full w-full bg-[linear-gradient(rgba(18,16,16,0)_50%,rgba(0,0,0,0.25)_50%),linear-gradient(90deg,rgba(255,0,0,0.06),rgba(0,255,0,0.02),rgba(0,0,255,0.06))] bg-[length:100%_2px,6px_100%] opacity-[0.15] animate-retro-flicker"></div>
  );
}
