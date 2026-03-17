import React from "react";
import { motion } from "framer-motion";
import { Link, useLocation } from "react-router-dom";

export default function NavButton({ to, icon: Icon, label, shortcut, onClick }) {
  const location = useLocation();

  const isActive = to === "/"
    ? location.pathname === "/"
    : location.pathname.startsWith(to);

  return (
    <Link
      to={to}
      onClick={onClick}
      aria-current={isActive ? "page" : undefined}
      className={`w-full flex items-center justify-between px-5 py-3.5 text-sm font-mono transition-all duration-300 relative group border-l-2 ${
        isActive
          ? "border-emerald-500 bg-emerald-500/10 text-emerald-400 shadow-[inset_4px_0_10px_-4px_rgba(16,185,129,0.2)]"
          : "border-transparent text-slate-400 hover:text-emerald-300 hover:bg-slate-800/40 hover:border-slate-700"
      }`}
    >
      {isActive && (
        <motion.div
          layoutId="active-indicator"
          className="absolute inset-0 bg-emerald-500/5 pointer-events-none"
          initial={false}
          transition={{ type: "spring", stiffness: 300, damping: 30 }}
        />
      )}
      <div className="flex items-center gap-3 relative z-10">
        <Icon 
          size={18} 
          className={`transition-transform duration-300 ${isActive ? "scale-110" : "group-hover:scale-110"}`} 
        />
        <span className={isActive ? "font-bold" : ""}>{label.toUpperCase()}</span>
      </div>
      {shortcut && (
        <span className="text-[11px] opacity-50 border border-slate-700 px-1 rounded hidden lg:block">
          {shortcut}
        </span>
      )}
    </Link>
  );
}
