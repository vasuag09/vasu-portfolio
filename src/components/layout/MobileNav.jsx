import React from "react";
import { Menu, X } from "lucide-react";
import { profile } from "../../data/profile";
import { useUI } from "../../hooks/useUI";

export default function MobileNav() {
  const { isMobileMenuOpen, setIsMobileMenuOpen } = useUI();
  
  return (
    <div className="md:hidden p-4 border-b border-slate-800 flex justify-between items-center bg-slate-950/90 backdrop-blur sticky top-0 z-50">
      <div className="font-mono font-bold text-emerald-500">
        ./{profile.name.replace(" ", "_")}
      </div>
      <button 
        onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
        aria-label="Toggle mobile menu"
        className="text-slate-400 hover:text-white transition-colors p-1"
      >
        {isMobileMenuOpen ? <X size={24} /> : <Menu size={24} />}
      </button>
    </div>
  );
}
