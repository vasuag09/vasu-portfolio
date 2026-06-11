"use client";

import { useEffect, useRef, useState } from "react";
import { sections } from "@/data/sections-v5";
import { scrollToSection } from "@/lib/scroll-to-section";
import { sectionIndexForKey } from "@/lib/nav-keys";
import type { SectionId } from "@/data/types";

const SECTION_IDS = new Set<string>(sections.map((s) => s.id));

/**
 * Chapter rail (AWARD-RESEARCH §3.10): dots + labels, keyboard accessible,
 * `?section=` deep links. Deep links jump to the END state (no replay);
 * active state tracks the section crossing the viewport center.
 */
export function ChapterNav() {
  const [active, setActive] = useState<SectionId>(sections[0].id);
  const hasInteracted = useRef(false);

  // Deep link: land directly on the requested chapter, then keep the URL in sync.
  useEffect(() => {
    const requested = new URLSearchParams(window.location.search).get("section");
    if (requested && SECTION_IDS.has(requested)) {
      // After first paint so layout (and Lenis) exist.
      requestAnimationFrame(() => scrollToSection(requested, true));
    }
  }, []);

  // Number keys 1–5 jump chapters (Phase 10). Damped flight, same path as
  // dot clicks. Suppressed while typing (Synapse input), while an overlay
  // dialog owns the screen, and during the boot ritual (whose skip handler
  // also listens on window — without this guard one keypress would both
  // skip boot AND jump).
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.metaKey || event.ctrlKey || event.altKey) return;
      if (document.documentElement.hasAttribute("data-boot")) return;
      const target = event.target as HTMLElement | null;
      if (target?.closest('input, textarea, select, [contenteditable="true"]')) return;
      if (document.querySelector('[role="dialog"]')) return;
      const index = sectionIndexForKey(event.key, sections.length);
      if (index === null) return;
      scrollToSection(sections[index].id, false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  // Active chapter = section crossing the viewport's vertical center.
  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (!entry.isIntersecting) continue;
          hasInteracted.current = true;
          setActive(entry.target.id as SectionId); // React bails out on same id
        }
      },
      { rootMargin: "-50% 0px -50% 0px" },
    );
    document
      .querySelectorAll("[data-chapter]")
      .forEach((el) => observer.observe(el));
    return () => observer.disconnect();
  }, []);

  // URL reflects the active chapter — but never before the first observation,
  // so a clean load of "/" stays "/" until the user actually scrolls.
  useEffect(() => {
    if (!hasInteracted.current) return;
    const url = new URL(window.location.href);
    url.searchParams.set("section", active);
    window.history.replaceState(null, "", url);
  }, [active]);

  return (
    <nav
      aria-label="Chapters"
      className="fixed right-5 top-1/2 -translate-y-1/2 hidden md:block"
      style={{ zIndex: "var(--z-nav)" }}
    >
      <ol className="flex flex-col gap-4">
        {sections.map((section) => {
          const isActive = section.id === active;
          return (
            <li key={section.id}>
              <button
                type="button"
                onClick={() => scrollToSection(section.id, false)}
                aria-current={isActive ? "true" : undefined}
                aria-label={`Go to ${section.label}`}
                className="group flex items-center justify-end gap-3 cursor-pointer"
              >
                <span
                  className="text-[length:var(--text-xs)] tracking-[var(--tracking-wide)] uppercase opacity-0 transition-opacity duration-[var(--duration-fast)] group-hover:opacity-100 group-focus-visible:opacity-100"
                  style={{
                    color: isActive ? "var(--accent)" : "var(--text-muted)",
                    opacity: isActive ? 1 : undefined,
                  }}
                >
                  {section.label}
                </span>
                <span
                  aria-hidden="true"
                  className="block rounded-full transition-all duration-[var(--duration-normal)]"
                  style={{
                    width: isActive ? "10px" : "6px",
                    height: isActive ? "10px" : "6px",
                    background: isActive ? "var(--accent)" : "var(--text-faint)",
                    boxShadow: isActive
                      ? "0 0 8px var(--accent-glow)"
                      : "none",
                  }}
                />
              </button>
            </li>
          );
        })}
      </ol>
    </nav>
  );
}
