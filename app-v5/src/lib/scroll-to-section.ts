import { scrollState } from "./scroll-state";
import { headingScrollOffset } from "./scroll-landing";

/**
 * Shared chapter navigation (ChapterNav dots, number keys, deep links,
 * panel camera alignment). Smooth via Lenis when it exists; native jump
 * otherwise (reduced motion). Lands the chapter HEADING at the optical
 * headline line, not the section's top edge — Section.tsx publishes its
 * alignment via data-align and its heading as `${id}-heading`.
 */
export function scrollToSection(
  id: string,
  immediate: boolean,
  options: { force?: boolean } = {},
): void {
  const section = document.getElementById(id);
  if (!section) return;
  const align = section.dataset.align === "start" ? "start" : "center";
  const heading = document.getElementById(`${id}-heading`);
  const target = align === "start" ? section : (heading ?? section);
  const offset = headingScrollOffset(window.innerHeight, align);
  if (scrollState.lenis) {
    scrollState.lenis.scrollTo(target, {
      offset,
      duration: immediate ? 0 : 1.4,
      immediate,
      // force: animate even while lenis is stopped (modal scroll lock).
      force: options.force ?? false,
    });
  } else {
    const top = target.getBoundingClientRect().top + window.scrollY + offset;
    window.scrollTo({ top, behavior: "auto" });
  }
}

/** True when the section is already roughly centered in the viewport. */
export function isSectionInView(id: string): boolean {
  const el = document.getElementById(id);
  if (!el) return false;
  const rect = el.getBoundingClientRect();
  return rect.top < window.innerHeight * 0.6 && rect.bottom > window.innerHeight * 0.4;
}
