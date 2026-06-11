import { scrollState } from "./scroll-state";

/**
 * Shared chapter navigation (ChapterNav dots, panel camera alignment).
 * Smooth via Lenis when it exists; native jump otherwise (reduced motion).
 */
export function scrollToSection(id: string, immediate: boolean): void {
  const el = document.getElementById(id);
  if (!el) return;
  if (scrollState.lenis) {
    scrollState.lenis.scrollTo(el, {
      offset: 0,
      duration: immediate ? 0 : 1.4,
      immediate,
    });
  } else {
    el.scrollIntoView({ behavior: "auto" });
  }
}

/** True when the section is already roughly centered in the viewport. */
export function isSectionInView(id: string): boolean {
  const el = document.getElementById(id);
  if (!el) return false;
  const rect = el.getBoundingClientRect();
  return rect.top < window.innerHeight * 0.6 && rect.bottom > window.innerHeight * 0.4;
}
