import { scrollState } from "./scroll-state";
import { landingOffset } from "./scroll-landing";

/**
 * Shared chapter navigation (ChapterNav dots, number keys, deep links,
 * panel camera alignment). Smooth via Lenis when it exists; native jump
 * otherwise (reduced motion). Lands on the section's CONTENT band, not its
 * top edge — Section.tsx publishes its alignment via data-align.
 */
export function scrollToSection(
  id: string,
  immediate: boolean,
  options: { force?: boolean } = {},
): void {
  const el = document.getElementById(id);
  if (!el) return;
  const align = el.dataset.align === "start" ? "start" : "center";
  const offset = landingOffset(
    el.getBoundingClientRect().height,
    window.innerHeight,
    align,
  );
  if (scrollState.lenis) {
    scrollState.lenis.scrollTo(el, {
      offset,
      duration: immediate ? 0 : 1.4,
      immediate,
      // force: animate even while lenis is stopped (modal scroll lock).
      force: options.force ?? false,
    });
  } else {
    const top = el.getBoundingClientRect().top + window.scrollY + offset;
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
