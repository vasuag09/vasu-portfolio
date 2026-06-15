/**
 * Tab-blur easter egg (Living wave 2) — when the visitor leaves the tab the
 * title speaks up in-character, restoring the real title on return. Pure
 * controller (inject the title setter); TabTitleEffect wires it to
 * document.title + window blur/focus. A static text swap, not animation, so
 * it stays on under reduced motion.
 */

export const BLUR_TITLES = [
  "◍ the network is dreaming…",
  "◍ neurons idle — come back",
  "◍ awaiting your signal…",
];

/** Blur title for the Nth blur (rotates through the variants, wraps). */
export function blurTitleAt(index: number): string {
  return BLUR_TITLES[index % BLUR_TITLES.length];
}

export interface TitleController {
  onBlur(): void;
  onFocus(): void;
}

/**
 * Counts blurs to rotate the message and restores `originalTitle` on focus.
 * `setTitle` is injected so the blur/focus/rotate behaviour is unit-testable
 * without a DOM.
 */
export function createTitleController(
  originalTitle: string,
  setTitle: (title: string) => void,
): TitleController {
  let blurs = 0;
  return {
    onBlur(): void {
      setTitle(blurTitleAt(blurs));
      blurs += 1;
    },
    onFocus(): void {
      setTitle(originalTitle);
    },
  };
}
