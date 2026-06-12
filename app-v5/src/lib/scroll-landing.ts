/**
 * Landing position for chapter navigation (design-elevation P0 fix).
 *
 * Sections are 160–280svh of camera runway with their DOM content at the
 * vertical CENTER (align="center") or pinned to the first viewport
 * (align="start"). Landing at the section TOP put centered content a full
 * viewport below the fold — nav dots, number keys, and deep links all
 * landed on empty scene. The reading position is the section's center band
 * aligned to the viewport center.
 */
export function landingOffset(
  sectionHeight: number,
  viewportHeight: number,
  align: "start" | "center",
): number {
  if (align === "start") return 0;
  return Math.max(0, Math.round((sectionHeight - viewportHeight) / 2));
}
