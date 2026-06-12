/**
 * Landing position for chapter navigation (design-elevation P0 fix).
 *
 * Sections are 160–280svh of camera runway. Landing anchors the chapter
 * HEADING at a fixed fraction from the viewport top — layout-proof: however
 * tall the content block is (chip grids, split columns), the title is what
 * the visitor reads first and it always arrives at the same optical line.
 * align="start" chapters (hero) pin to the section top instead.
 */

/** Heading rests at 22% from the viewport top — optical headline line. */
export const HEADING_VIEWPORT_FRACTION = 0.22;

export function headingScrollOffset(
  viewportHeight: number,
  align: "start" | "center",
): number {
  if (align === "start") return 0;
  return -Math.round(viewportHeight * HEADING_VIEWPORT_FRACTION);
}
