/**
 * Number-key chapter navigation (Phase 10): keys "1".."N" map to section
 * indices. Pure; ChapterNav guards focus/modal context before calling.
 */
export function sectionIndexForKey(
  key: string,
  sectionCount: number,
): number | null {
  if (key.length !== 1 || key < "1" || key > "9") return null;
  const index = Number(key) - 1;
  return index < sectionCount ? index : null;
}
