/**
 * Defer work until after the next paint. INP counts an interaction's
 * latency up to the first paint that follows it — closing an overlay
 * synchronously bundles the unmount AND the scroll-unlock (a relayout of
 * the whole ~1200svh document) into that first frame (~300–550ms measured
 * in Phase 8, before CPU throttling). Painting the click feedback first
 * and unmounting one task later keeps the interaction under budget; the
 * heavy frame still happens, but off the interaction's clock and invisible
 * to the user (the panel is gone either way within ~2 frames).
 */
export function afterNextPaint(fn: () => void): void {
  // rAF runs BEFORE the upcoming paint; the nested macrotask lands after it.
  requestAnimationFrame(() => {
    window.setTimeout(fn, 0);
  });
}
