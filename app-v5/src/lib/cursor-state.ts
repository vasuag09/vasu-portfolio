/**
 * Custom cursor (Phase 10) — pure capability/mode resolution. The
 * CustomCursor component gathers matchMedia results and pointer events;
 * the decisions live here (same pattern as particle-config.ts).
 */

export interface CursorCaps {
  finePointer: boolean;
  canHover: boolean;
  reducedMotion: boolean;
}

/**
 * Touch devices keep the native (no) cursor; reduced motion keeps the OS
 * cursor — the trailing-ring lerp is exactly the kind of decorative motion
 * the preference opts out of.
 */
export function resolveCursorEnabled(caps: CursorCaps): boolean {
  return caps.finePointer && caps.canHover && !caps.reducedMotion;
}

export type CursorMode = "default" | "interactive" | "pressed";

export function cursorMode(state: {
  overInteractive: boolean;
  pressed: boolean;
}): CursorMode {
  if (state.pressed) return "pressed";
  return state.overInteractive ? "interactive" : "default";
}

/** What the cursor treats as "interactive" for the morph state. */
export const INTERACTIVE_SELECTOR =
  'a, button, input, textarea, select, [role="button"], [data-cursor="interactive"]';
