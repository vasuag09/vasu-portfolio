/**
 * Signal Pulse logic (Living Network wave 1). Pure — the SignalDriver
 * component owns the rAF loop and uniforms; everything decidable without a
 * scene lives here.
 *
 * The pulse is the visitor's attention made visible: it travels the camera
 * target spline slightly AHEAD of scroll, and when scroll enters a chapter's
 * band the section core "fires" (flare + chord). Firing must happen exactly
 * once per entry — re-arm only after leaving the band (hysteresis), so
 * damped scroll settling never double-fires.
 */

/** How far ahead of scroll the pulse travels (normalized progress). */
export const PULSE_LEAD = 0.06;

/** Half-width of the arrival band around each section center. */
export const ARRIVAL_BAND = 0.035;

export function pulseProgress(progress: number, lead: number = PULSE_LEAD): number {
  return Math.min(1, Math.max(progress, progress + lead));
}

export interface ArrivalState {
  /** Chapter index currently occupied, or null when between chapters. */
  inside: number | null;
}

export function createArrivalState(): ArrivalState {
  return { inside: null };
}

/**
 * Reducer: given the previous state and current progress, returns the next
 * state and the chapter index that just fired (or null). Direction-agnostic.
 */
export function detectArrival(
  state: ArrivalState,
  progress: number,
  centers: readonly number[],
  band: number = ARRIVAL_BAND,
): [ArrivalState, number | null] {
  let occupied: number | null = null;
  for (let i = 0; i < centers.length; i += 1) {
    if (Math.abs(progress - centers[i]) <= band) {
      occupied = i;
      break;
    }
  }
  if (occupied === state.inside) return [state, null];
  const next: ArrivalState = { inside: occupied };
  // Entering a band fires; leaving one (occupied=null) just re-arms.
  return [next, occupied];
}
