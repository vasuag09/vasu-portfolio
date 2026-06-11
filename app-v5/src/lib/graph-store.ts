/**
 * Graph interaction store (ADR-4 DOM→Canvas channel).
 *
 * One source of truth, two read paths:
 *  - DOM components subscribe through useGraphState (useSyncExternalStore)
 *  - the canvas reads getGraphState() directly inside useFrame — no React
 *    re-render in the frame loop (same rationale as scroll-state.ts)
 *
 * State is replaced immutably on every set; only this module mutates the
 * reference.
 */

export interface GraphUIState {
  hoveredSkillId: string | null;
  hoveredProjectId: string | null;
  selectedSkillId: string | null;
  selectedProjectId: string | null;
}

let state: GraphUIState = {
  hoveredSkillId: null,
  hoveredProjectId: null,
  selectedSkillId: null,
  selectedProjectId: null,
};

const listeners = new Set<() => void>();

export function getGraphState(): GraphUIState {
  return state;
}

export function setGraphState(partial: Partial<GraphUIState>): void {
  state = { ...state, ...partial };
  listeners.forEach((listener) => listener());
}

export function subscribeGraphState(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

/** The ids that should drive canvas glow right now (hover wins over selection). */
export function getActiveIds(current: GraphUIState = state): {
  skillId: string | null;
  projectId: string | null;
} {
  return {
    skillId: current.hoveredSkillId ?? current.selectedSkillId,
    projectId: current.hoveredProjectId ?? current.selectedProjectId,
  };
}
