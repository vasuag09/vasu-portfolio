"use client";

import { useSyncExternalStore } from "react";
import {
  getGraphState,
  subscribeGraphState,
  type GraphUIState,
} from "@/lib/graph-store";

const getServerSnapshot = (): GraphUIState => getGraphState();

/** React-reactive view of the graph interaction store. */
export function useGraphState(): GraphUIState {
  return useSyncExternalStore(
    subscribeGraphState,
    getGraphState,
    getServerSnapshot,
  );
}
