import { useContext } from "react";
import { UIContext } from "../context/UIContext";

/**
 * useUI is a custom hook to access the UIContext.
 */
export const useUI = () => {
  const context = useContext(UIContext);
  if (!context) {
    throw new Error("useUI must be used within a UIProvider");
  }
  return context;
};
