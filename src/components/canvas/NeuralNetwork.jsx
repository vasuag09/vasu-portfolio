import React, { useRef } from "react";
import { useNeuralNetwork } from "./useNeuralNetwork";

/**
 * Full-screen neural network canvas that serves as the persistent background.
 * `activeLayer` (0-4) highlights the current section's layer in the network.
 */
export default function NeuralNetwork({ activeLayer = 0 }) {
  const canvasRef = useRef(null);
  useNeuralNetwork(canvasRef, activeLayer);

  return (
    <canvas
      ref={canvasRef}
      className="fixed inset-0 w-full h-full pointer-events-auto z-0 opacity-40"
      aria-hidden="true"
      style={{ background: "transparent" }}
    />
  );
}
