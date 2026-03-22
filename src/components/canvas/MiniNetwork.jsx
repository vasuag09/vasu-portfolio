import React, { useRef, useEffect } from "react";

/**
 * Small neural network animation for the Synapse "processing" state.
 * Shows a forward pass: nodes light up left-to-right in sequence.
 */
export default function MiniNetwork({ active = true }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !active) return;

    const ctx = canvas.getContext("2d");
    const w = 200;
    const h = 40;
    canvas.width = w * 2;
    canvas.height = h * 2;
    ctx.scale(2, 2);

    const layers = [3, 4, 3];
    const nodes = [];

    layers.forEach((count, layerIdx) => {
      const x = 30 + layerIdx * 70;
      const spacing = h / (count + 1);
      for (let i = 0; i < count; i++) {
        nodes.push({ x, y: spacing * (i + 1), layer: layerIdx });
      }
    });

    let time = 0;
    let animId;

    function render() {
      ctx.clearRect(0, 0, w, h);
      time += 0.03;

      // Connections
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          if (nodes[j].layer === nodes[i].layer + 1) {
            ctx.beginPath();
            ctx.moveTo(nodes[i].x, nodes[i].y);
            ctx.lineTo(nodes[j].x, nodes[j].y);
            ctx.strokeStyle = "rgba(0, 212, 255, 0.1)";
            ctx.lineWidth = 0.5;
            ctx.stroke();
          }
        }
      }

      // Nodes with sequential activation
      nodes.forEach((node) => {
        const activationPhase = (time - node.layer * 0.3) % 2;
        const brightness = activationPhase > 0 && activationPhase < 0.6
          ? 0.4 + Math.sin(activationPhase * Math.PI / 0.6) * 0.6
          : 0.2;

        const t = node.layer / 2;
        const r = Math.round(0 + 139 * t);
        const g = Math.round(212 - 120 * t);
        const b = Math.round(255 - 9 * t);

        // Glow
        if (brightness > 0.4) {
          const grad = ctx.createRadialGradient(node.x, node.y, 0, node.x, node.y, 8);
          grad.addColorStop(0, `rgba(${r}, ${g}, ${b}, ${brightness * 0.3})`);
          grad.addColorStop(1, `rgba(${r}, ${g}, ${b}, 0)`);
          ctx.beginPath();
          ctx.arc(node.x, node.y, 8, 0, Math.PI * 2);
          ctx.fillStyle = grad;
          ctx.fill();
        }

        ctx.beginPath();
        ctx.arc(node.x, node.y, 2.5, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${brightness})`;
        ctx.fill();
      });

      animId = requestAnimationFrame(render);
    }

    render();
    return () => cancelAnimationFrame(animId);
  }, [active]);

  if (!active) return null;

  return (
    <canvas
      ref={canvasRef}
      className="w-[200px] h-[40px]"
      aria-hidden="true"
    />
  );
}
