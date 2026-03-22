import { useRef, useEffect, useCallback, useLayoutEffect } from "react";

/**
 * Neural network canvas engine.
 * Renders a layered network with nodes, connections, and flowing data particles.
 * `activeLayer` highlights the current section's layer (0-4).
 */

const COLORS = {
  cyan: { r: 0, g: 212, b: 255 },
  purple: { r: 139, g: 92, b: 246 },
  white: { r: 241, g: 245, b: 249 },
};

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function lerpColor(c1, c2, t) {
  return {
    r: Math.round(lerp(c1.r, c2.r, t)),
    g: Math.round(lerp(c1.g, c2.g, t)),
    b: Math.round(lerp(c1.b, c2.b, t)),
  };
}

function createNetwork(width, height, isMobile) {
  const layers = isMobile ? [3, 4, 5, 4, 3] : [4, 6, 8, 6, 4];
  const nodes = [];
  const connections = [];
  const layerCount = layers.length;

  const paddingX = isMobile ? 60 : 120;
  const layerSpacing = (width - paddingX * 2) / (layerCount - 1);

  layers.forEach((nodeCount, layerIdx) => {
    const x = paddingX + layerIdx * layerSpacing;
    const layerHeight = height * (isMobile ? 0.6 : 0.7);
    const startY = (height - layerHeight) / 2;
    const nodeSpacing = layerHeight / (nodeCount + 1);

    for (let i = 0; i < nodeCount; i++) {
      const y = startY + nodeSpacing * (i + 1);
      const t = layerIdx / (layerCount - 1);
      const baseColor = lerpColor(COLORS.cyan, COLORS.purple, t);

      nodes.push({
        x,
        y,
        layer: layerIdx,
        radius: isMobile ? 3 : (layerIdx === 0 || layerIdx === layerCount - 1 ? 5 : 4),
        baseColor,
        brightness: 0.3,
        pulsePhase: Math.random() * Math.PI * 2,
      });
    }
  });

  // Create connections between adjacent layers
  for (let i = 0; i < nodes.length; i++) {
    for (let j = i + 1; j < nodes.length; j++) {
      if (nodes[j].layer === nodes[i].layer + 1) {
        // Don't connect every node — keep it sparse for aesthetics
        if (Math.random() > 0.4) {
          connections.push({ from: i, to: j, opacity: 0.06 + Math.random() * 0.06 });
        }
      }
    }
  }

  return { nodes, connections };
}

function createParticles(connections, nodes, count) {
  const particles = [];
  for (let i = 0; i < count; i++) {
    const connIdx = Math.floor(Math.random() * connections.length);
    const conn = connections[connIdx];
    const fromNode = nodes[conn.from];
    const t = fromNode.layer / 4; // normalize 0→1 across layers

    particles.push({
      connIdx,
      progress: Math.random(),
      speed: 0.002 + Math.random() * 0.004,
      size: 1 + Math.random() * 1.5,
      color: lerpColor(COLORS.cyan, COLORS.purple, t),
      opacity: 0.3 + Math.random() * 0.4,
    });
  }
  return particles;
}

export function useNeuralNetwork(canvasRef, activeLayer = 0) {
  const networkRef = useRef(null);
  const particlesRef = useRef([]);
  const animationRef = useRef(null);
  const mouseRef = useRef({ x: -1000, y: -1000 });
  const activeLayerRef = useRef(activeLayer);

  useLayoutEffect(() => {
    activeLayerRef.current = activeLayer;
  }, [activeLayer]);

  const init = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = canvas.offsetWidth * dpr;
    canvas.height = canvas.offsetHeight * dpr;

    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);

    const isMobile = canvas.offsetWidth < 768;
    const network = createNetwork(canvas.offsetWidth, canvas.offsetHeight, isMobile);
    networkRef.current = network;

    const particleCount = isMobile ? 30 : 80;
    particlesRef.current = createParticles(network.connections, network.nodes, particleCount);
  }, [canvasRef]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    init();

    const ctx = canvas.getContext("2d");
    const w = canvas.offsetWidth;
    const h = canvas.offsetHeight;

    const handleMouseMove = (e) => {
      const rect = canvas.getBoundingClientRect();
      mouseRef.current = { x: e.clientX - rect.left, y: e.clientY - rect.top };
    };

    const handleMouseLeave = () => {
      mouseRef.current = { x: -1000, y: -1000 };
    };

    canvas.addEventListener("mousemove", handleMouseMove);
    canvas.addEventListener("mouseleave", handleMouseLeave);

    let time = 0;

    function render() {
      ctx.clearRect(0, 0, w, h);
      const network = networkRef.current;
      if (!network) return;

      const { nodes, connections } = network;
      const mouse = mouseRef.current;
      const currentLayer = activeLayerRef.current;

      time += 0.016;

      // Draw connections
      connections.forEach((conn) => {
        const from = nodes[conn.from];
        const to = nodes[conn.to];

        const isActiveConnection =
          from.layer === currentLayer || to.layer === currentLayer;
        const opacity = isActiveConnection ? conn.opacity * 2.5 : conn.opacity;

        const t = from.layer / 4;
        const color = lerpColor(COLORS.cyan, COLORS.purple, t);

        ctx.beginPath();
        ctx.moveTo(from.x, from.y);
        ctx.lineTo(to.x, to.y);
        ctx.strokeStyle = `rgba(${color.r}, ${color.g}, ${color.b}, ${opacity})`;
        ctx.lineWidth = isActiveConnection ? 0.8 : 0.4;
        ctx.stroke();
      });

      // Draw nodes
      nodes.forEach((node) => {
        const dx = mouse.x - node.x;
        const dy = mouse.y - node.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        const mouseInfluence = Math.max(0, 1 - dist / 150);

        const isActiveLayer = node.layer === currentLayer;
        const pulse = Math.sin(time * 1.5 + node.pulsePhase) * 0.15 + 0.85;

        let brightness = node.brightness + mouseInfluence * 0.5;
        if (isActiveLayer) brightness = 0.8 * pulse;

        const { r, g, b } = node.baseColor;

        // Glow
        if (brightness > 0.4) {
          const glowRadius = node.radius * (3 + brightness * 2);
          const gradient = ctx.createRadialGradient(
            node.x, node.y, 0,
            node.x, node.y, glowRadius,
          );
          gradient.addColorStop(0, `rgba(${r}, ${g}, ${b}, ${brightness * 0.3})`);
          gradient.addColorStop(1, `rgba(${r}, ${g}, ${b}, 0)`);
          ctx.beginPath();
          ctx.arc(node.x, node.y, glowRadius, 0, Math.PI * 2);
          ctx.fillStyle = gradient;
          ctx.fill();
        }

        // Node core
        ctx.beginPath();
        ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${0.3 + brightness * 0.7})`;
        ctx.fill();

        // Active layer ring
        if (isActiveLayer) {
          ctx.beginPath();
          ctx.arc(node.x, node.y, node.radius + 3, 0, Math.PI * 2);
          ctx.strokeStyle = `rgba(${r}, ${g}, ${b}, ${0.3 * pulse})`;
          ctx.lineWidth = 1;
          ctx.stroke();
        }
      });

      // Draw particles
      if (!prefersReducedMotion) {
        particlesRef.current.forEach((particle) => {
          const conn = connections[particle.connIdx];
          const from = nodes[conn.from];
          const to = nodes[conn.to];

          particle.progress += particle.speed;
          if (particle.progress > 1) {
            particle.progress = 0;
            // Reassign to random connection
            particle.connIdx = Math.floor(Math.random() * connections.length);
          }

          const x = lerp(from.x, to.x, particle.progress);
          const y = lerp(from.y, to.y, particle.progress);
          const { r, g, b } = particle.color;

          const isActiveParticle =
            from.layer === currentLayer || to.layer === currentLayer;
          const alpha = isActiveParticle
            ? particle.opacity * 1.5
            : particle.opacity * 0.5;

          ctx.beginPath();
          ctx.arc(x, y, particle.size, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${alpha})`;
          ctx.fill();
        });
      }

      animationRef.current = requestAnimationFrame(render);
    }

    // If reduced motion, render once statically
    if (prefersReducedMotion) {
      render();
    } else {
      render();
    }

    let resizeTimer;
    const handleResize = () => {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(init, 200);
    };
    window.addEventListener("resize", handleResize);

    return () => {
      cancelAnimationFrame(animationRef.current);
      clearTimeout(resizeTimer);
      window.removeEventListener("resize", handleResize);
      canvas.removeEventListener("mousemove", handleMouseMove);
      canvas.removeEventListener("mouseleave", handleMouseLeave);
    };
  }, [canvasRef, init]);
}
