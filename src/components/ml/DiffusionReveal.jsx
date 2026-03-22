import React, { useRef, useEffect, useState, useCallback } from "react";

/**
 * Diffusion Name Reveal — particles start as noise and "denoise"
 * into the text. After settling, particles are interactive:
 * mouse proximity scatters them, and they spring back to form the name.
 *
 * Pure Canvas 2D — no ML model, just math that looks like diffusion.
 */

const TOTAL_STEPS = 28;
const STEP_DURATION = 75;
const SETTLE_BREATHE_AMP = 0.4;
const MOUSE_RADIUS = 80;       // scatter radius around cursor
const MOUSE_FORCE = 12;        // how hard particles get pushed
const SPRING_BACK = 0.06;      // how fast particles return (0-1)
const DAMPING = 0.88;          // velocity damping

function diffusionEase(t) {
  return 1 - Math.pow(1 - t, 3);
}

function sampleTextPositions(ctx, text, fontSize, x, y, width, height, maxParticles) {
  // Clear and draw text
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#ffffff";
  ctx.font = `900 ${fontSize}px Syne, system-ui, sans-serif`;
  ctx.textAlign = "left";
  ctx.textBaseline = "top";
  ctx.fillText(text, x, y);

  const imageData = ctx.getImageData(0, 0, width, height);
  const positions = [];

  // Dense sampling — step of 2 for readable text
  for (let py = 0; py < height; py += 2) {
    for (let px = 0; px < width; px += 2) {
      const idx = (py * width + px) * 4;
      if (imageData.data[idx + 3] > 100) {
        positions.push({ x: px, y: py });
      }
    }
  }

  // Shuffle and take up to maxParticles
  for (let i = positions.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    const tmp = positions[i];
    positions[i] = positions[j];
    positions[j] = tmp;
  }
  return positions.slice(0, maxParticles);
}

export default function DiffusionReveal({
  firstName = "VASU",
  lastName = "AGRAWAL",
  className = "",
  onComplete,
}) {
  const canvasRef = useRef(null);
  const containerRef = useRef(null);
  const animRef = useRef(null);
  const particlesRef = useRef(null);
  const mouseRef = useRef({ x: -9999, y: -9999, active: false });
  const dimensionsRef = useRef({ width: 0, height: 0 });

  const reducedMotion = typeof window !== "undefined" &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  const [step, setStep] = useState(() => reducedMotion ? TOTAL_STEPS : 0);
  const [settled, setSettled] = useState(() => reducedMotion);

  const initParticles = useCallback(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const rect = container.getBoundingClientRect();
    const width = rect.width;
    const height = rect.height;

    dimensionsRef.current = { width, height };
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;

    // Use an offscreen canvas for text sampling (at 1x scale for speed)
    const offscreen = document.createElement("canvas");
    offscreen.width = Math.round(width);
    offscreen.height = Math.round(height);
    const offCtx = offscreen.getContext("2d");

    // Auto-fit font size so the longest word fills the canvas width
    const isMobile = width < 640;
    const maxWidth = width - 8; // 4px padding each side
    let fontSize = isMobile ? Math.floor(width / 3.2) : Math.floor(width / 4.0);

    // Scale down until the longest word fits
    const longerWord = firstName.length >= lastName.length ? firstName : lastName;
    offCtx.font = `900 ${fontSize}px Syne, system-ui, sans-serif`;
    while (offCtx.measureText(longerWord).width > maxWidth && fontSize > 20) {
      fontSize -= 2;
      offCtx.font = `900 ${fontSize}px Syne, system-ui, sans-serif`;
    }

    const lineGap = fontSize * 0.18;
    const textX = 4; // near-flush left
    const firstNameY = 2;
    const lastNameY = firstNameY + fontSize + lineGap;

    // Particle budget per line (proportional to text length)
    const totalBudget = isMobile ? 1000 : 1600;
    const firstRatio = firstName.length / (firstName.length + lastName.length);
    const firstBudget = Math.floor(totalBudget * firstRatio);
    const lastBudget = totalBudget - firstBudget;

    const firstTargets = sampleTextPositions(
      offCtx, firstName, fontSize, textX, firstNameY,
      offscreen.width, offscreen.height, firstBudget
    );
    const lastTargets = sampleTextPositions(
      offCtx, lastName, fontSize, textX, lastNameY,
      offscreen.width, offscreen.height, lastBudget
    );

    const allTargets = [...firstTargets, ...lastTargets];
    const firstCount = firstTargets.length;

    const particles = allTargets.map((target, i) => {
      const isFirst = i < firstCount;
      const startX = Math.random() * width;
      const startY = Math.random() * height;

      // Cyan spectrum only — no pink
      const hue = isFirst
        ? 178 + Math.random() * 14  // cyan 178-192
        : 195 + Math.random() * 18; // blue-cyan 195-213
      const saturation = 80 + Math.random() * 20;
      const lightness = 58 + Math.random() * 16;

      return {
        startX,
        startY,
        targetX: target.x,
        targetY: target.y,
        x: startX,
        y: startY,
        vx: 0,
        vy: 0,
        size: 1.2 + Math.random() * 1.0,
        hue,
        saturation,
        lightness,
        delay: Math.random() * 0.1,
        noisePhaseX: Math.random() * Math.PI * 2,
        noisePhaseY: Math.random() * Math.PI * 2,
        noiseSpeed: 0.4 + Math.random() * 1.2,
      };
    });

    particlesRef.current = particles;
  }, [firstName, lastName]);

  // Mouse tracking
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const handleMove = (e) => {
      const rect = canvas.getBoundingClientRect();
      const clientX = e.touches ? e.touches[0].clientX : e.clientX;
      const clientY = e.touches ? e.touches[0].clientY : e.clientY;
      mouseRef.current.x = clientX - rect.left;
      mouseRef.current.y = clientY - rect.top;
      mouseRef.current.active = true;
    };

    const handleLeave = () => {
      mouseRef.current.active = false;
    };

    canvas.addEventListener("mousemove", handleMove);
    canvas.addEventListener("touchmove", handleMove, { passive: true });
    canvas.addEventListener("mouseleave", handleLeave);
    canvas.addEventListener("touchend", handleLeave);

    return () => {
      canvas.removeEventListener("mousemove", handleMove);
      canvas.removeEventListener("touchmove", handleMove);
      canvas.removeEventListener("mouseleave", handleLeave);
      canvas.removeEventListener("touchend", handleLeave);
    };
  }, []);

  // Initialize on mount
  useEffect(() => {
    if (reducedMotion) return;
    initParticles();
    const handleResize = () => initParticles();
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, [initParticles, reducedMotion]);

  // Step through diffusion process
  useEffect(() => {
    if (reducedMotion || !particlesRef.current || step >= TOTAL_STEPS) return;

    const timer = setTimeout(() => {
      const nextStep = step + 1;
      setStep(nextStep);
      if (nextStep >= TOTAL_STEPS) {
        setSettled(true);
        onComplete?.();
      }
    }, STEP_DURATION);

    return () => clearTimeout(timer);
  }, [step, reducedMotion, onComplete]);

  // Main render loop
  useEffect(() => {
    if (reducedMotion) return;
    const canvas = canvasRef.current;
    if (!canvas || !particlesRef.current) return;

    const ctx = canvas.getContext("2d");
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const { width, height } = dimensionsRef.current;
    const particles = particlesRef.current;
    let running = true;

    function render() {
      if (!running) return;

      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, width, height);

      const progress = Math.min(step / TOTAL_STEPS, 1);
      const time = performance.now() * 0.001;
      const mouse = mouseRef.current;

      for (let i = 0; i < particles.length; i++) {
        const p = particles[i];

        if (settled) {
          // Target position with gentle breathing
          const tx = p.targetX + Math.sin(time * p.noiseSpeed + p.noisePhaseX) * SETTLE_BREATHE_AMP;
          const ty = p.targetY + Math.cos(time * p.noiseSpeed + p.noisePhaseY) * SETTLE_BREATHE_AMP;

          // Mouse repulsion
          if (mouse.active) {
            const dx = p.x - mouse.x;
            const dy = p.y - mouse.y;
            const dist = Math.sqrt(dx * dx + dy * dy);
            if (dist < MOUSE_RADIUS && dist > 0) {
              const force = (1 - dist / MOUSE_RADIUS) * MOUSE_FORCE;
              p.vx += (dx / dist) * force;
              p.vy += (dy / dist) * force;
            }
          }

          // Spring back to target
          p.vx += (tx - p.x) * SPRING_BACK;
          p.vy += (ty - p.y) * SPRING_BACK;

          // Damping
          p.vx *= DAMPING;
          p.vy *= DAMPING;

          p.x += p.vx;
          p.y += p.vy;

          ctx.beginPath();
          ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
          ctx.fillStyle = `hsla(${p.hue}, ${p.saturation}%, ${p.lightness}%, 0.92)`;
          ctx.fill();
        } else {
          // Diffusion phase: interpolate from random to target
          const particleProgress = Math.max(0, Math.min(1, (progress - p.delay) / (1 - p.delay)));
          const eased = diffusionEase(particleProgress);

          let x = p.startX + (p.targetX - p.startX) * eased;
          let y = p.startY + (p.targetY - p.startY) * eased;

          const noiseScale = (1 - eased) * 30;
          x += Math.sin(time * 2 + p.noisePhaseX) * noiseScale;
          y += Math.cos(time * 2.3 + p.noisePhaseY) * noiseScale;

          // Initialize position for when settled begins
          p.x = x;
          p.y = y;
          p.vx = 0;
          p.vy = 0;

          const alpha = 0.3 + eased * 0.62;

          ctx.beginPath();
          ctx.arc(x, y, p.size, 0, Math.PI * 2);
          ctx.fillStyle = `hsla(${p.hue}, ${p.saturation}%, ${p.lightness}%, ${alpha})`;
          ctx.fill();
        }
      }

      animRef.current = requestAnimationFrame(render);
    }

    render();

    return () => {
      running = false;
      if (animRef.current) cancelAnimationFrame(animRef.current);
    };
  }, [step, settled, reducedMotion]);

  if (reducedMotion) {
    return (
      <div className={className}>
        <h1
          className="text-6xl md:text-8xl lg:text-9xl font-extrabold leading-[0.9] tracking-tighter"
          style={{ fontFamily: "var(--font-display)" }}
        >
          <span className="gradient-neural-text block">{firstName}</span>
          <span className="text-white block">{lastName}</span>
        </h1>
      </div>
    );
  }

  return (
    <div ref={containerRef} className={`relative ${className}`}>
      <canvas
        ref={canvasRef}
        className="w-full h-full"
        aria-hidden="true"
        style={{ touchAction: "pan-y" }}
      />

      {/* Step counter — fades out after completion */}
      <div
        className={`absolute bottom-0 left-0 flex transition-opacity duration-700 pointer-events-none ${
          settled ? "opacity-0" : "opacity-100"
        }`}
      >
        <div className="flex items-center gap-3 px-3 py-1.5 rounded-lg bg-[rgba(6,8,15,0.6)] backdrop-blur-sm border border-[rgba(0,212,255,0.08)]">
          <span className="text-[10px] font-mono text-cyan-500/60 tracking-wider">
            DENOISING
          </span>
          <div className="w-20 h-1 bg-[rgba(255,255,255,0.04)] rounded-full overflow-hidden">
            <div
              className="h-full rounded-full transition-all duration-75"
              style={{
                width: `${(step / TOTAL_STEPS) * 100}%`,
                background: "linear-gradient(90deg, var(--accent-electric), #38bdf8)",
              }}
            />
          </div>
          <span className="text-[10px] font-mono text-slate-600">
            {Math.min(step, TOTAL_STEPS)}/{TOTAL_STEPS}
          </span>
        </div>
      </div>

      {/* Hint for interactivity — appears after settled */}
      <div
        className={`absolute bottom-0 right-0 transition-opacity duration-1000 delay-500 pointer-events-none ${
          settled ? "opacity-100" : "opacity-0"
        }`}
      >
        <span className="text-[9px] font-mono text-cyan-500/30 tracking-wider">
          HOVER TO INTERACT
        </span>
      </div>

      <h1 className="sr-only">
        {firstName} {lastName}
      </h1>
    </div>
  );
}
