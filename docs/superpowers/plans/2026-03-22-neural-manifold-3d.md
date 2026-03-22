# Neural Manifold 3D — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform the portfolio from a flat 2D canvas-decorated site into an immersive 3D neural network experience that no other ML/AI portfolio has attempted — where the visitor physically navigates through a living, breathing neural architecture.

**Architecture:** Replace the 2D Canvas API neural network with a full Three.js (@react-three/fiber) 3D scene featuring instanced mesh nodes, custom shader connections with energy flow, GPU-accelerated particle systems, and post-processing (bloom, chromatic aberration, vignette). Add GSAP ScrollTrigger for cinematic scroll-driven animations, holographic tilt cards, magnetic cursor interactions, and a typography overhaul with distinctive fonts. The 3D scene serves as both background and interactive element — nodes respond to cursor proximity, active layers ignite with volumetric glow, and data particles stream through connections in 3D space.

**Tech Stack:** Three.js, @react-three/fiber, @react-three/drei, @react-three/postprocessing, GSAP + ScrollTrigger, custom GLSL shaders, Google Fonts (Syne + Outfit + JetBrains Mono)

---

## File Structure

### New Files
- `src/components/canvas/NeuralNetwork3D.jsx` — R3F Canvas wrapper with camera, fog, post-processing
- `src/components/canvas/NetworkNodes.jsx` — Instanced sphere meshes for neural nodes (GPU-instanced)
- `src/components/canvas/NetworkConnections.jsx` — Line geometry with custom shader material for energy flow
- `src/components/canvas/NetworkParticles.jsx` — Point cloud particles flowing along connections
- `src/components/canvas/CameraRig.jsx` — Mouse-responsive camera controller
- `src/components/canvas/shaders/connectionMaterial.js` — Custom ShaderMaterial for animated energy flow
- `src/components/canvas/shaders/nodeMaterial.js` — Custom ShaderMaterial for node glow + pulse
- `src/components/effects/ScrollReveal.jsx` — GSAP ScrollTrigger wrapper component
- `src/components/effects/TiltCard.jsx` — Perspective-aware tilt with holographic shimmer
- `src/components/effects/TextScramble.jsx` — Hover-triggered text decode effect
- `src/components/effects/MagneticElement.jsx` — Cursor-attracted element wrapper
- `src/components/ui/ScrollProgress.jsx` — Top scroll progress indicator

### Modified Files
- `package.json` — Add three, @react-three/fiber, @react-three/drei, @react-three/postprocessing, gsap
- `index.html` — Replace fonts (Syne, Outfit replacing Space Grotesk, Inter)
- `src/index.css` — New design tokens, holographic keyframes, updated font vars
- `src/App.jsx` — Replace NeuralNetwork with NeuralNetwork3D, add ScrollProgress
- `src/components/sections/Hero.jsx` — Enhanced with TextScramble, MagneticElement, scroll-driven parallax
- `src/components/sections/Projects.jsx` — GSAP scroll reveals, staggered grid
- `src/components/sections/ProjectCard.jsx` — TiltCard wrapper, enhanced hover states
- `src/components/sections/Skills.jsx` — Scroll-triggered weight bar animations, staggered reveals
- `src/components/sections/Blog.jsx` — Scroll reveals, enhanced waveform interaction
- `src/components/sections/About.jsx` — MagneticElement on contact nodes
- `src/components/ui/GlassCard.jsx` — Add TiltCard integration option
- `src/components/layout/LayerNav.jsx` — MagneticElement on nav nodes
- `src/components/synapse/SynapseButton.jsx` — MagneticElement wrapper

### Removed Files
- `src/components/canvas/useNeuralNetwork.js` — Replaced by 3D system
- `src/components/canvas/NeuralNetwork.jsx` — Replaced by NeuralNetwork3D
- `src/components/effects/ParticleField.jsx` — Replaced by 3D particles
- `src/components/effects/ParallaxFloat.jsx` — Replaced by GSAP scroll parallax

---

## Task Breakdown

### Task 1: Install Dependencies

**Files:**
- Modify: `package.json`

- [ ] **Step 1: Install Three.js ecosystem + GSAP**

```bash
npm install three @react-three/fiber @react-three/drei @react-three/postprocessing gsap
```

- [ ] **Step 2: Verify installation**

```bash
npm run build
```

- [ ] **Step 3: Commit**

```bash
git add package.json package-lock.json
git commit -m "deps: add three.js ecosystem and gsap for 3D neural network"
```

---

### Task 2: Typography & Design Token Overhaul

**Files:**
- Modify: `index.html:39-42` (font links)
- Modify: `src/index.css:7-35` (CSS custom properties)

- [ ] **Step 1: Update font imports in index.html**

Replace Google Fonts link with:
```html
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@500;600;700;800&family=Outfit:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet" />
```

- [ ] **Step 2: Update CSS custom properties**

```css
--font-display: "Syne", system-ui, sans-serif;
--font-sans: "Outfit", system-ui, -apple-system, sans-serif;
--font-mono: "JetBrains Mono", "Fira Code", ui-monospace, monospace;
```

Add new design tokens:
```css
--accent-electric: #00f0ff;
--accent-violet: #a855f7;
--accent-rose: #f43f5e;
--glow-electric: rgba(0, 240, 255, 0.2);
--glow-violet: rgba(168, 85, 247, 0.2);
```

- [ ] **Step 3: Add holographic/iridescent keyframes**

```css
@keyframes holographic-shimmer {
  0% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
}

@keyframes glow-breathe {
  0%, 100% { filter: brightness(1) drop-shadow(0 0 8px var(--glow-electric)); }
  50% { filter: brightness(1.2) drop-shadow(0 0 20px var(--glow-electric)); }
}
```

- [ ] **Step 4: Verify build and lint**

```bash
npm run build && npm run lint
```

- [ ] **Step 5: Commit**

```bash
git add index.html src/index.css
git commit -m "style: overhaul typography (Syne + Outfit) and add holographic design tokens"
```

---

### Task 3: 3D Neural Network — Core Scene

**Files:**
- Create: `src/components/canvas/NeuralNetwork3D.jsx`
- Create: `src/components/canvas/NetworkNodes.jsx`
- Create: `src/components/canvas/CameraRig.jsx`

- [ ] **Step 1: Create CameraRig component**

Mouse-responsive camera that subtly orbits based on cursor position. Uses `useFrame` to lerp camera position toward mouse-derived target. Adds gentle auto-rotation when idle.

- [ ] **Step 2: Create NetworkNodes component**

Use `@react-three/drei`'s `Instances` for GPU-instanced rendering. Create 5 layers of spheres (matching the 4→6→8→6→4 topology). Each node has:
- Base emissive color (cyan→violet gradient across layers)
- Pulse animation via uniform time
- Brightness boost when in active layer
- Scale response to cursor proximity via raycasting

- [ ] **Step 3: Create NeuralNetwork3D wrapper**

R3F `<Canvas>` with:
- Perspective camera (fov: 60, position: [0, 0, 18])
- Fog for depth (near: 15, far: 35)
- Ambient + point lights
- NetworkNodes, CameraRig children
- Fixed position, full viewport, z-0

- [ ] **Step 4: Wire into App.jsx**

Replace `<NeuralNetwork activeLayer={activeLayer} />` with `<NeuralNetwork3D activeLayer={activeLayer} />`.

- [ ] **Step 5: Verify renders correctly**

```bash
npm run dev
```

- [ ] **Step 6: Commit**

```bash
git add src/components/canvas/NeuralNetwork3D.jsx src/components/canvas/NetworkNodes.jsx src/components/canvas/CameraRig.jsx src/App.jsx
git commit -m "feat: 3D neural network scene with instanced nodes and mouse-reactive camera"
```

---

### Task 4: 3D Neural Network — Connections & Energy Flow

**Files:**
- Create: `src/components/canvas/NetworkConnections.jsx`
- Create: `src/components/canvas/shaders/connectionMaterial.js`

- [ ] **Step 1: Create connection shader material**

Custom `ShaderMaterial` with:
- Vertex shader: pass UV + position
- Fragment shader: animated energy flow using `sin(uv.x * freq - uTime * speed)` creating a pulsing glow that travels along the connection
- Uniforms: `uTime`, `uColor`, `uOpacity`, `uActive`
- Active connections glow brighter with faster flow

- [ ] **Step 2: Create NetworkConnections component**

For each connection between adjacent-layer nodes:
- Create a `Line` geometry from node A to node B
- Apply the custom connectionMaterial
- Active layer connections use higher opacity + faster flow speed
- Sparse connectivity (40% probability, matching current behavior)

- [ ] **Step 3: Integrate into NeuralNetwork3D**

Add `<NetworkConnections>` as child of the Canvas scene.

- [ ] **Step 4: Verify and commit**

```bash
npm run dev
git add src/components/canvas/NetworkConnections.jsx src/components/canvas/shaders/connectionMaterial.js src/components/canvas/NeuralNetwork3D.jsx
git commit -m "feat: 3D neural connections with custom energy flow shader"
```

---

### Task 5: 3D Neural Network — Particle System

**Files:**
- Create: `src/components/canvas/NetworkParticles.jsx`

- [ ] **Step 1: Create particle system**

Use Three.js `Points` geometry with `BufferGeometry`:
- 200 particles (desktop) / 60 (mobile) flowing along connection paths
- Each particle has: position, velocity, connection index, progress (0→1)
- In `useFrame`: advance progress, interpolate position along connection
- When progress > 1: reset to random connection start
- Color: lerp cyan→violet based on source node layer
- Size: 1.5-3px with additive blending for glow
- Active layer particles: 2x size, 1.5x speed, brighter

- [ ] **Step 2: Add to NeuralNetwork3D scene**

- [ ] **Step 3: Performance test on mobile**

```bash
npm run dev
# Test at mobile viewport width
```

- [ ] **Step 4: Commit**

```bash
git add src/components/canvas/NetworkParticles.jsx src/components/canvas/NeuralNetwork3D.jsx
git commit -m "feat: GPU-accelerated particle system flowing through 3D neural connections"
```

---

### Task 6: Post-Processing Effects

**Files:**
- Modify: `src/components/canvas/NeuralNetwork3D.jsx`

- [ ] **Step 1: Add post-processing stack**

Using `@react-three/postprocessing`:
```jsx
<EffectComposer>
  <Bloom luminanceThreshold={0.3} luminanceSmoothing={0.9} intensity={0.8} />
  <ChromaticAberration offset={[0.0006, 0.0006]} />
  <Vignette darkness={0.4} offset={0.3} />
</EffectComposer>
```

- [ ] **Step 2: Add emissive materials to nodes for bloom pickup**

Nodes with `emissiveIntensity > 1` will be caught by bloom, creating natural glow halos in 3D space.

- [ ] **Step 3: Add reduced motion check**

Skip post-processing and use static rendering when `prefers-reduced-motion: reduce`.

- [ ] **Step 4: Commit**

```bash
git add src/components/canvas/NeuralNetwork3D.jsx
git commit -m "feat: post-processing — bloom, chromatic aberration, vignette"
```

---

### Task 7: Custom Node Glow Shader

**Files:**
- Create: `src/components/canvas/shaders/nodeMaterial.js`
- Modify: `src/components/canvas/NetworkNodes.jsx`

- [ ] **Step 1: Create node shader material**

Custom `ShaderMaterial`:
- Vertex: standard position + pass normals
- Fragment: Fresnel rim glow effect (bright edges, transparent center feel)
- Uniforms: `uTime`, `uColor`, `uActive` (0 or 1), `uHover` (0→1 lerp)
- Active nodes: pulsing emissive intensity, outer glow ring
- Hover nodes: scale up + brighter emission

- [ ] **Step 2: Apply to NetworkNodes instances**

Replace basic `meshStandardMaterial` with custom shader.

- [ ] **Step 3: Commit**

```bash
git add src/components/canvas/shaders/nodeMaterial.js src/components/canvas/NetworkNodes.jsx
git commit -m "feat: custom Fresnel glow shader for neural nodes"
```

---

### Task 8: Holographic Tilt Card

**Files:**
- Create: `src/components/effects/TiltCard.jsx`
- Modify: `src/components/sections/ProjectCard.jsx`
- Modify: `src/components/ui/GlassCard.jsx`

- [ ] **Step 1: Create TiltCard component**

Perspective-aware card wrapper:
- Tracks mouse position relative to card center
- Applies `rotateX` and `rotateY` transforms (max ±8deg)
- Holographic shimmer: moving gradient overlay that follows cursor
- Iridescent border glow that shifts hue with angle
- Spring-based return to neutral on mouse leave
- Respects `prefers-reduced-motion`

CSS for holographic effect:
```css
background: linear-gradient(
  var(--shimmer-angle),
  transparent 0%,
  rgba(0, 240, 255, 0.03) 25%,
  rgba(168, 85, 247, 0.06) 50%,
  rgba(0, 240, 255, 0.03) 75%,
  transparent 100%
);
```

- [ ] **Step 2: Wrap ProjectCard in TiltCard**

Replace basic `whileHover={{ y: -4 }}` with full 3D tilt.

- [ ] **Step 3: Add tilt option to GlassCard**

New `tilt` prop to optionally wrap in TiltCard.

- [ ] **Step 4: Commit**

```bash
git add src/components/effects/TiltCard.jsx src/components/sections/ProjectCard.jsx src/components/ui/GlassCard.jsx
git commit -m "feat: holographic tilt cards with iridescent shimmer effect"
```

---

### Task 9: Magnetic Element Effect

**Files:**
- Create: `src/components/effects/MagneticElement.jsx`
- Modify: `src/components/layout/LayerNav.jsx`
- Modify: `src/components/synapse/SynapseButton.jsx`
- Modify: `src/components/sections/About.jsx`

- [ ] **Step 1: Create MagneticElement wrapper**

Wrapper that creates gravitational pull toward cursor:
- Track mouse position relative to element center
- Apply transform: `translate(dx * strength, dy * strength)`
- Default strength: 0.3 (subtle)
- Spring-based return on mouse leave
- Pure CSS transform (no JS animation loop when idle)

- [ ] **Step 2: Apply to LayerNav nodes**

Wrap each navigation node button in MagneticElement.

- [ ] **Step 3: Apply to SynapseButton**

Wrap floating button in MagneticElement.

- [ ] **Step 4: Apply to About contact nodes**

Wrap contact links in MagneticElement.

- [ ] **Step 5: Commit**

```bash
git add src/components/effects/MagneticElement.jsx src/components/layout/LayerNav.jsx src/components/synapse/SynapseButton.jsx src/components/sections/About.jsx
git commit -m "feat: magnetic cursor attraction on interactive neural nodes"
```

---

### Task 10: GSAP Scroll Animations

**Files:**
- Create: `src/components/effects/ScrollReveal.jsx`
- Modify: `src/components/sections/Hero.jsx`
- Modify: `src/components/sections/Projects.jsx`
- Modify: `src/components/sections/Skills.jsx`
- Modify: `src/components/sections/Blog.jsx`
- Modify: `src/components/sections/About.jsx`

- [ ] **Step 1: Create ScrollReveal component**

GSAP ScrollTrigger wrapper:
```jsx
<ScrollReveal animation="fadeUp" stagger={0.08} delay={0}>
  {children}
</ScrollReveal>
```

Supported animations:
- `fadeUp`: opacity 0→1, y 60→0
- `fadeIn`: opacity 0→1
- `slideLeft`: opacity 0→1, x -40→0
- `slideRight`: opacity 0→1, x 40→0
- `scaleIn`: opacity 0→1, scale 0.9→1

Uses `ScrollTrigger` with `start: "top 85%"` trigger point.
Stagger children with configurable delay.
Respects `prefers-reduced-motion`.

- [ ] **Step 2: Replace Framer Motion entrance animations in Hero.jsx**

Replace `motion.div initial/animate` patterns with ScrollReveal wrappers. Keep GlitchText and ParallaxFloat effects.

- [ ] **Step 3: Update Projects.jsx with staggered grid reveal**

Cards stagger in with GSAP instead of Framer Motion `delay: index * 0.08`.

- [ ] **Step 4: Update Skills.jsx with scroll-triggered weight bars**

Weight bars animate on scroll into view using GSAP instead of Framer Motion `whileInView`.

- [ ] **Step 5: Update Blog.jsx with staggered reveals**

Blog cards stagger in on scroll.

- [ ] **Step 6: Update About.jsx with section reveals**

Each glass card section reveals on scroll.

- [ ] **Step 7: Commit**

```bash
git add src/components/effects/ScrollReveal.jsx src/components/sections/Hero.jsx src/components/sections/Projects.jsx src/components/sections/Skills.jsx src/components/sections/Blog.jsx src/components/sections/About.jsx
git commit -m "feat: GSAP ScrollTrigger cinematic scroll-driven animations"
```

---

### Task 11: Text Scramble Effect

**Files:**
- Create: `src/components/effects/TextScramble.jsx`
- Modify: `src/components/sections/Hero.jsx`
- Modify: `src/components/sections/ProjectCard.jsx`

- [ ] **Step 1: Create TextScramble component**

Hover-triggered text decode effect:
- On mouse enter: scramble text with random characters
- Progressive character-by-character resolve (left to right)
- Character pool: `!@#$%^&*()_+-=[]{}|;:,.<>?0123456789`
- Duration: 400ms total
- Each character resolves ~30ms after previous
- Respects `prefers-reduced-motion` (shows text immediately)

- [ ] **Step 2: Apply to Hero stat labels**

Stat labels scramble on hover.

- [ ] **Step 3: Apply to ProjectCard "Deep Dive" link**

"Deep Dive" text scrambles on card hover.

- [ ] **Step 4: Commit**

```bash
git add src/components/effects/TextScramble.jsx src/components/sections/Hero.jsx src/components/sections/ProjectCard.jsx
git commit -m "feat: hover-triggered text scramble decode effect"
```

---

### Task 12: Scroll Progress Indicator

**Files:**
- Create: `src/components/ui/ScrollProgress.jsx`
- Modify: `src/App.jsx`

- [ ] **Step 1: Create ScrollProgress component**

Fixed top bar showing scroll progress:
- Thin gradient line (cyan→violet) at top of viewport
- Width maps to scroll percentage (0% → 100%)
- `position: fixed; top: 0; z-index: 50`
- Height: 2px, expands to 3px on hover
- Smooth width transition

- [ ] **Step 2: Add to App.jsx layout**

Place above all other content.

- [ ] **Step 3: Commit**

```bash
git add src/components/ui/ScrollProgress.jsx src/App.jsx
git commit -m "feat: scroll progress indicator with neural gradient"
```

---

### Task 13: Enhanced Button & Link Interactions

**Files:**
- Modify: `src/index.css`
- Modify: `src/components/sections/Hero.jsx`

- [ ] **Step 1: Add button glow-on-hover keyframe**

```css
.btn-neural {
  position: relative;
  overflow: hidden;
}
.btn-neural::before {
  content: '';
  position: absolute;
  inset: -2px;
  background: linear-gradient(135deg, var(--accent-electric), var(--accent-violet));
  border-radius: inherit;
  opacity: 0;
  transition: opacity 0.3s;
  z-index: -1;
  filter: blur(8px);
}
.btn-neural:hover::before {
  opacity: 0.6;
}
```

- [ ] **Step 2: Apply to Hero CTA buttons**

Add `btn-neural` class to "EXPLORE NETWORK" and "DOWNLOAD CV" buttons.

- [ ] **Step 3: Add link underline animation**

Animated gradient underline that draws left-to-right on hover.

- [ ] **Step 4: Commit**

```bash
git add src/index.css src/components/sections/Hero.jsx
git commit -m "feat: enhanced button glow and link hover animations"
```

---

### Task 14: Cleanup & Remove Legacy 2D Canvas

**Files:**
- Remove: `src/components/canvas/useNeuralNetwork.js`
- Remove: `src/components/canvas/NeuralNetwork.jsx` (old 2D version)
- Remove: `src/components/effects/ParticleField.jsx`
- Remove: `src/components/effects/GlowOrb.jsx`
- Modify: `src/components/canvas/MiniNetwork.jsx` (keep as-is, still used by Synapse)

- [ ] **Step 1: Remove old 2D canvas files**

Delete `useNeuralNetwork.js`, `NeuralNetwork.jsx` (2D), `ParticleField.jsx`, `GlowOrb.jsx`.

- [ ] **Step 2: Verify no broken imports**

```bash
npm run build
```

- [ ] **Step 3: Run tests**

```bash
npm run test:run
```

- [ ] **Step 4: Commit**

```bash
git add -u
git commit -m "refactor: remove legacy 2D canvas and unused effect components"
```

---

### Task 15: Final Polish & Performance

**Files:**
- Modify: `src/components/canvas/NeuralNetwork3D.jsx`
- Modify: `src/index.css`

- [ ] **Step 1: Add mobile performance optimizations**

- Reduce node count on mobile (3→4→5→4→3 instead of 4→6→8→6→4)
- Reduce particle count (60 vs 200)
- Disable post-processing on mobile
- Use `dpr={[1, 1.5]}` on Canvas for mobile DPR capping

- [ ] **Step 2: Add loading state**

Show a subtle fade-in while Three.js scene initializes.

- [ ] **Step 3: Final lint + build verification**

```bash
npm run lint && npm run test:run && npm run build
```

- [ ] **Step 4: Commit**

```bash
git add .
git commit -m "perf: mobile optimizations, loading state, final polish"
```

---

## Execution Notes

- **Performance budget:** 3D scene must maintain 60fps on desktop, 30fps on mobile
- **Bundle size:** Three.js adds ~150KB gzipped — offset by tree-shaking drei imports
- **Accessibility:** All visual effects respect `prefers-reduced-motion`; 3D canvas is `aria-hidden="true"`
- **Fallback:** If WebGL is unavailable, show static gradient background (no crash)
- **Font loading:** New fonts loaded with `display=swap` to prevent FOIT
