# Research: build-vs-reuse for VASU_OS 5 "Neural Core"

> Verified 2026-06-11 via in-repo scan, context7, npm registry, GitHub search.

## Reuse decision

### Port from v4 (in-repo — biggest win)
- **`src/components/canvas/`** — v4 already contains a working R3F neural network:
  `NeuralNetwork3D.jsx` (161 LOC), `NetworkNodes.jsx` (197), `NetworkParticles.jsx` (130),
  `SkillOrb.jsx`, `MiniNetwork.jsx`, plus **custom GLSL shaders** (`nodeMaterial.js` —
  fresnel rim-glow with uActive/uHover pulse; `connectionMaterial.js`). These are the seed
  of the v5 scene: keep the shaders, rewrite layout/data so the graph derives from
  projects/skills data instead of decorative layers.
- **`api/ai.js`** — Gemini proxy already a Vercel serverless function with rate limiting +
  input validation. Ports to a Next.js route handler nearly verbatim.
- **`SynapsePanel.jsx`** (180 LOC), `BootSequence.jsx`, effects (`TextScramble`,
  `GlitchText`, `CustomCursor`) — port with re-skin.
- **Not portable:** `CameraRig.jsx` is mouse-orbit only — v5's scroll-driven camera spline
  is new. `useNeuralNetwork.js` is the 2D `<canvas>` version — superseded.

### Adopt (libraries)
| Library | Version (verified) | Role | Why |
|---|---|---|---|
| `@react-three/fiber` | 9.6.1 (peer: react >=19 <19.3 ✓ matches React 19.2) | 3D renderer | Already in v4; v10 is still a branch — stay on 9 |
| `@react-three/drei` | 10.7.7 | **`ScrollControls` + `useScroll`** (scroll→0..1 offset with damping, `range/curve/visible` helpers) and **`MotionPathControls`** (camera along CubicBezier curves with `focus` + `damping`) | Purpose-built for our guided camera; no hand-rolled scroll-sync |
| `@react-three/postprocessing` | ^3 (in v4) | Bloom, vignette, grain | Already proven in v4 |
| `gsap` + `@gsap/react` | 3.15.0 / 2.1.2 — **"Standard no-charge license": GSAP incl. all plugins (ScrollTrigger, MotionPath, DrawSVG, MorphSVG) is now free** | DOM-side scroll choreography (`useGSAP` hook with `scope`, `scrub`), boot sequence timeline | Free since Webflow acquisition; premium plugins no longer gated |
| `lenis` | 1.3.23 | Smooth scroll on the DOM scroller (if we drive the camera from page scroll instead of drei's ScrollControls container) | Industry standard for showpiece sites |
| `next` | 16.2.9 | Framework on Vercel | Per spec; R3F mounts in a `'use client'` component, scene `dynamic()`-imported after first paint (meets JS budget) |

**Architecture choice the plan must make explicit:** ONE scroll authority. Either
(a) drei `ScrollControls` owns scroll (camera + HTML inside its container), or
(b) native page scroll + Lenis + GSAP ScrollTrigger drives `useScroll`-equivalent offset
into the Canvas. Recommendation: **(b)** — real DOM pages keep text indexable/accessible
(spec acceptance criterion) and GSAP orchestrates DOM and camera from one timeline.
Do NOT mix both — fighting scroll owners is the classic failure mode.

### Do NOT adopt
- `pmndrs/react-three-next` starter — last pushed 2024-06, stale (pre-Next-15/16,
  pre-React-19). Wire R3F into Next manually (~30 lines).
- Wholesale neural-network repos — GitHub search found only ≤1-star experiments. Nothing
  at 80% fit. The scene is the genuinely novel part; v4's own canvas code is the best seed.

### Build new (the novel ~20%)
1. **Data-driven graph layout** — projects/skills/sections as nodes+edges from typed data
   files; force-directed or hand-authored cluster positions baked at build time.
2. **Scroll-camera spline** — CubicBezierCurve3 path through region anchors, scrub-driven,
   with per-section "rest poses" (via drei MotionPathControls or a thin custom rig).
3. **GPU particle field** — instanced points around clusters (v4 shaders extended);
   tiered counts (mobile/reduced-motion).
4. **Veo clip pipeline** — Flow-generated clips for FundlyMart, NM-GPT, GeoVision;
   compressed (AV1/H.265 + H.264 fallback), lazy-loaded in case-study panels.

## Key API facts the plan must respect (from context7)
- drei `useScroll`: `data.offset` (0–1 damped), `data.range(start, len)`,
  `data.curve(start, len)` (0→1→0), `data.visible(start, len)` — all read inside
  `useFrame`, never in render.
- drei `MotionPathControls`: accepts declarative `<cubicBezierCurve3 v0..v3/>` children or
  `curves={[new THREE.CubicBezierCurve3(...)]}`; `offset` drives position, `focus` aims
  the camera, `damping` smooths.
- GSAP React: register `useGSAP` plugin; always pass `{ scope: containerRef }`; wrap event
  handlers in `contextSafe()`; call `ScrollTrigger.refresh()` after layout changes.
- R3F 9.6.1 pins `react >=19 <19.3` — do not jump to a future React minor without checking.
- Next.js: Canvas component must be `'use client'`; load three/R3F via `dynamic(() =>
  import(...))` so the landing JS budget (<150kb before 3D libs) holds.

## Risks of chosen dependencies
- **drei** (10.x): fast-moving major versions; pin minor, read changelog before bumps.
- **GSAP**: license is "no charge" but proprietary (not OSI); fine for a personal site.
- **lenis**: small (~8kb), healthy maintenance; replaceable if abandoned.
- **R3F v10 on the horizon**: branch exists; budget a migration later, don't chase it now.
- **Veo/Flow**: 8s clip ceiling and weak cross-clip continuity — design case-study clips as
  self-contained moments, not a continuous film (consistent with spec's "garnish" stance).
