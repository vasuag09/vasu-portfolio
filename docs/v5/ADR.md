# Architecture Decision Records — VASU_OS 5 "Neural Core"

> Produced 2026-06-11 by the architect agent. Six ADRs covering the decisions flagged in
> PLAN.md. Implementation note on ADR-5: the legacy `builds[]` vercel.json shape shown
> below should be implemented as TWO Vercel projects instead (main project with Root
> Directory = app-v5, plus an archive project for v4) or via the modern vercel.ts config —
> the routing intent stands, the config mechanism is updated at /implement time.

---

## ADR-1: Scroll-Authority Validation Criteria

### Context
RESEARCH.md locks native page scroll + Lenis + GSAP ScrollTrigger as the single scroll authority (not drei ScrollControls). Phase 1 prototype must validate this choice works before the remaining 11 phases build on it.

### Decision
**PASS criteria for the Phase-1 prototype:**
1. **Text selectability & indexability:** all section text is real DOM; triple-click selects paragraphs; copy-paste works; crawlable/screen-readable.
2. **Scroll input fidelity:** native momentum (trackpad/wheel inertia), keyboard (PageUp/Down, Home/End), touch momentum/pinch. No preventDefault hijacking.
3. **Frame-sync approach:** Lenis updates a ref on every RAF → R3F `useFrame` reads the ref → camera position/quaternion update. Max skew <16ms/frame at 60fps.
4. **Damping feel:** Lenis lerp ≤0.12; GSAP scrub ≤1. Subjective check: scroll→camera delay natural, not laggy/twitchy.
5. **Reduced-motion bypass:** `prefers-reduced-motion` disables camera flight entirely; content fully scrollable.
6. **No scroll-jacking:** camera never blocks default scroll.
7. **Fallback trigger:** sync skew >40ms consistently OR unselectable text OR keyboard scroll failure → fall back to drei ScrollControls (accepting loss of text-in-DOM).

### Alternatives & why not
- **drei ScrollControls + HTML overlay:** HTML scrolls in a container → text not indexable; violates spec AC. Rejected.
- **Hand-rolled scroll listener + RAF:** reinvents momentum/touch/a11y; frame-skip risk. Rejected.

### Consequences / trade-offs
Must wire Lenis + GSAP + R3F in precise order; GSAP proprietary-but-free license acceptable; no synthetic scroll polish (native momentum is the win); text stays accessible + SEO works.

### Interfaces & data model
```typescript
interface ScrollState { offset: number /* 0–1 */; velocity: number }
interface CameraUpdate { position: THREE.Vector3; target: THREE.Vector3; damping: number }
interface ScrollCameraProps {
  sections: SectionAnchor[];
  damping: number;            // Lenis lerp 0.08–0.12
  onScrollProgress: (offset: number) => void;
}
// src/hooks/useScrollCamera.ts — reads Lenis ref in useFrame, returns CameraUpdate
```

### Risks & mitigations
- Async ref update frame-skips → Lenis updates ref synchronously on RAF before useFrame; verify execution order.
- Unselectable text via z-index bugs → Phase-1 manual test: select-all + paste.
- Keyboard scroll out of sync → ScrollTrigger.refresh() after layout changes; no preventDefault in chain.

---

## ADR-2: Mobile Tier Strategy

### Context
AWARD-RESEARCH notes Vitasović ships pre-rendered video on mobile. Spec AC requires reduced tier with CWV targets met. Phase 8 measures; this ADR sets the framework and default.

### Decision
**Hybrid: live-scene primary, video fallback only for devices that fail the live baseline.**
1. **Live tier (default):** WebGL-capable modern devices. ~10k particles (vs 60–100k desktop), bloom only (no DoF), precomputed curl-noise texture. Camera still flies. LCP <2.5s via dynamic Canvas import after FCP.
2. **Video tier (fallback):** WebGL init failure or low-end detection (`navigator.deviceMemory` ≤4GB or context loss). Pre-rendered scene loop (~1.2MB, H.264 + VP9, lazy). Interactive elements stay DOM; optional scrub tied to `video.currentTime`.
3. **Maintenance:** one video re-capture per content cycle, not per edit. (Captured from the real scene, not Veo.)

### Alternatives & why not
- **Video-only on mobile:** loses interactivity differentiators. Rejected.
- **Live-only:** crash on old devices = score-killer. Rejected.
- **Force-directed-for-cheapness:** nondeterministic + heavy. Rejected.

### Consequences / trade-offs
Two code paths (~200 LOC flag logic); video is static for fallback users (acceptable — spec says "reduced tier"); live interactivity preserved as hero experience.

### Interfaces & data model
```typescript
type SceneMode = 'live' | 'video' | 'static'; // static = reduced-motion
interface MobileStrategy {
  useVideoFallback: boolean; videoSrcUrl: string; videoLoopDuration: number;
  particleCountMobile: number; particleCountDesktop: number; bloomOnly: boolean;
}
```
Phase-8: PerformanceObserver logs `{mode, LCP, INP, CLS, deviceType}`; if >10% of traffic hits video tier with CLS >0.1, pause and debug.

### Risks & mitigations
- Codec support → ship H.264 + VP9; Safari falls back to H.264; test iPhone-class device.
- Stale video vs new content → version files (`scene-v2.mp4`); flag at content changes.
- 10k particles still janky → drop to 5k, then force video tier.

---

## ADR-3: Scene-Graph Data Model

### Context
v4's network is decorative. v5 needs ~5 sections, ~30 projects, ~40 skills, skill→project edges, with deterministic positions (SSR/hydration-safe — no runtime Math.random()).

### Decision
**Build-time hand-authored section anchors + deterministic seeded scatter (hash-based PRNG, e.g., Mulberry32 keyed on node id).** Camera spline = CubicBezierCurve3 chain through section anchor rest-poses, built by `buildCameraSpline(anchors)`.

### Schema
```typescript
interface SectionAnchor {
  id: string; label: string;
  cameraPos: [number, number, number];
  cameraTarget: [number, number, number];
}
interface ProjectNode {
  id: string; sectionId: string;
  position: [number, number, number]; // derived: cluster placement + seeded jitter
  color: string;                       // OKLCH token
  scale: number; flagship: boolean;
}
interface SkillNode {
  id: string; label: string;
  position: [number, number, number];
  category: string; color: string;
}
interface Edge { skillId: string; projectId: string; strength: number /* 0–1 */ }
```
Files: `src/data/sections-v5.ts` (anchors), positions computed at build/module-load from `seededRandom(id)` + `placeInCluster(anchor, index, total, seed)`.

### Alternatives & why not
- **Runtime force-directed (d3-force):** nondeterministic across devices, ~40kb, hydration mismatch. Rejected.
- **Fully procedural (Fibonacci sphere etc.):** loses intentional region semantics. Rejected.

### Consequences / trade-offs
One-time hand-authoring of 5 anchors; seeded scatter may crowd — fix via per-node explicit offsets or one-pass build-time Lloyd relaxation; no runtime solver cost; SSR-safe.

### Risks & mitigations
- Crowding → explicit offsets in data file.
- Spline clipping through geometry → debug spline visualization in Phase 1.
- Anchor drift as content grows → anchors documented + locked in Phase-0 design review.

---

## ADR-4: DOM↔Canvas Integration

### Context
Recruiters need selectable/searchable/screen-readable text; the showpiece needs a unified 3D world. Spec mandates text as real DOM.

### Decision
**Fullscreen fixed Canvas behind (z-index -1, `pointer-events: none`, `aria-hidden`) + real scrolling DOM sections on top (z-index 1). ONE global ScrollTrigger maps document scroll 0–1 to camera spline progress. DOM→Canvas hover effects via event emitter updating shader uniforms (uActive/uHover).**

```html
<div class="canvas-container" style="position:fixed; inset:0; z-index:-1">
  <Canvas aria-hidden="true">{scene}</Canvas>
</div>
<main style="position:relative; z-index:1">
  <section id="hero">…</section> …
</main>
```
- Canvas default `pointerEvents:"none"`; enable `"auto"` ONLY on canvas elements that genuinely need direct interaction (e.g., Synapse node), or route entirely from DOM handlers.
- Text reveals animate opacity/translateY only; fixed/bounded section heights → CLS safe.
- Section→camera mapping: single trigger `start:"top top" end:"bottom bottom"`; `useFrame` reads progress, samples spline. No per-section pins in Phase 1 (add only if case studies need them in Phase 5).

### Alternatives & why not
- **Canvas in container:** scene disappears when scrolled past — breaks the world metaphor. Rejected.
- **drei Html for all text:** CLS, screen-reader gaps, fragile copy-paste. Rejected.
- **Canvas-rendered text (Troika):** violates real-DOM AC. Rejected.

### Interfaces
```typescript
interface CanvasNodeHoverState { nodeId: string; isHovered: boolean; worldPos: [number,number,number] }
interface ScrollCameraContext {
  progress: number; offset: number; sectionId: string;
  cameraPos: THREE.Vector3; cameraTarget: THREE.Vector3;
}
```
Files: `components/canvas/Canvas.tsx`, `components/layout/Section.tsx`, `hooks/useScrollCamera.ts`, `lib/scroll-events.ts` (DOM→Canvas emitter).

### Risks & mitigations
- Pointer fall-through → default none; opt-in per interactive node.
- Section↔spline desync when heights change → re-measure + ScrollTrigger.refresh() after Phase-5 layout.
- Screen readers + canvas → `aria-hidden`, all content in DOM.

---

## ADR-5: v4→v5 Cutover Strategy

### Context
v4 = Vite + Express SPA on Vercel. v5 = Next.js 16. Conflicting build systems can't share one package.json. Domain: vasuai.dev; v4 archives to v4.vasuai.dev.

### Decision
**Option A: new `app-v5/` directory in the same repo with its own package.json and Next.js config; v4 stays at root untouched until /ship.**

```
vasu-portfolio/
├── package.json     (v4 Vite — unchanged)
├── src/  api/  …    (v4 — unchanged)
├── app-v5/          (v5 Next.js: own package.json, app/, public/, tsconfig)
└── docs/v5/
```
- Dev: `npm run dev` (v4, :5173) and `cd app-v5 && npm run dev` (:3000) coexist.
- Deploy (modern form per header note): **two Vercel projects** — main project Root Directory = `app-v5` → vasuai.dev; archive project (v4 build) → v4.vasuai.dev. 301s for any public v4 routes added in `app-v5/next.config.js`.
- Git: phases as `feat(v5/phase-N): …` commits; at /ship tag `v4-final`, branch `archive/v4-eol`.
- Rollback: repoint vasuai.dev back to the v4 project (~2 min), monitor 48h.

### Alternatives & why not
- **Long-lived branch:** awkward dual-local-dev, merge pain, delays. Rejected.
- **New repo:** complicates DNS cutover + history. Rejected.
- **Replace root, delete v4:** loses archive + rollback. Rejected.

### Consequences / trade-offs
Two package.jsons/node_modules (trivial cost); v4 frozen post-archive; fully reversible cutover; clear ownership boundaries.

### Risks & mitigations
- Wrong Node version on Vercel → `.nvmrc` in both roots; preview-test before ship.
- Route collisions → preview tests: `/` → v5, archive domain → v4.
- v4 asset loss → assets remain under root; v5 assets under `app-v5/public/`. No overlap.

---

## ADR-6: Reduced-Motion Layout Model

### Context
Spec AC: with `prefers-reduced-motion`, no camera flight or particle churn; all content reachable. Static-tiled grid vs normal column?

### Decision
**Normal tall column — same vertical section flow as motion users, no camera flight, static hero composition. ONE layout tree.** Canvas hidden (`display:none`) or single static frame; project/skill content renders as DOM cards/chips; reveals become single fade-ins or none.

```css
@media (prefers-reduced-motion: reduce) {
  .canvas-container { display: none; }
  * { animation: none !important; transition: none !important; }
}
```
```typescript
export function useReducedMotion(): boolean { /* matchMedia hook with change listener */ }
// Rule: every GSAP tween lives inside a motion-aware hook that guards on useReducedMotion().
// Lint convention: no bare gsap.to() outside motion-aware hooks.
```
Files: `hooks/useReducedMotion.ts`, `styles/reduced-motion.css` (centralized overrides).

### Alternatives & why not
- **Static tiled grid (all sections at once):** breaks narrative pacing, overwhelming, second layout tree to maintain. Rejected.
- **Keep particles visible but static-ish:** floating dots still trigger vestibular issues. Rejected.

### Consequences / trade-offs
Single layout tree (maintenance win, CLS-safe, identical recruiter path); both audiences share the same scroll journey rather than per-mode optimization (acceptable per spec).

### Risks & mitigations
- Tweens leaking past the guard → centralize via motion-aware hook + lint convention.
- OG images depend on canvas → generate static OG at build time; canvas is decorative.
- Third-party motion not respecting the pref → Phase-7 dependency audit.

---

## Cross-dependencies
ADR-4's pointer routing feeds ADR-1's frame-sync tests · ADR-3's anchors feed ADR-4's section mapping and ADR-1's spline · ADR-2's video tier reuses ADR-4's Canvas component · ADR-6's motion guard applies to every animation site-wide · ADR-5 scopes where all of it lives (`app-v5/`).

---

## ADR-7: Per-Chapter Particle Scene Identity (design elevation, 2026-06-12)

### Context
Audit finding P0.3: the 80k stateless curl-noise field is uniform confetti — every chapter looks identical. Must stay ONE GPU system, one draw call, stateless, 60fps, inherited by the 10k mobile tier.

### Decision
**Spawn-bias + per-particle proximity modulation.** Build-time spawn distribution re-seeded to cluster density around each chapter's world-space core (`sections[i].cameraTarget`); vertex shader gains `uChapterCenters[5]` + `uScrollProgress` and modulates size/brightness/turbulence by distance-to-nearest-core, weighted toward the active chapter.

### Alternatives & why not
- Multiple particle systems: 5 draw calls + JS visibility management. Rejected.
- Morph-to-formation targets: per-particle target attributes + morph math at 80k = cost without atmosphere gain. Rejected.

### Consequences
Deterministic, SSR-safe, zero new draw calls; density baked at build (anchor moves require re-seed); bloom tuning needed (`uNearbyBrightness` start 1.5 — the threshold-1.0 selective bloom must not wash out).

### Risks
O(5) loop per vertex is cheap; NaN guards untouched; verify "fine without bloom" after every shader edit.

## ADR-8: Graph Node Labels via DOM Overlay (design elevation, 2026-06-12)

### Context
Audit finding P1.4: nodes unlabeled/unreadable. ~10–15 labels (flagships + key skills), decorative/duplicative of the DOM lists → `aria-hidden`.

### Decision
**DOM-overlay labels** projected from world positions (`vector.project(camera)`) each frame into a fixed `pointer-events:none aria-hidden` layer; opacity fades by camera distance; culled when behind camera; skipped entirely on the mobile tier.

### Alternatives & why not
- drei `<Text>` (troika): ~50kb gz into the scene chunk for decorative text. Rejected.
- Canvas sprite atlas: shader/material complexity for 15 strings. Rejected.

### Consequences
Zero bundle cost, crisp at any DPR, real HTML, CSS-styled; costs ~15 projections/frame (throttle if >1ms); transform-only positioning (compositor-safe).

---

## ADR-9: Camera Authority for the Neuron Dive (2026-06-12)

### Context
CameraRig is the sole camera writer (scroll → spline pose per frame, ADR-1).
The dive needs the camera at a project node while the panel is open, then back.
Two-owner designs (rig pauses, tween library takes over) race on rapid
open/close and on scroll-lock edges.

### Decision
**Single writer, blended authority.** A `camera-state.ts` module singleton
(same pattern as scroll-state) holds `diveTarget: CameraPose | null` and
`diveBlend: 0..1`. CameraRig computes the scroll pose EVERY frame, then lerps
position/target toward diveTarget by an exponentially-damped diveBlend
(target 1 while diving, 0 otherwise). Open/close/rapid-toggle are just blend
retargets — the camera can never be stranded because the scroll pose is
always live underneath. Deep links set blend=1 before first frame (end state,
no replay). Dive pose derived per node: camera at node center + offset along
(restPose→node) direction at radius×k, looking at the node.

### Alternatives & why not
- GSAP tween of camera + rig pause/resume: two owners, kill/overwrite races,
  fights ADR-1's "no extra smoothing" rule. Rejected.
- Extending the spline with a dive segment: per-node spline rebuilds at open
  time, pollutes scroll mapping. Rejected.

### Consequences
Dive easing = exp damping (matches activation glow idiom, ~9/s rate);
panel UI keys off diveBlend threshold; reduced-motion path untouched (no
canvas). One extra lerp per frame — negligible.
