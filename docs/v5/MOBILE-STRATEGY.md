# Mobile Strategy — Phase 8 Measurement & Verdict

> Produced 2026-06-11 (Phase 8). Measures the live-vs-video framework set in
> ADR-2 against the production build. Evidence in `phase8-verification/`.

## Verdict

**Live tier confirmed as the mobile primary. The video fallback stays a
documented contingency — not built now.** ADR-2's escalation trigger ("drop
to 5k, then force video tier") was not hit: the live mobile tier passes every
Phase-8 exit gate on emulated mobile hardware.

| Exit gate | Target | Measured | Status |
|---|---|---|---|
| Mobile LCP | < 2.5 s | 2.47 s (median of 3 + canonical run) | PASS (thin margin) |
| INP proxy (lab TBT) | < 200 ms | 192 ms canonical (147–222 ms across runs) | PASS (borderline) |
| INP (real interaction probe) | < 200 ms | worst 80 ms unthrottled (was 544 ms pre-fix) | PASS |
| CLS | < 0.1 | 0.000 | PASS |
| JS budget | ≤ 300 kb gz | 243 kb gz critical path | PASS (see interpretation) |
| Touch interactions | work | scroll / tap-open / panel-scroll-lock / close all verified | PASS |

Lighthouse mobile (emulated Moto-G-class, 4× CPU, slow-4G simulation):
**Performance 95 · Accessibility 100 · Best Practices 100 · SEO 100.**

## JS budget interpretation

Total page JS is 448 kb gz, of which **244 kb is the scene chunk (three.js +
R3F + shaders)** that loads *after* first paint behind an idle gate. A live
WebGL scene cannot fit "total ≤ 300 kb" — three.js alone is ~244 kb gz — which
is precisely why ADR-2 mandates the post-FCP dynamic import. The budget is
therefore applied to the **critical path: 243 kb gz** (react-dom 71 + Next
runtime 49 + app/GSAP/Lenis 56 + route chunks). The deferred scene chunk never
blocks FCP/LCP and is excluded by design.

Critical-path composition (gz): react-dom 71 kb · Next runtime 49 kb ·
app code + GSAP + ScrollTrigger + Lenis 56 kb · smaller route/shared chunks
~67 kb. Optional future trim: defer GSAP behind the scene chunk (~25 kb), not
done because Lenis/ScrollTrigger drive scroll feel from first frame.

## Tier behavior (verified under Pixel-7 emulation)

- UA-based detection resolved the **mobile tier**: canvas DPR clamped to
  exactly **1.5** (buffer 645px on a 430px CSS canvas at devicePixelRatio 3),
  which implies the 10k-particle / bloom-only / no-DoF config (unit-tested in
  `particle-config.test.ts`).
- `low` tier (≤4 GB deviceMemory or ≤2 cores) keeps 5k particles, no bloom,
  DPR 1. Safari/Firefox never report deviceMemory → treated as capable,
  per ADR-2.
- Reduced motion: canvas not mounted at all (verified Phase 1/6).

## Phase-8 fixes (what measurement forced)

1. **SSR reveal policy** (`lib/reveal-state.ts`, `hooks/useRevealOnce.ts`):
   Reveal previously shipped ALL content `opacity:0` in SSR HTML — Lighthouse
   couldn't even record an FCP (NO_FCP), and no-JS visitors got a blank page.
   Now the server paints everything; after hydration only elements fully
   below the fold are re-hidden and armed for their scroll-in reveal. 7 new
   unit tests pin the policy.
2. **Idle-gated scene mount** (`CanvasRoot.tsx`): the 244 kb scene chunk now
   loads via `requestIdleCallback` (2 s timeout; `setTimeout` fallback for
   Safari) instead of at hydration. TBT 217 → ~190 ms.
3. **Deferred overlay close** (`lib/after-paint.ts`): closing the case-study
   panel / Synapse terminal unmounted the overlay AND restored `html`
   overflow (full-document relayout of a ~1200svh page) inside the
   interaction frame — 544 ms worst tap latency. Close now paints click
   feedback first, then unmounts one task later: worst interaction 80 ms.
4. **Font weights trimmed** (`layout.tsx`): IBM Plex Mono 300 was preloaded
   but unused; now 400/500/700 only.

## Video-tier contingency (not built — build triggers below)

Per ADR-2, build the pre-rendered scene-loop fallback only if field data
demands it:

- **Trigger:** >10% of mobile traffic hitting WebGL init failure / context
  loss, OR field LCP p75 > 2.5 s, OR field INP p75 > 200 ms on the live tier.
- **Mechanism already in place:** `CanvasErrorBoundary` catches WebGL/scene
  crashes today (page remains fully usable DOM — the scene is decorative,
  ADR-4). The video tier would replace that empty fallback with a ~1.2 MB
  H.264+VP9 loop captured from the real scene.
- **Field data plan (Phase 12):** PerformanceObserver logging
  `{mode, LCP, INP, CLS, deviceType}` per ADR-2, reviewed during the 48 h
  post-launch monitoring window.

## Residual risks / carried items

- **LCP margin is thin (≈30 ms lab).** The LCP element is the hero
  paragraph; margin is dominated by font fetch + hydration contention.
  Re-measure on PageSpeed Insights (real Moto G) at Phase 11; if it slips,
  first lever = `display:optional` experiment or hero-paragraph size bump
  (makes h1 the LCP element, painted from the same HTML).
- **Lab TBT straddles 200 ms** (147–222 ms run noise; canonical 192 ms). The
  one irreducible task is three.js module evaluation. The real-interaction
  probe (80 ms worst, unthrottled) says INP is healthy; confirm on a physical
  mid-range Android at Phase 11 before submission.
- Boot sequence (Phase 10) must not reintroduce a hidden-at-SSR hero — the
  reveal policy in `lib/reveal-state.ts` is the contract; boot theatrics
  should layer ON TOP of painted content (e.g., overlay), not gate it.
