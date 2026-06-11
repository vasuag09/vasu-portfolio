# Phase 8 Verification — Mobile Tier & Performance Budget

Date: 2026-06-11 · Production build (`next build` + `next start`, Next 16.2.9)

## Exit criteria vs measured

| Criterion | Target | Measured | Evidence |
|---|---|---|---|
| Mobile LCP | < 2.5 s | 2465 / 2467 / 2477 ms (3 runs) · 2472 ms canonical | `lh-run*.json`, `lighthouse-mobile.report.html` |
| INP < 200 ms | lab proxy TBT | TBT 192 ms canonical (147–222 noise); real-interaction probe worst **80 ms** | this file, §interaction probe |
| CLS | < 0.1 | **0.000** every run | Lighthouse reports |
| JS ≤ 300 kb gz | critical path | **243 kb gz** initial (scene chunk 244 kb deferred post-FCP behind idle gate) | §bundle |
| Touch interactions | work | scroll, tap-open, panel scroll-lock, close — all pass under Pixel-7 emulation | §touch, screenshots |

Lighthouse categories (canonical run): **Performance 95 · A11y 100 · Best Practices 100 · SEO 100.**

## Bundle (gzipped, measured over the wire)

| Chunk | gz |
|---|---|
| scene (three.js + R3F, idle-deferred) | 244.0 kb |
| react-dom | 69.3 kb |
| app code + GSAP + ScrollTrigger + Lenis | 54.6 kb |
| Next runtime | 48.0 kb |
| shared/route + turbopack runtime | ~26.9 kb |
| **Initial (HTML-referenced) total** | **242.9 kb** |
| **Total incl. deferred scene** | **447.7 kb** |

## Touch & tier probe (Playwright, Pixel-7 emulation: 390×844, DPR 3, touch, Android UA)

- Tier: canvas backing buffer 645 px / 430 px CSS = **DPR 1.5 clamp** → mobile tier (10k particles, bloom-only) resolved correctly.
- Touch scroll (CDP `synthesizeScrollGesture`, real touch events — NOT `window.scrollTo`): page 0 → 800 px. Camera rig driven by Lenis follows.
- Tap project → case-study panel opens (`role="dialog"` appears).
- Panel swipe: panel `scrollTop` 0 → 700 while page stays frozen at 2430.5 px (scroll lock holds under touch).
- Close tap → panel unmounts. Repeated open/close cycles stable.
- Screenshots: `mobile-panel-open.png`, `mobile-after-close.png`.

## Interaction-latency probe (event-timing API, durationThreshold 16 ms)

| Interaction | Before fixes | After |
|---|---|---|
| Panel open (tap) | 72–80 ms | 80 ms |
| Panel close (tap) | **296–544 ms** | **48–56 ms** |

Root cause of slow closes: overlay unmount + `html` overflow restore (full
relayout of ~1200svh document) inside the interaction frame. Fixed by
deferring the store write past the next paint (`lib/after-paint.ts`).

## Fixes shipped in this phase

1. `lib/reveal-state.ts` + `hooks/useRevealOnce.ts` — SSR ships visible HTML;
   only below-the-fold content re-hides for scroll-in reveals. Before: every
   section `opacity:0` in SSR → Lighthouse NO_FCP (could not record ANY paint).
   7 new unit tests (51/51 green).
2. `components/canvas/CanvasRoot.tsx` — scene chunk mount gated on
   `requestIdleCallback` (2 s timeout, Safari setTimeout fallback).
3. `lib/after-paint.ts` + CaseStudyPanel/SynapseTerminal close paths —
   deferred overlay close (INP fix above).
4. `app/layout.tsx` — dropped unused IBM Plex Mono 300 weight (one fewer
   preload ahead of LCP).

## Verdict

ADR-2 live-tier strategy **confirmed** — no video-tier escalation triggered.
Full decision record: `../MOBILE-STRATEGY.md`. Carried to Phase 11: real-device
PageSpeed/INP spot check (lab LCP margin ≈30 ms; TBT noise straddles 200 ms).
