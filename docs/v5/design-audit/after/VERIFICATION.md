# Design Elevation — Verification vs Spec

Date: 2026-06-12 · Production builds throughout · Pipeline: /spec → /plan →
/architect (ADR-7/8) → /implement (TDD) → /review (code-reviewer agent:
approve; 1 HIGH + 2 MEDIUM fixed) → /test → /verify.
Commits: 1b36afb, fdae683, c0c88b4, c5f267b, + review-fix commit.
Security review: skipped with rationale — no new input handling, network
calls, auth, or storage beyond the existing localStorage booleans.

## Spec acceptance criteria — all met

| Criterion | Result |
|---|---|
| Nav landings show heading in viewport (5 chapters × 3 input paths) | 14/14 pre-refinement; after heading-anchor + fonts.ready re-anchor: all 5 chapters land at exactly topFrac 0.22 |
| 5 layouts distinct (canvas hidden), ≥3 compositions | hero asymmetric / projects+about split-editorial / skills end-aligned / contact centered-oversized — screenshots `layout-*.png` |
| 5 scenes distinct (DOM hidden) | hero luminous core / projects labeled graph / skills constellation (+about/contact) — `scene-*.png` |
| Flagships ≥2× archive; ≥3 labels readable; hover <200ms | scale 0.52 vs 0.24; 4 labels readable at projects rest (`labels-projects.png`); activation damping unchanged (9/s ≈ 110ms visible) |
| No line spans >60% frame; depth fade | curved polylines + edgeLengthFade (long edges idle at 0.12) + camera-depth fade — streaks gone (`scene-projects.png`) |
| Titles ≥4rem @1440; ≥3 stagger delays | 101px (6.3rem); stagger 0/110/230ms + hero 0/140/240 |
| Veo posters on journey | 3 teasers on flagship rows; hover mounts the loop (verified video element) |
| Boot convergence ≤1.5s, paths intact | 26 glyphs converge during scramble (`boot-burst.png`); timeline constants untouched; gate/skip/return verified |
| Scroll p95 ≤18ms, zero >33ms | **avg 8.3ms (~120fps), p95 9.1ms, worst 9.4ms, 0 >33ms** (was avg 15.2 / p95 25) |
| Existing gates hold | 95/95 tests · axe 0 violations (fixed new aria-prohibited-attr) · Lighthouse perf 93–95, LCP ~2.47s, TBT 181–251 (pre-existing noise band), CLS 0.001, a11y/bp/seo 100 · reduced-motion: canvas+labels not mounted, hero visible |

## Perf lever outcomes (P2)
MSAA 8→4 (no visible jaggies — `msaa4-spheres.png`), BokehPass removed,
DPR cap 1.75. **Adaptive scroll-DPR tried and removed**: each setDpr resize
reallocated the composer target (~175ms spike) — worse than the disease.

## Carried to Phase 11
- TBT lab noise still straddles 200ms (irreducible three.js eval); real-device check stands.
- Veo loop-restart seams (fundlymart/geovision) — crossfade decision.
- Subjective pass by Vasu: hero core brightness, label density, stagger feel, sound palette.
