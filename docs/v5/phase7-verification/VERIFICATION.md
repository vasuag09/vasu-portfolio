# Phase 7 Verification — Sound Layer, OG Metadata, A11y Polish

Date: 2026-06-11 · Production build · Evidence: `gate.png`,
`lighthouse-mobile-gate.report.html`, results inline below.

## Asset decision

All audio is **synthesized at runtime via Web Audio** (`lib/sound-engine.ts`):
boot jingle (rising triangle arpeggio), ambient pad (two detuned low
oscillators + LFO-driven lowpass), and 4 SFX (hover tick, click blip,
open/close sweeps). Zero network weight, zero licensing, and the phosphor
voice matches the terminal aesthetic. Licensed files can replace any voice
later by swapping the `play*` internals — the public API stays.
**Vasu: subjective listen pending** (headless verification covers state and
wiring, not taste).

## Exit criteria vs verified

| Criterion | Verified |
|---|---|
| Sound choice persists | Gate choice → `v5:sound=1`, toggle shows "snd on"; toggle click flips to `0` / "snd off"; survives reload |
| Two-button gate, sound primary | Dialog with "Enter with sound" (accent, focused by default) + "Enter muted"; click is the Web Audio unlock gesture (autoplay/iOS) |
| No autoplay | Engine fully inert until a user-gesture `unlock()` |
| LinkedIn/social rich card | `og:title/description/image` (absolute `https://vasuai.dev/og.png`, 1200×630 from the real scene) + `twitter:summary_large_image` in served HTML; `/og.png` → 200 |
| AA contrast / a11y | axe-core 4.10.2: **0 violations** in all three states (default, case-study panel, Synapse terminal) after the `aside→div` dialog fix; Lighthouse a11y 100 |
| Visible focus everywhere | 12-stop Tab journey: every stop shows the 2px accent outline (`:focus-visible` global) |
| Full keyboard journey | Tab order follows DOM (toggle → hero trigger → projects → skill chips…); Esc paths verified in Phases 5/6/10 |

## Flow matrix (Playwright, production)

| Path | Result |
|---|---|
| First visit | Gate inside boot overlay; primary focused; choose sound → jingle + scramble → `booted=1` |
| Esc during gate | Enters silently, **no** sound choice persisted (toggle remains the path back), boot marked seen |
| Return visit | No gate, no boot; toggle remembers last choice; ambient (if on) resumes on first gesture via SoundController unlock listener |
| Reduced motion | No gate ever (no boot overlay); toggle available; no autoplay |
| Overlay SFX | open/close sweeps derived from graph-store transitions (`soundEventsForChange`, unit-tested) |

## Performance gates re-checked (gate visible — worst case)

perf **96** · FCP 756 ms · LCP 2468 ms · TBT 152 ms · CLS **0.001** · a11y/bp/seo **100**

(The Phase-10 boot-text CLS note resolved itself: the gate's buttons render
with the overlay from SSR, so the swap reflow no longer moves the timeline
LCP element.)

## Implementation notes

- `lib/sound-state.ts` — pure: preference parsing, gate predicate, store-
  transition → SFX derivation. 9 new tests (82/82 green).
- `lib/sound-store.ts` — graph-store-shaped preference store; SSR snapshot
  "unset"; persists + drives the engine.
- `components/sound/SoundController.tsx` — gesture unlock, store
  subscription for open/close, hover/click delegation (90 ms throttle) on
  `INTERACTIVE_SELECTOR`.
- `components/sound/SoundToggle.tsx` — fixed bottom-left, `aria-pressed`,
  its own click doubles as the unlock gesture.
- BootSequence hosts the gate: real `role="dialog"` while gating (focusable
  buttons), decorative `presentation` during the scramble. CSS dead-JS
  failsafe stands down once hydration tags `data-hydrated` (the gate may
  legitimately hold the overlay open).

## Carried

- Phase 9/11: re-listen on a real device; consider licensed audio upgrade.
- Phase 12 CSP: `connect-src`/inline-script notes unchanged; no new origins
  (CDN axe was test-only).
- OG image: regenerate after any hero/scene redesign (`public/og.png`,
  capture recipe in session notes).
