# Phase 10 Verification — Boot Sequence, Custom Cursor, Nav Polish

Date: 2026-06-11 · Production build · Evidence: screenshots in this folder,
`lighthouse-mobile-boot.report.html`.

## Exit criteria vs measured

| Criterion | Target | Measured |
|---|---|---|
| Boot duration | ≤ 1.5 s | overlay cleared ~1.30 s after load (timeline constant 1.4 s, unit-tested ≤1.5 s) |
| Boot skippable | any input | overlay gone **10 ms** after keypress; localStorage still set |
| Return-visit skip | no boot | `data-boot` never set; overlay `display:none` **pre-paint** (inline script) |
| Cursor morphs | hover states | `default → interactive` on hover over button; `cursor:none` active; pressed state wired |
| All nav inputs smooth | keys/dots/links | number key `3` flew 0→4500 px (damped 1.4 s Lenis); deep link `?section=about` landed at top = 0 px (END state, no replay) |

## Architecture notes

- **Pre-paint skip decision**: inline `<script>` as first child of `<body>`
  reads `localStorage["v5:booted"]` + `prefers-reduced-motion` and sets
  `html[data-boot="play"]` during parsing. CSS shows the SSR overlay only
  under that attribute → no flash in either direction; **no-JS visitors
  never see the overlay** (script never runs → hidden).
- **Phase-8 contract honored**: hero `opacity: 1` underneath the overlay
  while boot plays (verified) — boot layers on top of painted content, LCP
  element is still the hero paragraph (2457 ms).
- **Hydration-failure failsafe**: pure-CSS animation clears the overlay at
  2.5 s even if JS dies after the inline script.
- **Jingle hook for Phase 7**: `v5:boot-complete` CustomEvent dispatched on
  finish/skip (`BOOT_COMPLETE_EVENT` in `lib/boot-state.ts`).
- **Cursor capability gate** (`lib/cursor-state.ts`): fine pointer + hover +
  motion allowed. Verified: not mounted under reduced motion; not mounted on
  touch (native cursor untouched). Position via rAF refs (dot snaps, ring
  lerps 0.22); morphs via CSS on inner spans so JS and CSS never fight over
  one transform.
- **Number-key guards**: suppressed while typing (verified: "22" into the
  Synapse input moved nothing), while a `[role="dialog"]` is open, during
  boot (whose skip listener shares the window), and with meta/ctrl/alt.

## Performance gates re-checked (boot overlay active — clean-storage run)

perf **96** · FCP 753 ms · LCP 2457 ms · TBT **123 ms** · CLS 0.021 · a11y 100

CLS 0.021 (was 0.000) comes from the IBM Plex swap re-flowing the overlay's
centered text on fast connections; first-visit-only and far under the 0.1
gate. Carried note: if Phase 11 wants CLS back to zero, pin the boot name to
the system mono stack (no webfont dependency).

## Tests

73/73 green (22 new): `boot-state` (timeline ≤1.5 s, phase mapping, scramble
determinism/spaces/clamping), `cursor-state` (capability gate, mode
precedence), `nav-keys` (digit mapping bounds).

## Carried to later phases

- Phase 12 CSP: the inline boot script needs a nonce or hash in the
  production CSP header.
- Phase 7: subscribe to `BOOT_COMPLETE_EVENT` for the jingle; SoundGate may
  want to render inside the boot overlay's final frame.
