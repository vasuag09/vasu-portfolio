# Design Audit — v5 "Neural Core" vs the Award Bar

Date: 2026-06-11 · Method: full-journey screenshots at DPR2 (this folder),
deep-link rest-pose captures, rAF frame sampling during wheel scroll.
Verdict up front: **the engineering is award-grade; the composition is not.**
The site currently reads as "good text column over a particle wallpaper."

## P0 — Score-killers (fix before anything cosmetic)

### 1. Navigation lands on empty space
`rest-projects.png` / `rest-skills.png`: dot-click, number-key, and
`?section=` landings show **zero DOM content** — just particles. Content
sits at the vertical *center* of 240–280svh sections, but `scrollToSection`
lands at the section *top*, a full viewport+ above the text. A juror who
clicks a nav dot sees an empty screen. (Phase-10 "deep link lands on END
state" verified `top=0` — the right scroll position, the wrong reading
position.)

### 2. One layout, five times
Every chapter is the identical centered `max-w-3xl` column with the same
radial scrim, same label treatment, same reveal. This is the exact
"uniform spacing, no hierarchy, flat layout" pattern the design-quality
rules ban. Five chapters should not be five instances of one component
with different words in it.

### 3. The scene is one scene
The particle field is uniform confetti — same density, same dot size, same
green, everywhere. Chapters differ only by camera angle, so every
screenshot looks like every other screenshot. Award 3D sites give each
chapter a scene identity (density gradient toward a core, a structural
formation, a color temperature shift). Currently: dots, dots, dots.

## P1 — The differentiators jurors actually score

### 4. The "data-driven graph" story is invisible
`rest-projects.png`: overlapping translucent spheres of similar size, gray
satellites, no labels, no readable structure. The single best concept in
the spec — *a real skills→projects graph* — cannot be perceived without
opening dev tools. Nodes need: size hierarchy by tier, labels on approach
(billboarded, fading by camera distance), and visual binding to the DOM
list (hover a project button → leader line to its actual node).

### 5. Connection lines read as artifacts
Long straight streaks crossing the full frame (both rest poses) look like
rendering glitches, not network edges. They need: curve, depth fade,
distance-based opacity, and far fewer of them. (This was the carried
"connection-line art direction" item — it is now actively hurting.)

### 6. Typography has no display moment
IBM Plex Mono at modest sizes for *everything*. Mono-only can win awards,
but only with scale drama: chapter titles at hero scale (currently
`--text-xl` ≈ 2.75rem — caption-sized for a cinematic site), weight/width
contrast, char-level staggered reveals (SplitText was deferred — this is
where it pays), oversized section numerals, something confidently bright.
Body-muted-gray everywhere = no value hierarchy in the type, mirroring the
no-value-hierarchy in the scene.

### 7. Micro-interactions are one-note
One fade+18px translate per whole section block. No stagger between label
→ heading → body; no parallax between text and scene; buttons are
identical thin-border rectangles ("boot synapse" — the signature AI
feature — is the smallest element on the hero). Hover states exist but
none surprise.

### 8. Flagship visuals are buried
The Veo clips — the most expensive pixels on the site — appear only inside
the case-study panel, two clicks deep. The main journey never shows a
single frame of them. Surface a poster/inline teaser on the projects
chapter.

### 9. Boot/gate is generic
Black screen + three centered text lines + two small buttons. The scramble
is good; the composition around it is a default. The boot is the first 1.5s
of a site competing on first impressions — it should use the particle
language (converge particles INTO the name, then hand off to the hero).

## P2 — Performance headroom (the "feels heavy" report)

Measured during continuous wheel scroll, 1440×900 @ DPR2 — **dev server**
(production will be better; treat as relative signal):
avg 15.2ms (66fps), **p95 25ms (~40fps dips)**, 0 frames >33ms.

Levers, in order of cost/benefit:
- **8x MSAA on the composer target at DPR2** is the single biggest fill
  cost. 4x is visually indistinguishable at these dot sizes; or swap to
  SMAA pass. (8x was a Phase-2 jaggies fix — re-evaluate at 4x + DPR 1.75.)
- **Bokeh DoF runs full-res** every frame on desktop. Either kill it (DoF
  reads barely at all in current screenshots) or run it at half-res.
- **Desktop DPR cap 2 → 1.75**: ~23% fewer fragments, invisible on a
  particle field.
- Adaptive degradation during fast scroll (drop DPR while |velocity| high,
  restore at rest) — standard award-site trick, ~30 LOC.

## What is already award-grade (don't touch)

Scroll feel (Lenis+scrub), text-as-real-DOM architecture, reduced-motion
tier, a11y (axe 0), CWV gates, Synapse server hardening, Veo encode
pipeline, boot skip logic, the *concept* of the data-driven graph.

## Recommended execution order

1. **P0.1 landing fix** — land nav/deep-links at content, not section top
   (one-line offset change + verify all 3 input paths).
2. **P0.2/0.3 + P1.4/1.5 together** — this is one body of work: per-chapter
   scene identity + graph legibility + line art direction + chapter layout
   variation (editorial splits: numerals, asymmetric two-column, overlap).
3. **P1.6/1.7** — typographic scale system + staggered reveals + button/CTA
   hierarchy + SplitText hero.
4. **P1.8** — Veo poster strip on the projects chapter.
5. **P1.9** — boot particle convergence.
6. **P2** — MSAA 4x, DoF decision, DPR 1.75, adaptive scroll degradation.
