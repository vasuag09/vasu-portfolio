# Award-Level Website Research — Synthesis for VASU_OS 5 "Neural Core"

> Compiled 2026-06-11 from four parallel research tracks: (1) award-winning 3D portfolios,
> (2) judging criteria, (3) scroll-narrative anatomy, (4) AI/particle/terminal visual language.
> Figures marked ~ are inferred from case studies, not official jury data.

---

## 1. How winners are judged

**Awwwards** scores 1–10 across **Design / Usability / Creativity / Content** (jury of
international designers/developers + community vote). Observed thresholds: Honorable
Mention ~6.5–7.5, **Site of the Day ~7.5–8.5**, Developer Award judged separately on code
quality, cross-browser, accessibility, mobile optimization ("the real beauty lies hidden
deep within"). **CSSDA** scores UI / UX / Innovation (recent WOTD example: 8.35/8.18/8.35).
**FWA** is more experimental-tolerant, less strict on a11y/perf.

**Score-killers (instant downgrade):** broken mobile, LCP >3s / INP >300ms, WCAG AA
failures, scroll-jacking, generic template look, hollow content, cross-browser failures.

**Self-evaluation targets for v5:** Design ≥8.0, Usability ≥8.0, Creativity ≥8.5, Content ≥8.0.

## 2. The single trait all winners share

**The portfolio itself is a finished product — not a container for finished products.**
Winners are *ownable*: a metaphor only that person could authentically ship (Bruno Simon =
drive a car; Rauno Freiberg = an OS because he builds UX systems; Pacôme Pertant = sound
because he's a sound designer). For Vasu: **the neural network is ownable because his work
IS production AI** — and the graph must be real data (skills wired to actual projects), not
decoration. Judges increasingly reward "how I think" over "what I made".

## 3. Structural patterns shared by 80%+ of winners

1. **Loader/boot ritual** (<2s, skippable; pairs with audio-context unlock) — our VASU_OS
   boot = native fit, but cap at ~1.5s with instant skip.
2. **Sound design with explicit choice** — "Enter with/without sound" gate or toggle;
   never forced. (Strong differentiator; most dev portfolios skip it.)
3. **Custom cursor** (context-aware, morphs on interactive elements) — ~90% of winners.
4. **Page transitions, never reloads** — animated bridges between routes.
5. **Projects as narrative case studies, not cards** — hero → problem → role → visuals
   (~80% visual / 20% text) → outcome → next-project funnel.
6. **Footer as designed element** — big-type "let's work together" CTA.
7. **About as editorial POV**, not a résumé.
8. **GSAP ScrollTrigger** in ~95% of winners; animate transform/opacity only.
9. **Dark mode as intentional choice** (~70% of winners); palette = base + 1 accent
   (~80% use ≤3 colors).
10. **Navigation redundancy** — header + footer + overlay menu; chapter dots/progress
    rail for scroll journeys; dual CTA (case study for designers, resume/GitHub for hiring).
11. **Award badges as social proof** once earned.
12. **Dedicated mobile experience** — reduced 3D tiers or video fallback (Vitasović ships
    HTML5 video on mobile instead of WebGL).

## 4. Scroll-journey motion grammar (numbers to build with)

- **Chapter length:** ~10–20 viewport-heights per major chapter; 4–6 chapters total.
  Project case-study region anatomy: hero 200–300svh → problem/role text pinned
  300–400svh → visual payload 400–500svh → CTA/transition ~100svh.
- **Text pinning:** pin copy for ~25–40% of a section's scroll while the 3D animates;
  exit via opacity+translateY over the final 20–30%. Stagger reveals (SplitText-style);
  trigger once, never re-animate on re-scroll (NN/g finding: repeat animations = "slow").
- **Scrub/damping:** `scrub: 1` standard (0.5 snappy, 2 meditative); Lenis lerp ~0.08–0.12;
  easing `power1.inOut`/`expo.inOut` for camera, never linear for flights.
- **Navigation:** chapter dots/progress rail + overlay menu + keyboard (arrows, Home/End)
  + deep links (`?section=projects` reconstructs scroll position, jumps to END state of
  animations, never replays).
- **Anti-motion-sickness:** ~35% of adults have some vestibular sensitivity. Mitigate:
  damped scrub, camera pans <45°/0.5s, visual rest moments, `prefers-reduced-motion` →
  no camera flight + static composition, visible skip control.
- **Scroll authority:** winners use NATIVE scroll + smoothing (Lenis/ScrollSmoother),
  never hijacked wheel events — preserves momentum, pinch-zoom, keyboard. (Confirms
  RESEARCH.md architecture choice (b).)

## 5. Visual language: what reads premium in 2026

**AI-cliché blacklist** (signals "AI-generated template" to judges): purple gradient hero,
four-pointed sparkles, glowing orbs, soft blur everywhere, Inter/Roboto, centered hero +
3 cards, Matrix falling text, neon everywhere.

**What the strongest AI brands do instead** (Cohere, Sierra, Faculty, Cresta): identity
through custom/characterful TYPE, geometric systems that literally represent the tech,
restraint = trust. Direction statement for Neural Core:

> "Credible neural interface: phosphor-terminal heritage meets scientific particle
> systems. Don't visualize 'intelligence' — design trust through restraint: high contrast,
> characterful mono type, physics-driven motion, selective glow."

**Particle quality bar** (what separates Lusion-grade from particles.js):
- 60–100k particles, GPU-driven (instancing/GPGPU), curl-noise/physics motion (never
  linear drift), per-particle lifetime, interaction memory (responds to cursor/scroll
  persistently), **selective** bloom (accent hue only) + depth-of-field bokeh on far
  field, data encoded in textures not JSON. Sparse intentional density > maxed count.
- Reference implementations: Codrops "Living Particle System" (UntilLabs, 60k particles
  + curl noise + FBO), Phantom.land 3D face particles, Codrops "Cinematic 3D Scroll
  Experiences with GSAP" (2025-11-19), three.js journey particles-morphing-shader.

**Terminal aesthetic done premium** (Poolsuite, teenage engineering, Heffernan): coherent
system not gimmick; monospace with character; one accent; skeuomorphic detail sparingly;
animation restraint (cursor blink yes, falling glyphs no).

**Palette (OKLCH discipline):**
```css
--bg-base:     oklch(8% 0.02 250);   /* deep space navy-black, not pure #000 */
--bg-elevated: oklch(14% 0.01 250);
--accent:      oklch(58% 0.18 150);  /* phosphor/signal green — THE accent */
--text:        oklch(92% 0.02 250);
--text-muted:  oklch(68% 0.01 250);
```
Lightness = elevation in dark UI; chroma discipline (0.01–0.03 on neutrals); one accent
used sparingly hits harder than five used everywhere (Vercel/Linear principle).

**Typography candidates:** Berkeley Mono ($75, the premium pick — terminal at 13px,
magazine cover at 96px) or IBM Plex Mono (free, 8 weights, engineering character) as the
brand mono; optionally paired with a characterful display face for editorial headlines.
Never Inter/Roboto as the identity face.

## 6. Named reference sites (study before /architect)

| Site | Steal this |
|---|---|
| Igloo Inc (SOTY 2024) | WebGL-rendered UI text in heavy scenes; pinned project carousel; procedural intro |
| Lusion.co | Particle/motion quality bar; playful physicality |
| Bruno Simon | Joy + ownable metaphor; mobile presets; achievement/easter-egg layer |
| Henry Heffernan | OS-aesthetic executed premium; embedded interactive payoff |
| Stefan Vitasović (SOTD 2025) | Swiss restraint + shader accents; mobile = video fallback; char-assembly text |
| Rauno Freiberg | Micro-interaction polish; sound feedback; "satisfying to click" |
| Pacôme Pertant (SOTD 2026) | Sound entry choice; restraint; 7.76 scored recently — the bar |
| Apple AirPods Pro pages | Scroll-scrub pacing; text/visual alternation rhythm |
| Basement Studio | Physics-feel scroll (inertia on 3D planes) |

## 7. What this changes for the v5 plan (deltas to fold into /plan)

1. **Add a sound layer** to scope: ambient + interaction SFX, "enter with/without sound"
   choice in the boot gate, persistent toggle. (Was not in SPEC — high-leverage gap.)
2. **Custom cursor** + designed hover/active states everywhere — table stakes.
3. **Chapter navigation rail** (dots + labels + keyboard + deep links) — first-class
   component, not an afterthought; doubles as the recruiter skip-path (AC #3).
4. **Case-study template** follows winner anatomy: hero → problem → role → visual payload
   (Veo clip here) → outcome metrics → next-project funnel.
5. **Boot ritual ≤1.5s** with instant skip; repeat visitors skip automatically.
6. **Text reveals trigger once**; pin durations per §4 numbers.
7. **Typography decision** becomes a Phase-0 design-token task (Berkeley Mono vs IBM Plex
   Mono); palette locked to OKLCH tokens above (evolves v4 green, per SPEC open Q4).
8. **Particle system must meet the §5 quality bar** — curl noise + selective bloom + DoF,
   not floating dots; v4's fresnel node shader survives, motion system upgraded.
9. **Mobile tier = possibly pre-rendered video of the scene** (Vitasović pattern) rather
   than a degraded live scene — decide in /architect.
10. **Self-score gate before /ship:** Design ≥8 / Usability ≥8 / Creativity ≥8.5 /
    Content ≥8 + zero score-killers checklist (mobile, CWV, AA contrast, no scroll-jack).
