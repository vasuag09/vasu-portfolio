# Veo Brief — Phase 4 Fast-Tier Exploration

> Copy-paste-ready Flow session. Budget law (PLAN.md): Fast tier only in this
> phase, **≤60 credits total**, hard project ceiling 400, 100 reserve untouched.
> Quality-tier finals happen in Phase 9 ONLY after direction is locked here.

## The one rule of visual direction

Every clip must look like **credible lab footage from inside the Neural Core**
— scientific, restrained, dark — never a sci-fi trailer. The site palette is
deep navy-black `#05070d` with ONE phosphor-green accent (`#27b173` → bright
`#4fe3a1`). A clip that fights the palette gets rejected no matter how pretty.

**Global negative guidance (append to every prompt):**

> No text, no letters, no numbers, no logos, no user interfaces, no humans,
> no faces, no hands, no purple or magenta, no rainbow gradients, no lens
> flares, no bright daylight. Dark scene on near-black background.

Why: Veo renders text/UI as garbage glyphs; faces/hands artifact; purple
gradients are the #1 "AI template" tell (AWARD-RESEARCH §5 blacklist).

## Session protocol

1. Flow → new project per flagship, **Fast tier, 16:9, 8 seconds**.
2. Generate variant A and B for one flagship. Judge against the checklist
   below BEFORE generating more. Max 2 extra iterations per flagship.
3. Stop the session at 60 credits spent, even mid-flagship.
4. Download everything (Fast clips are reference material for Phase 9
   prompts, even rejects).

## Prompts

### 1. FundlyMart — "chaos becomes order"

Story beat: fragmented WhatsApp order chaos → one structured conversational
flow. Abstract metaphor only — no phones, no chat bubbles.

**Variant A (macro, particles):**
> Macro photography of thousands of tiny luminous green particles drifting
> chaotically in dark space, slowly converging into a single elegant ordered
> stream that flows left to right like a supply line, particles snapping into
> a precise moving lattice of rows, deep blacks, single green phosphor glow,
> shallow depth of field, slow cinematic dolly forward, scientific
> visualization aesthetic, 8 seconds, seamless loop.

**Variant B (architectural):**
> Dark warehouse of floating translucent glass cubes scattered in disarray in
> black void, cubes begin gliding into perfect orderly rows on an invisible
> conveyor path, each cube ignites with a soft green phosphor edge-light as it
> docks into formation, camera slow lateral track, volumetric haze, high
> contrast, minimal scientific aesthetic, 8 seconds, loopable.

### 2. NM-GPT — "a thousand pages, one answer"

Story beat: scattered institutional documents → grounded single answer with
citations. Metaphor: light extracted from paper.

**Variant A (dissolve and converge):**
> Hundreds of white paper sheets floating suspended in dark space slowly
> dissolving edge-first into fine glowing dust, the dust streams spiral
> gently into a single small intensely bright green core of light at center
> frame, the core pulses once and emits one thin clean horizontal beam,
> pitch black background, macro depth of field, slow orbital camera,
> scientific elegance, 8 seconds, seamless loop.

**Variant B (library of light):**
> Endless dark library aisle of faintly glowing translucent panels receding
> into blackness, a soft green wave of light travels along the aisle touching
> each panel which briefly brightens, the wave converges to a single floating
> point of green light in the foreground, slow push-in camera, fog, deep
> shadows, minimal and precise, 8 seconds, loopable.

### 3. GeoVision — "the land, classified"

Story beat: raw satellite imagery → clean semantic segmentation. This one can
be literal — terrain from above IS the product.

**Variant A (classification sweep):**
> Top-down aerial view of dark moody terrain at night, rivers fields and
> urban blocks barely visible in deep blue-black tones, a thin glowing green
> scanline sweeps slowly across the landscape, behind the scanline land
> regions illuminate as clean flat translucent zones in muted tones with
> green edge highlights, like a living map being classified, slow steady
> drift, satellite photography aesthetic, 8 seconds, seamless loop.

**Variant B (terrain lattice):**
> Slow flight over a dark stylized landscape that transitions from
> photorealistic terrain into a precise glowing wireframe topography, contour
> lines and field boundaries tracing themselves in phosphor green light over
> black, subtle data-grid shimmer, camera gliding forward smoothly,
> cartographic minimalism, 8 seconds, loopable.

## Review checklist (judge every clip on this, in order)

1. **Palette obedience** — sits on near-black, green is the only accent.
2. **Loop seam** — last frame ≈ first frame, or seam hideable with a 0.3s
   crossfade. (Prompts say "seamless loop"; Veo often ignores it — check.)
3. **Reads at half size** — the case-study slot is ~700px wide; squint test.
4. **No artifacts** — text-garbage, melting geometry, flicker.
5. **Mood** — lab footage, not trailer. If it feels "epic", reject.

Score each A/B; the winner per flagship + one sentence why = the locked
direction. Record in the table below and Phase 4 is exit-complete.

## Direction lock (fill in to close Phase 4)

| Flagship | Winner | Why | Credits spent |
| --- | --- | --- | --- |
| FundlyMart | A / B / retry | | |
| NM-GPT | A / B / retry | | |
| GeoVision | A / B / retry | | |

Total spent: ___ / 60 max · Remaining balance: ___ (must be ≥ 500)

## Clip delivery spec (Phase 9 finals, but Fast clips follow it too)

Budget per clip: **≤4MB**, 8s, 1080p, no audio track (site sound is separate).
Encode three sources + one poster from the Flow download (`in.mp4`):

```bash
# AV1 — best compression, Chrome/Firefox/Edge (+ Safari on M3+)
ffmpeg -i in.mp4 -c:v libsvtav1 -crf 38 -preset 6 -pix_fmt yuv420p10le -an clip.av1.mp4

# HEVC — Safari fallback (hvc1 tag is REQUIRED for Safari to play it)
ffmpeg -i in.mp4 -c:v libx265 -crf 26 -tag:v hvc1 -pix_fmt yuv420p -an clip.hevc.mp4

# H.264 — universal fallback
ffmpeg -i in.mp4 -c:v libx264 -crf 21 -profile:v high -pix_fmt yuv420p -movflags +faststart -an clip.h264.mp4

# Poster — first frame as AVIF (the reduced-motion/lazy fallback image)
ffmpeg -i in.mp4 -frames:v 1 -q:v 50 clip-poster.avif
```

Check sizes (`ls -la`); if any source exceeds 4MB, raise its CRF by 2 and
re-encode. Files live in `app-v5/public/veo/` named
`{project}.{codec}.mp4` + `{project}-poster.avif` and render through the
`<VeoClip>` component (`src/components/media/VeoClip.tsx`), which handles
source ordering, lazy loading, and the reduced-motion poster fallback.
