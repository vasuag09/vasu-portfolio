# Spec: VASU_OS 5 — "Neural Core" (portfolio v5, full revamp)

## Problem

The current portfolio (v4, "VASU_OS" terminal aesthetic) proved the concept — thousands of
LinkedIn viewers — but it presents Vasu's *past* identity (ML Engineer) and a 2D interface.
v5 must be a complete revamp: a 3D, showpiece-quality site that repositions Vasu as an
**AI Developer**, keeps the VASU_OS brand DNA the audience already loves, and refreshes all
content (bio, projects, skills) to current reality.

**Concept:** a living 3D neural network ("the Neural Core") is the world; VASU_OS terminal
fiction is the interface layer (boot = "initializing neural core"); scroll drives a guided
camera flight between network regions (sections). Google Flow/Veo clips (567-credit budget)
provide cinematic moments inside project case studies — garnish, not foundation.

**Audience:** recruiters & hiring managers first; showpiece/Awwwards quality is the
differentiator, never at the cost of the recruiter's 90-second path.

## In scope

- Fresh v5 build: **Next.js (App Router) on Vercel**, React 19 + react-three-fiber + drei +
  postprocessing + GSAP ScrollTrigger + Tailwind 4.
- Persistent 3D neural-network scene (GPU-instanced particles + synaptic connections,
  cursor-reactive), camera on a scroll-driven spline between section "regions".
- Sections as network regions: Hero/identity, Projects (nodes → case-study panels),
  Skills (a real graph — skill nodes literally linked to the projects that use them),
  About, Contact.
- **Synapse AI terminal** ported and re-skinned as a node you talk to inside the network.
- **Project deep-dives** ported; presentation fully redesigned; 3–5 Veo-generated cinematic
  clips embedded (≤ ~400 Flow credits incl. retries; keep ≥100 in reserve).
- **Guided tour** reborn as the guided camera path (the default scroll journey).
- **Content refresh:** all copy rewritten for the AI Developer identity; project list,
  skills, and bio updated. Content inventory is a Phase-0 task.
- Accessibility & fallback tier: reduced-motion mode (static composition, no camera
  flight), low-GPU/mobile tier (reduced particle count), "skip to content" recruiter path.
- Per-route OG metadata so LinkedIn shares render rich cards.

## Out of scope

- Blog / engineering logs (not selected for v5 launch; routes may 301 to home or be ported
  later — decide at /ship).
- Building the whole site as scroll-scrubbed video (Concept 3); free-roam museum navigation.
- Real-time ML demos in-browser beyond what Synapse already does (tfjs carryover optional,
  not required).
- CMS — content stays in typed data files as in v4.

## Acceptance criteria

- [ ] Given a first visit on desktop, when the page loads, then a VASU_OS-flavored boot
      sequence resolves into the 3D Neural Core with name + AI Developer positioning,
      LCP < 2.5s, and the scene holds ~60fps on a mid-tier laptop GPU.
- [ ] Given a visitor scrolls, when moving through the journey, then the camera flies
      between all five regions in a fixed order and every section's text remains real DOM
      (selectable, indexable), not canvas-rendered.
- [ ] Given a recruiter in a hurry, when they use the visible "skip" affordance or nav,
      then they reach Projects/Contact in ≤ 2 interactions without riding the full journey.
- [ ] Given `prefers-reduced-motion`, when the site loads, then no camera flight or particle
      churn occurs and all content is fully reachable.
- [ ] Given a mobile/low-GPU device, when the scene initializes, then a reduced tier renders
      (fewer particles, simplified post-processing) with no crash and CWV targets still met
      (LCP < 2.5s, INP < 200ms, CLS < 0.1).
- [ ] Given a project node is opened, when its deep-dive renders, then it shows refreshed
      content and (for 3–5 flagship projects) an embedded Veo clip ≤ 4MB stream-optimized.
- [ ] Given Synapse is opened inside the 3D world, when the user chats, then the existing
      Gemini-backed functionality works end-to-end (API on Vercel functions).
- [ ] Given any route is shared on LinkedIn, when the card unfurls, then a per-route OG
      image/title/description renders.
- [ ] Given the skills region, when a skill node is focused, then its connected project
      nodes highlight (the graph is data-driven from the projects data file).
- [ ] No hardcoded secrets; Gemini key stays server-side; CSP and security headers configured.

## Award-level requirements (see docs/v5/AWARD-RESEARCH.md)

Sound layer (enter with/without sound + toggle), custom cursor + designed interaction
states, chapter navigation rail (dots/keyboard/deep links — doubles as recruiter skip
path), winner-anatomy case-study template, boot ritual ≤1.5s skippable, once-only text
reveals, OKLCH palette + characterful mono typography (Berkeley Mono vs IBM Plex Mono),
particle quality bar (curl noise, selective bloom, DoF — never floating dots).
Pre-ship self-score gate: Design ≥8 / Usability ≥8 / Creativity ≥8.5 / Content ≥8,
zero score-killers (mobile, CWV, AA contrast, no scroll-jacking).

## Build-vs-reuse (see docs/v5/RESEARCH.md)

Port v4's R3F neural-net components + GLSL shaders and the `api/ai.js` Gemini proxy.
Adopt drei ScrollControls/MotionPathControls, GSAP 3.15 (now free incl. all plugins) +
useGSAP, lenis, Next.js 16. Single scroll authority: native page scroll + Lenis + GSAP
drives the camera offset (keeps text as real DOM). Build new: data-driven graph layout,
scroll-camera spline, tiered GPU particles, Veo clip pipeline.

## Constraints

- Performance: CWV targets above; JS budget ≤ ~300kb gzipped initial (3D libs
  dynamically imported after first paint); video lazy-loaded.
- Flow/Veo budget: 567 credits total — plan generations before spending; Fast (~20 cr)
  for exploration, Quality (~100 cr) only for finals.
- Brand: must read as an evolution of VASU_OS (terminal DNA, boot ritual), not a reboot.
- v4 stays live until v5 ships (build in new app directory / branch; cut over at /ship).

## Open questions (with recommended defaults)

1. Domain/hosting cutover — default: same domain (vasuai.dev) on Vercel, v4 archived on a
   subdomain (`v4.vasuai.dev`) as a museum piece.
2. Blog content fate — default: keep markdown files in repo, no routes at launch, revisit
   post-launch.
3. ~~Which 3–5 projects are "flagship"~~ — RESOLVED (see docs/v5/CONTEXT.md):
   #1 FundlyMart WhatsApp Bot, #2 NM-GPT/CollegeGPT, #3 Ray Serve PR (frame as
   "under review", not merged), #4 Insightify (DECIDED 2026-06-11 — Streamlit demoted
   to Open Source section with honest framing), #5 GeoVision-LULC.
   Veo clips prioritized for FundlyMart, NM-GPT, GeoVision (most cinematic subjects).
   Typography DECIDED: IBM Plex Mono.
4. Color identity — default: evolve the green-phosphor palette toward a broader
   "bioluminescent" range (deep space blues/violets + signal green as the accent) rather
   than abandoning it.
