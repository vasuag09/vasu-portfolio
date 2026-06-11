# Plan: VASU_OS 5 "Neural Core"

> Produced 2026-06-11 by the planner agent from SPEC.md + RESEARCH.md + AWARD-RESEARCH.md
> + CONTEXT.md. 12 phases, risk-first ordering, walking skeleton early.
> Note: scroll authority was already decided in RESEARCH.md (native scroll + Lenis + GSAP);
> Phase 1 is the prototype that CONFIRMS it, /architect item 1 is a validation pass, not
> a re-decision.

## Objective
Build portfolio v5 (Next.js 16 / React 19 / R3F 9.6.1) targeting Awwwards SOTD quality,
repositioning Vasu as "AI Developer": 3D neural-network world, scroll-driven guided camera,
data-driven skills graph, 3–5 Veo-clip case studies, Synapse AI node, sound design, custom
cursor, chapter nav rail, ≤1.5s boot, reduced-motion + mobile tiers. Launch on vasuai.dev,
v4 archived to v4.vasuai.dev.

## Phases

### Phase 0 — Repo setup, design tokens, content data, v4 safety snapshot (S)
- Repo strategy: recommend new `/app` dir in same repo (Option A); document cutover plan.
- Next.js 16 + React 19 + TS scaffold; design-tokens (OKLCH palette; Berkeley Mono vs
  IBM Plex Mono decision); export v4 (screenshots + content bundle) to docs/v5/v4-export/.
- Create data files from CONTEXT.md: `projects-v5.ts` (flagships, "under review" Ray label),
  `skills-graph.ts` (skill→project edges), `sections.ts` (5 region anchors).
- Vasu decisions: Streamlit framing/swap, 5th flagship, CGPA display, sound default.
- **Exit:** scaffold `npm run build` green; tokens compile; v4 export exists; flagship
  decisions in writing.
- Parallel: content copy refresh, Veo budget planning.

### Phase 1 — Scroll-camera rig & navigation skeleton (M) ← WALKING SKELETON
- Lenis on native scroll; `useScrollCamera` (Lenis→GSAP ScrollTrigger→camera offset);
  drei MotionPathControls with 5 hand-authored CubicBezierCurve3 anchors; placeholder
  sections (real DOM); ChapterNav rail (dots + keyboard + `?section=` deep links);
  reduced-motion bypass.
- **Exit:** scroll flies camera through 5 chapters in order; dot click + keyboard + deep
  link work; reduced-motion = no flight; build green.

### Phase 2 — 3D scene foundation: ported network + GPU particles (L)
- Port v4 canvas components + shaders; data-driven node layout (build-time positions from
  data files); particle upgrade: curl-noise motion, per-particle lifetime, selective bloom,
  DoF; `useParticleConfig` tiers (desktop 60–100k / low 5–10k); Safari/Firefox check.
- **Exit:** curl-noise particles at 60fps desktop; reduced-motion = static; text stays
  real DOM; renders on Safari + Firefox.

### Phase 3 — Data-driven skills graph & node interactivity (M)
- NetworkGraph from data files; skill↔project edge highlighting (uActive uniform);
  HTML overlay skill list; click-to-expand project panel; keyboard navigation.
- **Exit:** hover skill → connected projects glow; click opens panel; Tab/arrows/Enter work.

### Phase 4 — Veo pipeline & Fast-tier exploration (M) ‖ parallel with 2–3
- Prompts for FundlyMart / NM-GPT / GeoVision moments; generate Fast (~20cr) explorations;
  lock creative direction; clip spec doc (AV1/H.265 + H.264 fallback, ≤4MB); `<VeoClip>`
  lazy wrapper with image fallback.
- **Exit:** 3 Fast clips reviewed; direction locked; ~60/567 credits spent; ≥400 + 100
  buffer remain.

### Phase 5 — Case-study template & content integration (L)
- CaseStudyPanel per winner anatomy (hero → problem/role pinned → Veo payload → outcome
  metrics → next-project funnel); SplitText reveals (trigger once); camera rest-pose
  alignment; flagship copy in AI-Developer voice.
- **Exit:** node click → panel; reveals fire once; no CLS; copy metric-driven.

### Phase 6 — Synapse 3D integration & reduced-motion tier (M)
- SynapseNode in scene → terminal UI; port api/ai.js → `/app/api/synapse/route.ts`
  (rate limit, env key); full reduced-motion tier (static composition, frozen particles,
  fade-only reveals); GPU-tier detection.
- **Exit:** Gemini chat works end-to-end in 3D; reduced-motion fully static; no key in client.

### Phase 7 — Sound layer, OG metadata, accessibility polish (M)
- Boot jingle + ambient loop + 3–5 SFX (licensed); SoundGate (enter with/without) +
  persistent toggle + iOS unlock; per-route OG images/tags; WCAG AA contrast audit;
  focus-visible states; full keyboard journey.
- **Exit:** sound choice persists; LinkedIn share renders rich card; AA verified; visible
  focus everywhere.

### Phase 8 — Mobile tier & performance budget (M)
- Lighthouse mobile profiling; dynamic-import Canvas (`ssr:false`); A/B mobile strategy:
  reduced live scene vs pre-rendered scene video (Vitasović pattern) — /architect decides
  criteria, this phase measures; document MOBILE-STRATEGY.md.
- **Exit:** PageSpeed mobile LCP <2.5s / INP <200ms / CLS <0.1; JS ≤300kb gz total;
  touch interactions work.

### Phase 9 — Quality-tier Veo generation & finalization (S, after 5)
- Quality (~100cr) finals from Fast learnings; compress; integrate; 4G throttle test.
- **Exit:** clips <4MB, no buffering on 4G; credits ≤400 spent.

### Phase 10 — Boot sequence, custom cursor, interaction polish (M)
- Boot ritual ≤1.5s (scramble → particle burst → name reveal), skippable, localStorage
  skip for return visits, jingle hook; CustomCursor (morph states, Safari fallback);
  chapter nav polish (number keys 1–5, damped jumps, deep links land on END state).
- **Exit:** boot ≤1.5s + skip + return-skip; cursor morphs; all nav input methods smooth.

### Phase 11 — Self-score gate & Awwwards prep (S) — HARD GATE
- Score vs rubric: Design ≥8.0 / Usability ≥8.0 / Creativity ≥8.5 / Content ≥8.0.
- Score-killer checklist (any = fail): broken mobile; LCP>3s/INP>300ms/CLS>0.1; AA contrast
  fail; scroll-jacking; cross-browser failure; dead links. Re-check Ray PR merge status.
- Submission pack: screenshots, tagline, text, fees, staggered Awwwards → CSSDA → FWA.
- **Exit:** scores ≥ targets; zero killers; SELF-SCORE.md + LAUNCH-CHECKLIST.md signed off.

### Phase 12 — E2E tests, cutover, /ship (M)
- Playwright suite (boot, journey, nav, case studies, Synapse, reduced-motion, mobile, OG);
  deploy v5 → vasuai.dev, v4 → v4.vasuai.dev; 48h error monitoring.
- **Exit:** E2E green in CI; production live; zero launch errors; OG card verified.

## Critical path
0 → 1 → 2 → 3 → 5 → 9 → 10 → 11 → 12, with 4 ‖ (2–3), 6/7/8 largely ‖ after 5.

## Risks (top 5, ordered)
1. Scroll-authority feel (P1 prototype proves before everything builds on it).
2. Mobile tier strategy (live-vs-video — measured in P8, decided via /architect criteria).
3. Veo budget burn (Fast-first exploration, 100cr reserve, hard stop at 400).
4. Performance budget vs 3D library weight (dynamic import; measured in P8).
5. Particle quality bar (curl noise + selective bloom is new shader work — P2 is L-sized
   for this reason; Codrops references in AWARD-RESEARCH §5).

## Where /architect is needed (before /implement)
1. Scroll-authority validation criteria (what makes the P1 prototype "pass").
2. Mobile tier decision framework (live scene vs pre-rendered video).
3. Scene-graph data model (build-time hand-authored positions vs force-directed; camera
   rest-pose schema).
4. DOM↔canvas integration (fullscreen canvas + DOM overlay vs container; SEO/a11y/CLS).
5. v4→v5 cutover strategy (recommend Option A: new /app dir, same repo).
6. Reduced-motion layout model (static tiles vs tall column).

## Open questions — RESOLVED 2026-06-11
1. ~~Streamlit flagship~~ → DECIDED: Insightify becomes flagship #4; Streamlit demoted to
   Open Source section with honest framing (see CONTEXT.md). Flagships: FundlyMart,
   NM-GPT, Ray Serve PR ("under review"), Insightify, GeoVision-LULC.
2. Veo clip shape → DEFAULT: one self-contained ~8s story clip per flagship (3 clips
   quality-tier ≈ 300cr fits budget); hero+artifact pairs would double spend. Vasu may
   override before Phase 9.
3. ~~Typography~~ → DECIDED: IBM Plex Mono (free, 8 weights) as the brand mono.
4. Sound default → DEFAULT: explicit two-button gate, "Enter with sound" styled as the
   primary/encouraged action, no autoplay (browsers block it anyway); choice persisted.
5. Blog markdown: keep in repo unrouted at launch (per SPEC out-of-scope).
