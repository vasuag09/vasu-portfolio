# v4 Export Snapshot — 2026-06-11

Safety snapshot of VASU_OS v4 (Vite + Express SPA) taken during Phase 0 of the
v5 rebuild, per PLAN.md. v4 source stays live at the repo root until /ship
(ADR-5); this folder preserves content and visuals independently of the code.

## Contents

| Path | What it is |
| --- | --- |
| `content/` | All v4 data files (`src/data/*.js`): projects, skills, profile, blog posts, navigation, tour, pipeline, constants |
| `screenshots/` | Full-page captures of all six v4 routes at 1440px desktop + home at 375px mobile, from the live dev server (commit `7d00001`) |
| `index.html` | v4 HTML shell (meta tags, fonts, OG setup) |
| `v4-package.json` | v4 dependency manifest at snapshot time |

## Screenshots

- `v4-desktop-1440-home.png` — hero viewport (tour dismissed)
- `v4-desktop-1440-full.png` — home, full page
- `v4-desktop-1440-projects.png` · `-skills.png` · `-research.png` · `-about.png` · `-blog.png` — full-page per route
- `v4-mobile-375-home.png` — mobile home, full page

## Notes

- v4 routes: `/` (Overview), `/projects`, `/skills`, `/research`, `/about`, `/blog`
  — the neural-layer naming (INPUT/HIDDEN/OUTPUT/SIGNAL) is v4 brand DNA carried
  into v5's region concept.
- The v4 Gemini terminal (Synapse), boot sequence, and canvas shaders are ported
  directly from source in Phase 2/6 — they live in `src/components/`, not here.
