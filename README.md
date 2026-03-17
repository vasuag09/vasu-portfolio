# VASU_OS v4.1 — AI-Powered Engineering Portfolio

An interactive, terminal-enhanced, AI-augmented engineering portfolio built as a pseudo-operating system. Features an AI terminal (Gemini 2.5 Flash), deep-dive project pages, animated MLOps pipeline, retro CRT mode, guided tour, and GIF demos.

Showcases high-signal work across AI/ML, Deep Learning, GenAI, Computer Vision, and Full-Stack Engineering.

---

## Live Demo

https://vasuai.dev

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Framework | React 19, Vite 7 |
| Styling | Tailwind CSS 4, custom CSS animations |
| Animation | Framer Motion 12 |
| Icons | Lucide React |
| Routing | React Router v7 |
| AI | Gemini 2.5 Flash API |
| Testing | Vitest 4, Testing Library, jsdom |
| Deployment | Render |

---

## Repository Structure

```
vasu-portfolio/
├── public/                     # Favicon, static assets
├── src/
│   ├── App.jsx                 # Root component with React Router
│   ├── main.jsx                # Entry point with BrowserRouter
│   ├── index.css               # Custom animations, scrollbar, retro mode, a11y
│   ├── test-setup.js           # Vitest setup (jest-dom)
│   │
│   ├── data/                   # Static data (separated from components)
│   │   ├── blog-posts.js       # 5 engineering logs with slugs
│   │   ├── pipeline.js         # MLOps pipeline stages
│   │   ├── profile.js          # Bio, stats, career trajectory
│   │   ├── projects.js         # 12 projects, tier sorting, filtering
│   │   ├── skills.js           # 4 skill categories, 8 certifications
│   │   ├── tour.js             # Guided tour steps & tab mapping
│   │   └── __tests__/          # Data validation tests
│   │
│   ├── hooks/                  # Custom React hooks
│   │   ├── useBootSequence.js  # Boot animation with skip support
│   │   ├── useDocumentTitle.js # Dynamic page titles
│   │   ├── useKeyboardShortcuts.js
│   │   ├── useTerminal.js      # Terminal state & command handling
│   │   └── __tests__/          # Hook tests
│   │
│   ├── utils/                  # Pure utility functions
│   │   ├── markdown.jsx        # Markdown parser (bold, code, links, headings)
│   │   ├── terminal-commands.js # Command processor & Gemini AI integration
│   │   └── __tests__/          # Utility tests
│   │
│   ├── components/
│   │   ├── effects/            # Visual effects
│   │   │   ├── BootSequence.jsx
│   │   │   ├── GlowOrb.jsx
│   │   │   ├── ParticleField.jsx
│   │   │   └── RetroOverlay.jsx
│   │   │
│   │   ├── layout/             # App shell & navigation
│   │   │   ├── Header.jsx
│   │   │   ├── MobileNav.jsx
│   │   │   ├── NavButton.jsx
│   │   │   └── Sidebar.jsx
│   │   │
│   │   ├── sections/           # Page-level content sections
│   │   │   ├── About.jsx
│   │   │   ├── Blog.jsx
│   │   │   ├── Overview.jsx
│   │   │   ├── Pipeline.jsx
│   │   │   ├── ProjectCard.jsx
│   │   │   ├── Projects.jsx
│   │   │   └── Skills.jsx
│   │   │
│   │   ├── views/              # Full-page views (code-split)
│   │   │   ├── BlogReader.jsx
│   │   │   └── ProjectDeepDive.jsx
│   │   │
│   │   ├── modals/             # Overlay components
│   │   │   ├── GifPreview.jsx
│   │   │   ├── GuidedTour.jsx
│   │   │   └── Terminal.jsx
│   │   │
│   │   └── ui/                 # Reusable UI primitives
│   │       ├── ErrorBoundary.jsx
│   │       ├── SkipToContent.jsx
│   │       ├── StatusBadge.jsx
│   │       ├── TechTag.jsx
│   │       ├── TierBadge.jsx
│   │       └── __tests__/      # Component tests
│   │
│   └── assets/                 # Images
│
├── index.html                  # Meta tags (OG, theme-color)
├── package.json
├── vite.config.js              # Vite + Vitest configuration
├── tailwind.config.js
├── postcss.config.js
├── eslint.config.js
├── .prettierrc
└── .gitignore
```

---

## Routes

| Path | View |
|------|------|
| `/` | Overview (dashboard) |
| `/projects` | Project gallery with tier/tech filtering |
| `/projects/:alias` | Project deep-dive |
| `/skills` | Skills grid + certifications |
| `/blog` | Engineering logs list |
| `/blog/:slug` | Full blog reader |
| `/about` | About, education, contact |

---

## Core Features

### 1. OS-Style UI + Boot Sequence
- Animated kernel boot logs with "Skip" button
- Floating particle field background and glow orb accents
- Retro CRT overlay mode with scanline animation
- Keyboard shortcuts (Cmd+K terminal, 1-5 tabs, ESC close)

### 2. AI-Powered Terminal (Gemini API)
- Natural language questions about projects, skills, experience
- Built-in commands with portfolio data context injection

| Command | Action |
|---------|--------|
| `./help` | List available commands |
| `./whoami` | Display user context |
| `./list-projects` / `./ls` | List all 12 projects |
| `./open [alias/id]` | Open project deep-dive |
| `./stats [alias/id]` | View project quick stats |
| `./ship-log` | List engineering logs |
| `./clear` | Clear terminal |
| `./retro` | Toggle retro CRT mode |
| Any text | Ask VASU_OS AI |

### 3. Deep-Dive Project Pages
- Problem statement, architecture, pipeline, design decisions, failure modes, metrics
- GIF demonstrations, tech-stack tags, tier badges (S/A/B)
- 4 project statuses: LIVE, RESEARCH, CODE, BUILDING

### 4. Animated MLOps Pipeline
- 8 stages: Raw > Clean > Transform > Batch > Train > Eval > Save > Deploy
- Responsive layout (horizontal on desktop, vertical on mobile)

### 5. Engineering Blog
- 5 long-form technical write-ups with rendered markdown
- Tags, read time estimates, full-screen article mode
- URL-routable via slugs

### 6. Skills + Career Trajectory
- 4 skill categories with filterable project links
- 8 professional certifications
- Interactive career timeline (2021-2024)

### 7. Guided Tour
- 5-step overlay for first-time visitors
- Persisted via localStorage

### 8. Accessibility
- Skip-to-content link
- `aria-current="page"` on active nav
- `aria-label` on all icon links
- `role="dialog"` + `aria-modal` on modals
- `focus-visible` styling
- Minimum 11px font sizes

---

## Performance

- **Code splitting**: `React.lazy()` + `Suspense` for Terminal, ProjectDeepDive, BlogReader
- **Separate chunks**: Terminal (2.79KB), BlogReader (3.50KB), ProjectDeepDive (4.92KB)
- **Main bundle**: ~414KB (gzipped much smaller)

---

## Testing

```bash
npm test          # Watch mode
npm run test:run  # Single run
```

**11 test files, 79 tests** covering:
- Data integrity (profile, projects, skills, blog posts)
- Markdown parser (headings, bold, code, links, numbered lists)
- Terminal commands (all built-in commands, AI response mocking)
- Hooks (boot sequence timing/skip, document title lifecycle)
- UI components (TierBadge, StatusBadge, ErrorBoundary)

---

## Environment Setup

1. Clone and install
```bash
git clone https://github.com/vasuag09/vasu-portfolio.git
cd vasu-portfolio
npm install
```

2. Add your Gemini API key
```bash
echo "VITE_GEMINI_API_KEY=your_api_key_here" > .env
```

3. Run locally
```bash
npm run dev
```

4. Build for production
```bash
npm run build
```

### Deployment Notes
- Deployed on Render
- Uses `postcss.config.js` (CommonJS-compatible via Vite)
- Tailwind CSS 4 uses `@import "tailwindcss"` syntax (no `@tailwind` directives)

---

## Architecture Decisions

- **Monolith decomposition**: Original 2,012-line `App.jsx` split into 30+ focused files
- **Data separation**: All static data in `src/data/` for easy editing without touching components
- **Custom hooks**: Boot sequence, document title, terminal, keyboard shortcuts extracted as reusable hooks
- **URL routing**: React Router v7 with aliased project/blog routes instead of tab-based navigation
- **Error boundaries**: Class-based ErrorBoundary wrapping the entire app with a themed error page
- **AI context injection**: Terminal builds a system prompt from all portfolio data for accurate Gemini responses

---

## Contact

**Vasu Agrawal** — AI/ML Engineer, Full-Stack Developer

- Email: vasuagrawal1040@gmail.com
- GitHub: https://github.com/vasuag09
- LinkedIn: https://www.linkedin.com/in/vasu-agrawal20/

---

## License

MIT License.
