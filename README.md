# Vasu OS v4.1 — AI-Powered Engineering Portfolio

An interactive, terminal-enhanced, AI-augmented engineering portfolio showcasing high-signal AI/ML, Deep Learning, GenAI, Classical ML, and Full-Stack Engineering work. Built as a pseudo-operating system with an AI terminal, deep-dive project pages, animated MLOps pipelines, retro CRT mode, guided tour, and GIF demos.

---

## Live Demo
https://vasu-portfolio-m8p1.onrender.com/

---

## Repository Structure
```
/src
 ├─ Portfolio.jsx      # Main interactive portfolio OS
 ├─ assets/            # GIF demos, screenshots, icons
 ├─ components/        # Shared UI components
/public                # Favicon, metadata
README.md
package.json
tailwind.config.js
postcss.config.cjs
```

---

## Core Features

1. OS-Style UI + Boot Sequence
    - Animated kernel logs
    - System-status indicators
    - Retro CRT overlay mode
    - Keyboard shortcuts:
      - ⌘ + K → Open AI Terminal
      - 1–5 → Switch tabs
      - ESC → Close everything

2. AI-Powered Terminal (Gemini API)
    - Answer questions about projects
    - List projects, view stats, open deep dives
    - Maintain command history
    - Produce short, technical answers in a “hacker aesthetic”
    - Commands:
      - `./help`
      - `./whoami`
      - `./list-projects` or `./ls`
      - `./open [alias/id]`
      - `./stats [alias/id]`
      - `./ship-log`
      - `./clear`
      - `./retro`

3. Deep-Dive Project Pages
    - Problem statement, system architecture, engineering pipeline, design decisions, failure modes, technical metrics, links (live demo / GitHub)
    - GIF demonstrations, tech-stack filtering, 4-section analysis layout

4. Animated MLOps Pipeline
    - Stages: Raw → Clean → Transform → Batch → Train → Eval → Save → Deploy
    - Each stage animates independently, shows past/active state, and auto-loops

5. Engineering Logs
    - Long-form write-ups: LangChain optimization, TensorFlow models, system design, MATLAB CV pipelines, build failures & debugging
    - Each entry includes tags, read time, rendered markdown, split headings, full-screen article mode

6. Skills Visualization + Career Trajectory
    - Timeline progress tracker, grouped skills (GenAI, DL, FS, Data Eng), filter projects by skill

7. Guided Tour (First-Time Visitors)
    - 5-step overlay covering deployments, terminal, blog/logs, CV/contact, collaboration options

8. About Panel
    - Target roles, offered expertise, education, contact buttons, styled in OS/terminal theme

---

## Tech Stack

Frontend
- React + Vite
- TailwindCSS
- Framer Motion
- Lucide Icons

AI Integration
- Gemini 2.5 Flash API
- Custom system prompt
- Context injection from portfolio data

Other
- LocalStorage (guided tour)
- Keyboard listeners
- Component animation pipelines

---

## Environment Setup

1. Clone repo
```bash
git clone https://github.com/YOUR_USERNAME/portfolio-os.git
cd portfolio-os
```

2. Install dependencies
```bash
npm install
```

3. Add your Gemini API key
Create a `.env` file:
```
VITE_GEMINI_API_KEY=your_api_key_here
```

4. Run locally
```bash
npm run dev
```

Build
```bash
npm run build
```

Deployment
- Render recommended
- Ensure `postcss.config.cjs` (CommonJS) is used
- Tailwind must be in devDependencies

---

## Notable Engineering Decisions
- Terminal implemented with state history + auto-scroll
- AI calls include a custom `systemContext` built from skills, projects, and engineering logs
- Deep-dive view uses route-less single-page transitions
- Complex animations separated into small state machines
- Retro CRT mode uses layered gradients + scanline animation

---

## Shortcuts Reference

| Shortcut | Action |
|--------:|:-------|
| ⌘ + K  | Open AI Terminal |
| ESC    | Close overlays |
| 1–5    | Switch tabs |
| ALT/CTRL prevention | Avoids mode disruption |

---


## Contact

Portfolio Owner:  
Vasu Agrawal — AI/ML Engineer, Full-Stack Developer  
Email: vasuagrawal1040@gmail.com  
GitHub: https://github.com/vasuag09

---

## License
MIT License.
