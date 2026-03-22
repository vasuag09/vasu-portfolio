# Neural Voyage — Full Portfolio Redesign

## Vision

The portfolio IS a living neural network. Visitors don't browse a website — they traverse
a deep neural network, moving through layers, activating nodes, and watching data propagate.
Every interaction feels like a forward pass through the network.

The design mixes:
- **Neural Voyage** — core navigation and identity
- **Layers (depth parallax)** — Z-axis immersion, floating elements at different depths
- **Blueprint** — technical annotations on project deep-dives
- **Signal** — waveform accents and transitions between sections

---

## Design System

### Color Palette

| Token              | Hex       | Usage                                    |
|--------------------|-----------|------------------------------------------|
| `--bg-void`        | `#06080f` | Deepest background (the void)            |
| `--bg-deep`        | `#0c1020` | Section backgrounds                      |
| `--bg-surface`     | `#111827` | Cards, panels                            |
| `--bg-elevated`    | `#1a2035` | Hover states, active panels              |
| `--accent-cyan`    | `#00d4ff` | Primary accent (neural connections)      |
| `--accent-purple`  | `#8b5cf6` | Secondary accent (active nodes)          |
| `--accent-emerald` | `#10b981` | Success states, live indicators          |
| `--accent-amber`   | `#f59e0b` | Warnings, S-tier glow                    |
| `--text-primary`   | `#f1f5f9` | Headings, important text                 |
| `--text-secondary` | `#94a3b8` | Body text, descriptions                  |
| `--text-muted`     | `#475569` | Labels, annotations                      |
| `--gradient-neural`| cyan→purple | Neural connections, active paths        |
| `--glow-cyan`      | `rgba(0,212,255,0.15)` | Glow effects, halos           |
| `--glow-purple`    | `rgba(139,92,246,0.15)` | Node activation glow          |

### Typography

| Role     | Font            | Weight    | Usage                              |
|----------|-----------------|-----------|-------------------------------------|
| Display  | Space Grotesk   | 700       | Hero title, section headings        |
| Body     | Inter           | 400, 500  | Paragraphs, descriptions            |
| Mono     | JetBrains Mono  | 400       | Technical labels, annotations, code |

### Spacing & Layout

- Max content width: 1280px
- Section padding: 80px vertical
- Card gap: 24px
- Border radius: 12px (cards), 8px (buttons), 999px (pills)
- Consistent 8px grid

### Glassmorphism Cards

```css
background: rgba(17, 24, 39, 0.6);
backdrop-filter: blur(12px);
border: 1px solid rgba(0, 212, 255, 0.08);
box-shadow: 0 0 30px rgba(0, 212, 255, 0.03);
```

On hover:
```css
border-color: rgba(0, 212, 255, 0.2);
box-shadow: 0 0 40px rgba(0, 212, 255, 0.08);
transform: translateY(-2px);
```

---

## Component Architecture

### Background Layer: Neural Network Canvas

A persistent, interactive canvas that renders across all pages:

```
┌─────────────────────────────────────────────────┐
│  ○───○───○    ○───○───○    ○───○    ○───○───○   │
│  │ ╲ │ ╱ │    │ ╲ │ ╱ │    │ ╲ │    │ ╲ │ ╱ │   │
│  ○───○───○    ○───○───○    ○───○    ○───○───○   │
│  │ ╱ │ ╲ │    │ ╱ │ ╲ │    │ ╱ │    │ ╱ │ ╲ │   │
│  ○───○───○    ○───○───○    ○───○    ○───○───○   │
│                                                   │
│  INPUT     HIDDEN 1    HIDDEN 2    OUTPUT         │
│  (About)   (Projects)  (Skills)   (Contact)       │
└─────────────────────────────────────────────────┘
```

**Behavior:**
- Nodes gently pulse (breathing animation)
- Connections have flowing gradient particles (data flow)
- The layer you're currently viewing is BRIGHTLY LIT
- Other layers are dimmed but visible
- Hovering a layer in the nav brightens it with a preview
- Clicking triggers a "forward pass" animation: data particles flow from current layer → target layer
- Canvas opacity: ~25% so content is readable on top
- On mobile: simplified to a horizontal strip at top

**Node Types:**
- Input nodes: Small, cyan, steady glow
- Hidden nodes: Medium, gradient cyan→purple, pulse on interaction
- Output nodes: Larger, purple, strong glow
- Active node: Bright ring + particle emission
- Data particles: Tiny dots that flow along connections at varying speeds

### Navigation: Network Layer Selector

Not a traditional sidebar — a **layer indicator** docked to the left (desktop) or top (mobile):

```
Desktop:                          Mobile:
┌──────┐                         ┌─────────────────────────┐
│  ◉ I │ ← Input (Overview)     │ ◉ I ─ ◉ H1 ─ ◉ H2 ─ ◉ O │
│  │   │                         └─────────────────────────┘
│  ◉ H1│ ← Hidden 1 (Projects)
│  │   │
│  ◉ H2│ ← Hidden 2 (Skills)
│  │   │
│  ◉ H3│ ← Hidden 3 (Blog)
│  │   │
│  ◉ O │ ← Output (About)
└──────┘
```

**Behavior:**
- Vertical chain of connected nodes
- Active section: bright cyan node with glow ring
- Hover: node brightens, shows section name tooltip
- Click: forward-pass particle animation to that layer
- Scroll-synced: nodes light up as you scroll past sections
- The connections between nodes show flowing particles (direction = scroll direction)

---

## Page Designs

### 1. Landing / Hero ("Input Layer")

```
┌──────────────────────────────────────────────────────────┐
│                                                            │
│     [Neural network canvas fills background]               │
│                                                            │
│           ╭─────────────────────────────╮                  │
│           │                             │                  │
│           │   V A S U   A G R A W A L   │  ← Glitch-in    │
│           │                             │                  │
│           │   AI/ML Engineer            │  ← Typed out     │
│           │   & Full-Stack Builder      │                  │
│           │                             │                  │
│           │   Building deployable       │                  │
│           │   intelligence systems.     │                  │
│           │                             │                  │
│           │  [Explore Network]  [CV]    │                  │
│           │                             │                  │
│           ╰─────────────────────────────╯                  │
│                                                            │
│   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐    │
│   │ 10       │ │ 3.82     │ │ 8        │ │ FULL     │    │
│   │ Projects │ │ CGPA     │ │ Certs    │ │ Stack    │    │
│   │ ◉ ◉ ◉   │ │ ◉ ◉ ◉   │ │ ◉ ◉ ◉   │ │ ◉ ◉ ◉   │    │
│   └──────────┘ └──────────┘ └──────────┘ └──────────┘    │
│   ↑ Stats as floating "telemetry nodes" with pulse        │
│                                                            │
└──────────────────────────────────────────────────────────┘
```

**Animations:**
1. Page load: Neural network canvas fades in with nodes appearing one by one
2. Name: Letters appear with a subtle glitch/decode effect (not typed — decoded)
3. Subtitle: Smooth fade-in after name settles
4. Stats: Float up from below with staggered delay, gentle parallax on mouse move
5. "Explore Network" button: Has a neural pulse animation on hover (ripple outward)
6. Background particles flow continuously through the network

**Boot Sequence Replacement:**
Instead of terminal boot logs, show a **"Network Initialization"** sequence:
- Neural network nodes appear one by one, from input → output
- Connections draw themselves between nodes
- A forward pass of data particles flows through the entire network
- The network "activates" and the content fades in
- Total: 2-3 seconds (skip with click or reduced-motion)

### 2. Projects ("Hidden Layer 1")

```
┌──────────────────────────────────────────────────────────┐
│                                                            │
│  HIDDEN LAYER 1: TRAINED MODELS                ◉ Filter   │
│  ─────────────────────────────                             │
│                                                            │
│  ┌─────────────────┐  ┌─────────────────┐  ┌────────────┐│
│  │ ◉ GeoVision-LULC│  │ ◉ Traffic       │  │ ◉ Insightify││
│  │   ┈┈┈┈┈┈┈┈┈┈┈   │  │   Analytics     │  │            ││
│  │ [S] RESEARCH     │  │   ┈┈┈┈┈┈┈┈┈┈┈  │  │ [S] LIVE   ││
│  │                   │  │ [S] LIVE        │  │            ││
│  │ SegFormer         │  │                 │  │ MERN  NLP  ││
│  │ DeepLabV3+        │  │ YOLO  Kalman    │  │            ││
│  │ PyTorch           │  │ Filter          │  │            ││
│  │                   │  │                 │  │            ││
│  │ ──────────────── │  │ ──────────────  │  │────────────││
│  │ mIoU: 0.461      │  │ Accuracy: 95%   │  │ Acc: 98%   ││
│  │ [↗ Deep Dive]    │  │ [↗ Deep Dive]   │  │[↗ Deep Dive]││
│  └─────────────────┘  └─────────────────┘  └────────────┘│
│                                                            │
│  ↑ Cards float at slightly different Z-depths              │
│  ↑ Hover: card comes forward, neural glow ring appears     │
│  ↑ The "neural connection" line from nav pulses to cards    │
│                                                            │
└──────────────────────────────────────────────────────────┘
```

**Card Design:**
- Glass card with subtle gradient border
- Top-left: A pulsing neural node (color based on tier: cyan=S, purple=A, blue=B)
- Tier badge styled as a node classification label
- Status badge with live indicator dot
- Tech tags as small pills with hover glow
- Bottom: Key metric highlighted + "Deep Dive →" link
- Hover: Card lifts, border glows, the node at top-left emits particles
- Click tech tag: Filters + shows "connection paths" only to matching projects

**Project Deep Dive Page:**
Blueprint-inspired technical breakdown:
```
┌──────────────────────────────────────────────────────────┐
│  ← Back to Network                                        │
│                                                            │
│  ◉ GeoVision-LULC                          [S] RESEARCH   │
│  ═══════════════════════════════════                       │
│                                                            │
│  ┌─── PROBLEM ──────────────────────────────────────────┐ │
│  │ Accurate large-scale LULC mapping requires models    │ │
│  │ that balance spatial precision and computational...  │ │
│  └──────────────────────────────────────────────────────┘ │
│                                                            │
│  ┌─── ARCHITECTURE ──── (blueprint style) ──────────────┐ │
│  │                                                       │ │
│  │  Input ──→ Backbone ──→ Decoder ──→ Output           │ │
│  │  [Sentinel-2]  [SegFormer]  [MLP]   [GeoTIFF]       │ │
│  │                                                       │ │
│  │  ·· Annotated with callout lines ··                   │ │
│  └──────────────────────────────────────────────────────┘ │
│                                                            │
│  ┌─── METRICS ──────────┐  ┌─── FAILURES ──────────────┐ │
│  │ mIoU: 0.461 (+8.7%)  │  │ RGB-only input limits    │ │
│  │ Model: 10.6x smaller │  │ separability of agri...  │ │
│  └──────────────────────┘  └──────────────────────────┘  │
│                                                            │
└──────────────────────────────────────────────────────────┘
```

### 3. Skills ("Hidden Layer 2")

```
┌──────────────────────────────────────────────────────────┐
│                                                            │
│  HIDDEN LAYER 2: WEIGHTS & BIASES                          │
│  ────────────────────────────────                          │
│                                                            │
│  ┌─ CORE ML / CV ──────────────────────────────────────┐  │
│  │                                                       │  │
│  │  ◉━━━━━━━━━━━━━ PyTorch           ████████████░ 95%  │  │
│  │  ◉━━━━━━━━━━━━  TensorFlow        ████████████░ 90%  │  │
│  │  ◉━━━━━━━━━━━   Semantic Seg      ███████████░░ 85%  │  │
│  │  ◉━━━━━━━━━━    Object Detection  ██████████░░░ 80%  │  │
│  │                                                       │  │
│  │  ↑ Each skill is a node with connection weight bar     │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                            │
│  ┌─ TRAINING EPOCHS (Career Timeline) ─────────────────┐  │
│  │                                                       │  │
│  │  2021 ──◉── 2022 ──◉── 2023 ──◉── 2024 ──◉──→      │  │
│  │  ML          DL/CNN     Full-Stack   LLMs &           │  │
│  │  Fundamentals           AI           Agents           │  │
│  │                                                       │  │
│  │  ↑ Horizontal timeline with expanding nodes           │  │
│  │  ↑ Click epoch → shows skills learned in that period  │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                            │
│  ┌─ CERTIFICATIONS (Validated Weights) ────────────────┐  │
│  │  ◉ AWS ML Specialty  ◉ Deep Learning  ◉ TensorFlow  │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                            │
└──────────────────────────────────────────────────────────┘
```

### 4. Blog ("Hidden Layer 3: Signal Propagation")

```
┌──────────────────────────────────────────────────────────┐
│                                                            │
│  HIDDEN LAYER 3: SIGNAL PROPAGATION                        │
│  ──────────────────────────────────                        │
│                                                            │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  ◉ Signal #5                          March 2025      │  │
│  │  ╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌                    │  │
│  │  ~~~∿∿∿~~~∿∿∿~~~  ← waveform decoration              │  │
│  │                                                        │  │
│  │  Title of the Blog Post                                │  │
│  │  Brief preview text that gives context...              │  │
│  │                                                        │  │
│  │  #tag1  #tag2  #tag3               4 min read          │  │
│  │  [Decode Signal →]                                     │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                            │
│  ↑ Cards have a subtle waveform line across the top        │
│  ↑ Hover: waveform animates, card glows                    │
│  ↑ Click: "decoding" animation → blog reader opens         │
│                                                            │
└──────────────────────────────────────────────────────────┘
```

### 5. About ("Output Layer")

```
┌──────────────────────────────────────────────────────────┐
│                                                            │
│  OUTPUT LAYER: ABOUT                                       │
│  ───────────────────                                       │
│                                                            │
│  ┌─────────────────────────────────────────────────────┐  │
│  │                                                       │  │
│  │   ◉ VASU AGRAWAL                                     │  │
│  │   AI/ML Engineer & Full-Stack Developer              │  │
│  │                                                       │  │
│  │   [Bio paragraph — clean, readable]                  │  │
│  │                                                       │  │
│  │   ┌─ ROLES ────────────┐  ┌─ EDUCATION ───────────┐ │  │
│  │   │ ML Engineer         │  │ B.Tech CS              │ │  │
│  │   │ Full-Stack Dev      │  │ CGPA: 3.82             │ │  │
│  │   │ System Designer     │  │                        │ │  │
│  │   └────────────────────┘  └───────────────────────┘ │  │
│  │                                                       │  │
│  │   ┌─ ESTABLISH CONNECTION ──────────────────────────┐ │  │
│  │   │  ◉ GitHub    ◉ LinkedIn    ◉ Email              │ │  │
│  │   │  ↑ Nodes with neural connection lines            │ │  │
│  │   └─────────────────────────────────────────────────┘ │  │
│  │                                                       │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                            │
└──────────────────────────────────────────────────────────┘
```

### 6. AI Chat: "Synapse" (NOT a Terminal)

**Concept:** A floating neural node in the bottom-right that expands into a
conversational AI interface. It looks like you're communicating directly with the
neural network.

```
Collapsed (corner):          Expanded:
                              ┌─────────────────────────────┐
                              │  ◉ SYNAPSE              ─ ✕ │
    ◉  ← Pulsing neuron      │  Neural Intelligence Link    │
    (click to expand)         │  ───────────────────────────│
                              │                             │
                              │  ◉ Ask me anything about    │
                              │    Vasu's work, projects,   │
                              │    or skills.               │
                              │                             │
                              │  ┌─ YOU ─────────────────┐  │
                              │  │ What's his best       │  │
                              │  │ project?              │  │
                              │  └───────────────────────┘  │
                              │                             │
                              │  ◉◉◉ Processing...         │
                              │  ↑ mini neural network      │
                              │    animation showing a      │
                              │    forward pass             │
                              │                             │
                              │  ┌─ SYNAPSE ─────────────┐  │
                              │  │ Based on my analysis, │  │
                              │  │ GeoVision-LULC is the │  │
                              │  │ highest-signal work...│  │
                              │  └───────────────────────┘  │
                              │                             │
                              │  ┌────────────────────┐     │
                              │  │ Type a question...  │ ➤  │
                              │  └────────────────────┘     │
                              └─────────────────────────────┘
```

**Key Differences from Terminal:**
1. No command syntax — pure natural language
2. "Processing" shows a mini neural network doing a forward pass (3-4 nodes animating)
3. Messages styled as neural signals, not terminal output
4. Suggested questions as "pre-trained prompts" (clickable chips)
5. Smooth slide-up panel (not modal overlay) — content stays visible behind
6. Can still navigate: "Show me the GeoVision project" → navigates there

---

## Animation Inventory

| Animation              | Where                  | Technique                  | Duration |
|------------------------|------------------------|----------------------------|----------|
| Network initialization | Landing (boot)         | Canvas, staggered nodes    | 2.5s     |
| Data particle flow     | Background, always     | Canvas requestAnimationFrame| Infinite |
| Node pulse/breathe     | Nav nodes, card nodes  | CSS keyframes              | 3s loop  |
| Glitch-decode text     | Hero name              | CSS clip-path + opacity    | 1.2s     |
| Card float-up          | Section enter          | Framer Motion, staggered   | 0.4s     |
| Card hover lift        | Project/blog cards     | Framer Motion whileHover   | 0.2s     |
| Neural glow ring       | Active nav, hover      | CSS box-shadow animation   | 0.3s     |
| Forward pass           | Page transition        | Canvas particle burst      | 0.8s     |
| Waveform draw          | Blog cards top border  | SVG stroke-dashoffset      | 1s       |
| Synapse expand         | AI chat open           | Framer Motion layout       | 0.3s     |
| Processing pass        | AI thinking            | Mini canvas animation      | Loop     |
| Parallax float         | Stats, background      | Mouse-move transform       | Instant  |
| Skill weight bars      | Skills section enter   | CSS width transition       | 0.6s     |
| Timeline epoch         | Skills career line     | SVG draw + node pop        | 1.2s     |
| Blueprint annotations  | Project deep dive      | SVG line draw + fade       | 0.8s     |

---

## File Structure (New/Modified)

```
src/
├── components/
│   ├── canvas/
│   │   ├── NeuralNetwork.jsx      ← Main neural network canvas (background)
│   │   ├── useNeuralNetwork.js     ← Canvas logic hook (nodes, connections, particles)
│   │   ├── MiniNetwork.jsx         ← Small network for Synapse processing animation
│   │   └── Waveform.jsx            ← SVG waveform for blog cards
│   │
│   ├── layout/
│   │   ├── LayerNav.jsx            ← Neural layer navigation (replaces Sidebar)
│   │   ├── LayerNavNode.jsx        ← Individual nav node
│   │   ├── Header.jsx              ← Updated header with neural theme
│   │   ├── MobileNav.jsx           ← Updated mobile nav
│   │   └── SectionWrapper.jsx      ← Updated with layer labels
│   │
│   ├── sections/
│   │   ├── Hero.jsx                ← New landing (replaces Overview)
│   │   ├── Projects.jsx            ← Redesigned with neural cards
│   │   ├── ProjectCard.jsx         ← Neural node card design
│   │   ├── Skills.jsx              ← Weights & biases design
│   │   ├── Blog.jsx                ← Signal propagation design
│   │   ├── About.jsx               ← Output layer design
│   │   ├── NotFound.jsx            ← Updated 404
│   │   └── Pipeline.jsx            ← Kept or merged into Hero
│   │
│   ├── views/
│   │   ├── ProjectDeepDive.jsx     ← Blueprint-style redesign
│   │   └── BlogReader.jsx          ← Updated reader
│   │
│   ├── synapse/
│   │   ├── SynapseButton.jsx       ← Floating neuron trigger
│   │   ├── SynapsePanel.jsx        ← Chat panel UI
│   │   ├── SynapseMessage.jsx      ← Message bubble component
│   │   └── SynapseSuggestions.jsx   ← Pre-trained prompt chips
│   │
│   ├── effects/
│   │   ├── GlitchText.jsx          ← Glitch-decode text effect
│   │   ├── NeuralGlow.jsx          ← Reusable glow ring effect
│   │   ├── ParallaxFloat.jsx       ← Mouse-responsive parallax wrapper
│   │   └── NetworkInit.jsx         ← Boot/initialization sequence
│   │
│   └── ui/
│       ├── GlassCard.jsx           ← Reusable glassmorphism card
│       ├── NeuralBadge.jsx         ← Tier/status badge (neural style)
│       ├── TechPill.jsx            ← Tech tag pill with glow
│       ├── WeightBar.jsx           ← Skill proficiency bar
│       ├── ErrorBoundary.jsx       ← Kept
│       └── SkipToContent.jsx       ← Kept
│
├── hooks/
│   ├── useNeuralBackground.js      ← Hook for neural network canvas state
│   ├── useParallax.js              ← Mouse parallax tracking
│   ├── useGlitchText.js            ← Glitch text animation hook
│   ├── useSynapse.js               ← AI chat state management
│   ├── useBootSequence.js          ← Updated: network initialization
│   ├── useDocumentTitle.js         ← Kept
│   ├── useKeyboardShortcuts.js     ← Updated shortcuts
│   ├── useFocusTrap.js             ← Kept
│   └── useUI.js                    ← Kept
│
├── styles/
│   └── animations.css              ← All keyframe animations
│
├── data/                           ← Kept as-is (data layer unchanged)
├── context/                        ← Kept as-is
└── utils/                          ← Kept as-is
```

---

## Implementation Phases

### Phase 1: Design Foundation (Day 1)
1. Update `index.css` with new design tokens (CSS custom properties)
2. Add Google Fonts: Space Grotesk + JetBrains Mono
3. Create `animations.css` with all keyframe definitions
4. Create `GlassCard.jsx` base component
5. Create `NeuralBadge.jsx` and `TechPill.jsx`
6. Create `WeightBar.jsx`

### Phase 2: Neural Network Canvas (Day 2)
1. Build `useNeuralNetwork.js` hook (node management, particle system, connections)
2. Build `NeuralNetwork.jsx` canvas component (replaces ParticleField)
3. Add mouse interaction (nodes brighten near cursor)
4. Add section-awareness (current layer highlights)
5. Add forward-pass animation function (data particles flow between layers)

### Phase 3: Navigation Redesign (Day 2-3)
1. Build `LayerNavNode.jsx` (single node with label, glow, connection line)
2. Build `LayerNav.jsx` (vertical chain of nodes, replaces Sidebar)
3. Wire up route-awareness (active node matches current route)
4. Add forward-pass transition on navigation
5. Update `MobileNav.jsx` (horizontal node chain)
6. Update `App.jsx` layout structure

### Phase 4: Hero / Landing Page (Day 3)
1. Build `GlitchText.jsx` effect component
2. Build `ParallaxFloat.jsx` wrapper
3. Build `NetworkInit.jsx` (replaces BootSequence)
4. Redesign `Hero.jsx` (replaces Overview)
5. Stats as floating telemetry nodes with parallax
6. CTA buttons with neural pulse hover

### Phase 5: Projects Redesign (Day 4)
1. Redesign `ProjectCard.jsx` (glass card + neural node + tier glow)
2. Update `Projects.jsx` (section header + layout)
3. Redesign `ProjectDeepDive.jsx` (blueprint-style with annotations)
4. Add card hover animations (lift + glow + particle emit)

### Phase 6: Skills Redesign (Day 5)
1. Build `WeightBar.jsx` animations (width transition on scroll-in)
2. Redesign `Skills.jsx` (weights & biases layout)
3. Career timeline as "training epochs" (horizontal node chain)
4. Certifications as "validated weights"

### Phase 7: Blog Redesign (Day 5)
1. Build `Waveform.jsx` SVG component
2. Redesign blog cards (waveform header + signal theme)
3. Update `BlogReader.jsx` styling
4. "Decode Signal" click animation

### Phase 8: About / Output Layer (Day 6)
1. Redesign `About.jsx` (output layer theme)
2. Contact links as neural connection nodes
3. Social links with connection-line animations

### Phase 9: Synapse AI Chat (Day 6-7)
1. Build `SynapseButton.jsx` (floating neuron with pulse)
2. Build `MiniNetwork.jsx` (processing animation)
3. Build `SynapseMessage.jsx` (message bubbles)
4. Build `SynapseSuggestions.jsx` (pre-trained prompt chips)
5. Build `SynapsePanel.jsx` (full chat interface)
6. Build `useSynapse.js` hook (state + AI integration)
7. Wire up existing Gemini proxy backend

### Phase 10: Page Transitions & Polish (Day 7)
1. Add page transition animations (forward-pass between sections)
2. Smooth scroll behavior
3. Reduced motion fallbacks for all animations
4. Mobile responsiveness pass on all components
5. Accessibility audit (focus states, ARIA, keyboard nav)
6. Performance optimization (canvas throttling, lazy loading)
7. Final visual QA

---

## Interaction Details

### Mouse Parallax
Stats cards and background elements respond to mouse position:
```
offsetX = (mouseX - centerX) * 0.02
offsetY = (mouseY - centerY) * 0.02
transform: translate(offsetX, offsetY)
```
Different depths have different multipliers (0.01 to 0.04).

### Neural Network Interaction
- Mouse proximity: Nodes within 150px of cursor glow brighter
- Click on nav node: Forward-pass particle burst from current → target
- Scroll: Connections subtly shift to show flow direction
- Idle: Gentle breathing animation on all nodes

### Card Interactions
- Hover: `translateY(-4px)`, border glow, node particle emission
- Click: Quick scale(0.98) → navigate with forward-pass transition
- Filter: Non-matching cards dim (opacity 0.3), matching cards pulse

### Synapse Chat
- Open: Slide up from bottom-right, 400px wide, max 60vh tall
- Close: Slide down with fade
- Message send: Input signal animation (pulse from input → network → output)
- Processing: Mini neural network nodes light up sequentially
- Response arrive: Fade in with subtle left-slide

---

## Mobile Strategy

**Philosophy:** Simplified but still distinctive. The neural theme persists
but complex canvas animations are reduced.

- Neural network canvas: Reduced to 30-50 nodes (vs 100+ desktop)
- Navigation: Horizontal node chain at top (scrollable if needed)
- Cards: Single column, full width
- Synapse: Full-width bottom sheet instead of side panel
- Parallax: Disabled (gyroscope optional)
- Animations: Reduced but not eliminated — stagger reveals kept
- Touch: Swipe between sections, tap to expand

---

## Accessibility Checklist

- [ ] All animations respect `prefers-reduced-motion`
- [ ] Neural network canvas has `aria-hidden="true"`
- [ ] Skip-to-content link preserved
- [ ] Focus-visible styles on all interactive elements (cyan glow ring)
- [ ] ARIA labels on all icon buttons
- [ ] Synapse chat: `role="dialog"`, focus trap, `aria-live` for messages
- [ ] Color contrast: All text meets WCAG AA (4.5:1)
- [ ] Keyboard navigation: Tab through nav nodes, Enter to activate
- [ ] Semantic HTML: section, nav, main, article elements
- [ ] Screen reader: Section labels announce layer names

---

## Performance Budget

- Neural network canvas: max 60fps, throttle to 30fps on battery
- Max nodes: 120 (desktop), 40 (mobile)
- Max particles: 200 (desktop), 50 (mobile)
- Canvas resolution: Scale to devicePixelRatio (cap at 2x)
- Lazy load: All section components, Synapse panel, deep-dive views
- Font loading: `display=swap`, preconnect
- Image: Convert remaining GIFs to WebM
- Target: Lighthouse Performance > 90
