import { BootSequence } from "@/components/boot/BootSequence";
import { CustomCursor } from "@/components/cursor/CustomCursor";
import { CanvasRoot } from "@/components/canvas/CanvasRoot";
import { SmoothScroll } from "@/components/scroll/SmoothScroll";
import { ChapterNav } from "@/components/nav/ChapterNav";
import { Section } from "@/components/layout/Section";
import { SkillsOverlay } from "@/components/graph/SkillsOverlay";
import { ProjectList } from "@/components/graph/ProjectList";
import { CaseStudyPanel } from "@/components/graph/CaseStudyPanel";
import { SynapseTerminal } from "@/components/synapse/SynapseTerminal";
import { SynapseTrigger } from "@/components/synapse/SynapseTrigger";

/**
 * Five real-DOM chapters over the fixed canvas. The skills and projects
 * regions are interactive (Phase 3): DOM hover/selection drives the scene
 * glow through the graph store. Final flagship copy lands in Phase 5.
 */

export default function Home() {
  return (
    <>
      <BootSequence />
      <CustomCursor />
      <SmoothScroll />
      <CanvasRoot />
      <ChapterNav />
      <CaseStudyPanel />
      <SynapseTerminal />

      <main className="relative" style={{ zIndex: "var(--z-content)" }}>
        <Section id="hero" label="Neural Core" heightSvh={220} align="start">
          <h1
            className="mt-4 font-bold leading-[var(--leading-tight)]"
            style={{ fontSize: "var(--text-hero)" }}
          >
            Vasu Agrawal
          </h1>
          <p
            className="mt-6 max-w-xl"
            style={{ color: "var(--text-muted)", fontSize: "var(--text-lg)" }}
          >
            AI Developer. Building production LLM systems and agentic AI — from
            campus chatbots to WhatsApp-native B2B commerce.
          </p>
          <div className="mt-8">
            <SynapseTrigger />
          </div>
        </Section>

        <Section id="projects" label="Projects" heightSvh={280}>
          <h2
            className="mt-4 font-bold leading-[var(--leading-tight)]"
            style={{ fontSize: "var(--text-xl)" }}
          >
            Deployments
          </h2>
          <ProjectList />
        </Section>

        <Section id="skills" label="Skills Graph" heightSvh={280}>
          <h2
            className="mt-4 font-bold leading-[var(--leading-tight)]"
            style={{ fontSize: "var(--text-xl)" }}
          >
            Weights &amp; Biases
          </h2>
          <p className="mt-4 max-w-xl" style={{ color: "var(--text-muted)" }}>
            Every skill below is wired to the projects that actually use it —
            a real graph, not decoration. Hover to trace the connections.
          </p>
          <div className="mt-8">
            <SkillsOverlay />
          </div>
        </Section>

        <Section id="about" label="About" heightSvh={240}>
          <h2
            className="mt-4 font-bold leading-[var(--leading-tight)]"
            style={{ fontSize: "var(--text-xl)" }}
          >
            Output Layer
          </h2>
          <p className="mt-4 max-w-xl" style={{ color: "var(--text-muted)" }}>
            Mumbai-based. Shipped systems used by 1,200+ students, architected a
            17-tool LLM agent loop for a live B2B pharmacy platform, and
            contributed observability code to Ray Serve&apos;s LLM routing layer.
          </p>
        </Section>

        <Section id="contact" label="Contact" heightSvh={160}>
          <h2
            className="mt-4 font-bold leading-[var(--leading-tight)]"
            style={{ fontSize: "var(--text-xl)" }}
          >
            Open a connection
          </h2>
          <p className="mt-4" style={{ color: "var(--text-muted)" }}>
            <a
              className="underline decoration-[var(--accent-dim)] underline-offset-4 hover:decoration-[var(--accent)]"
              href="mailto:vasuagrawal1040@gmail.com"
            >
              vasuagrawal1040@gmail.com
            </a>{" "}
            · GitHub{" "}
            <a
              className="underline decoration-[var(--accent-dim)] underline-offset-4 hover:decoration-[var(--accent)]"
              href="https://github.com/vasuag09"
            >
              vasuag09
            </a>
          </p>
          <div className="mt-6">
            <SynapseTrigger />
          </div>
        </Section>
      </main>
    </>
  );
}
