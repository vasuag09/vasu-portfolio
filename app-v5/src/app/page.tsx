import { BootSequence } from "@/components/boot/BootSequence";
import { SoundController } from "@/components/sound/SoundController";
import { SoundToggle } from "@/components/sound/SoundToggle";
import { CustomCursor } from "@/components/cursor/CustomCursor";
import { TabTitleEffect } from "@/components/TabTitleEffect";
import { CanvasRoot } from "@/components/canvas/CanvasRoot";
import { SmoothScroll } from "@/components/scroll/SmoothScroll";
import { ChapterNav } from "@/components/nav/ChapterNav";
import { Section } from "@/components/layout/Section";
import { CharReveal } from "@/components/ui/CharReveal";
import { Reveal } from "@/components/ui/Reveal";
import { SkillsOverlay } from "@/components/graph/SkillsOverlay";
import { ProjectList } from "@/components/graph/ProjectList";
import { CaseStudyPanel } from "@/components/graph/CaseStudyPanel";
import { SynapseTerminal } from "@/components/synapse/SynapseTerminal";
import { SynapseTrigger } from "@/components/synapse/SynapseTrigger";

/**
 * Five real-DOM chapters over the fixed canvas — each with its own
 * editorial composition (design elevation): bespoke asymmetric hero,
 * split-editorial projects/about, end-aligned skills, centered contact.
 */

export default function Home() {
  return (
    <>
      <BootSequence />
      <SoundController />
      <SoundToggle />
      <CustomCursor />
      <TabTitleEffect />
      <SmoothScroll />
      <CanvasRoot />
      <ChapterNav />
      <CaseStudyPanel />
      <SynapseTerminal />

      <main className="relative" style={{ zIndex: "var(--z-content)" }}>
        {/* ---- 01 · Hero: bespoke asymmetric composition ---- */}
        <Section id="hero" label="Neural Core" heightSvh={220} align="start" index={1}>
          <Reveal>
            <div className="flex items-baseline justify-between gap-6">
              <p
                className="text-[length:var(--text-xs)] tracking-[var(--tracking-terminal)] uppercase"
                style={{ color: "var(--accent)" }}
              >
                Neural Core · online
              </p>
              <p
                className="hidden text-[length:var(--text-xs)] tracking-[var(--tracking-wide)] uppercase md:block"
                style={{ color: "var(--text-faint)" }}
              >
                19.07°N 72.87°E · Mumbai
              </p>
            </div>
          </Reveal>
          <h1
            className="mt-6 font-bold leading-[0.95]"
            style={{ fontSize: "var(--text-hero)" }}
          >
            <CharReveal text="Vasu Agrawal" />
          </h1>
          <div className="mt-10 grid gap-8 md:grid-cols-12">
            <Reveal delay={140} className="md:col-span-5">
              <p
                className="text-[length:var(--text-xs)] tracking-[var(--tracking-terminal)] uppercase"
                style={{ color: "var(--text-faint)" }}
              >
                AI Developer
              </p>
              <div className="mt-4">
                <SynapseTrigger primary />
              </div>
            </Reveal>
            <Reveal delay={240} className="md:col-span-6 md:col-start-7">
              <p style={{ color: "var(--text-muted)", fontSize: "var(--text-lg)" }}>
                Building production LLM systems and agentic AI — from campus
                chatbots to WhatsApp-native B2B commerce.
              </p>
            </Reveal>
          </div>
        </Section>

        {/* ---- 02 · Projects: split editorial ---- */}
        <Section
          id="projects"
          label="Projects"
          heightSvh={280}
          index={2}
          title="Deployments"
          variant="split"
        >
          <ProjectList />
        </Section>

        {/* ---- 03 · Skills: end-aligned, full-width graph overlay ---- */}
        <Section
          id="skills"
          label="Skills Graph"
          heightSvh={280}
          index={3}
          title="Weights & Biases"
          variant="end"
        >
          <p
            className="mt-4 ml-auto max-w-xl text-right"
            style={{ color: "var(--text-muted)" }}
          >
            Every skill below is wired to the projects that actually use it —
            a real graph, not decoration. Hover to trace the connections.
          </p>
          <div className="mt-10">
            <SkillsOverlay />
          </div>
        </Section>

        {/* ---- 04 · About: split editorial ---- */}
        <Section
          id="about"
          label="About"
          heightSvh={240}
          index={4}
          title="Output Layer"
          variant="split"
        >
          <p style={{ color: "var(--text-muted)", fontSize: "var(--text-lg)" }}>
            Mumbai-based. Shipped systems used by 1,200+ students, architected a
            17-tool LLM agent loop for a live B2B pharmacy platform, and
            contributed observability code to Ray Serve&apos;s LLM routing layer.
          </p>
        </Section>

        {/* ---- 05 · Contact: centered, oversized link ---- */}
        <Section
          id="contact"
          label="Contact"
          heightSvh={160}
          index={5}
          title="Open a connection"
          variant="center"
        >
          <p className="mt-6">
            <a
              className="font-bold underline decoration-[var(--accent-dim)] decoration-2 underline-offset-8 transition-colors duration-[var(--duration-fast)] hover:decoration-[var(--accent)]"
              style={{ fontSize: "var(--text-xl)" }}
              href="mailto:vasuagrawal1040@gmail.com"
            >
              vasuagrawal1040@gmail.com
            </a>
          </p>
          <p className="mt-4" style={{ color: "var(--text-muted)" }}>
            GitHub{" "}
            <a
              className="underline decoration-[var(--accent-dim)] underline-offset-4 hover:decoration-[var(--accent)]"
              href="https://github.com/vasuag09"
            >
              vasuag09
            </a>
          </p>
          <div className="mt-8">
            <SynapseTrigger primary />
          </div>
        </Section>
      </main>
    </>
  );
}
