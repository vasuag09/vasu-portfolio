import { CanvasRoot } from "@/components/canvas/CanvasRoot";
import { SmoothScroll } from "@/components/scroll/SmoothScroll";
import { ChapterNav } from "@/components/nav/ChapterNav";
import { Section } from "@/components/layout/Section";
import { projects } from "@/data/projects-v5";
import { skills } from "@/data/skills-graph";

/**
 * Phase-1 walking skeleton: five real-DOM chapters over the fixed canvas.
 * Copy is placeholder voice — final flagship copy lands in Phase 5.
 */

const flagships = projects.filter((p) => p.tier === "flagship");

export default function Home() {
  return (
    <>
      <SmoothScroll />
      <CanvasRoot />
      <ChapterNav />

      <main className="relative" style={{ zIndex: "var(--z-content)" }}>
        <Section id="hero" label="Neural Core" heightSvh={220}>
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
        </Section>

        <Section id="projects" label="Projects" heightSvh={280}>
          <h2
            className="mt-4 font-bold leading-[var(--leading-tight)]"
            style={{ fontSize: "var(--text-xl)" }}
          >
            Deployments
          </h2>
          <ul className="mt-6 flex flex-col gap-4">
            {flagships.map((project) => (
              <li key={project.id}>
                <h3 className="font-medium" style={{ fontSize: "var(--text-base)" }}>
                  {project.title}
                </h3>
                <p
                  className="mt-1 max-w-xl"
                  style={{
                    color: "var(--text-muted)",
                    fontSize: "var(--text-sm)",
                  }}
                >
                  {project.oneLiner}
                </p>
              </li>
            ))}
          </ul>
        </Section>

        <Section id="skills" label="Skills Graph" heightSvh={280}>
          <h2
            className="mt-4 font-bold leading-[var(--leading-tight)]"
            style={{ fontSize: "var(--text-xl)" }}
          >
            Weights &amp; Biases
          </h2>
          <p className="mt-4 max-w-xl" style={{ color: "var(--text-muted)" }}>
            {skills.length} skills wired to the projects that actually use them
            — a real graph, not decoration. Interactive in Phase 3.
          </p>
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
        </Section>
      </main>
    </>
  );
}
