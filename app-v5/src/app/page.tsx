import { sections } from "@/data/sections-v5";
import { projects } from "@/data/projects-v5";
import { skills, edges } from "@/data/skills-graph";

/**
 * Phase-0 placeholder. Exercises the design tokens and the data layer so the
 * build proves both compile. Replaced by the real section components in Phase 1.
 */
export default function Home() {
  return (
    <main className="relative z-1 mx-auto max-w-3xl px-6 py-[var(--space-section)]">
      <p
        className="text-[length:var(--text-xs)] tracking-[var(--tracking-terminal)] uppercase"
        style={{ color: "var(--accent)" }}
      >
        VASU_OS 5 — neural core / phase 0
      </p>
      <h1
        className="mt-4 font-bold leading-[var(--leading-tight)]"
        style={{ fontSize: "var(--text-hero)" }}
      >
        Vasu Agrawal
      </h1>
      <p className="mt-6" style={{ color: "var(--text-muted)" }}>
        AI Developer — production LLM systems and agentic AI. Scaffold online:{" "}
        {sections.length} sections · {projects.length} project nodes ·{" "}
        {skills.length} skills · {edges.length} graph edges.
      </p>
    </main>
  );
}
