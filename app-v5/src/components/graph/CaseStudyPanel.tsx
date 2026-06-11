"use client";

import { useEffect, useRef } from "react";
import { getProject, projects } from "@/data/projects-v5";
import { getSkillsForProject } from "@/lib/graph-adjacency";
import { getSkill } from "@/data/skills-graph";
import { setGraphState } from "@/lib/graph-store";
import { afterNextPaint } from "@/lib/after-paint";
import { useGraphState } from "@/hooks/useGraphState";
import { useScrollLock } from "@/hooks/useScrollLock";
import { scrollToSection, isSectionInView } from "@/lib/scroll-to-section";
import { VeoClip } from "@/components/media/VeoClip";
import { veoSources, veoPoster } from "@/lib/veo-sources";
import { Reveal } from "@/components/ui/Reveal";
import type { ProjectStatus } from "@/data/types";

/**
 * Case-study panel (Phase 5) — winner anatomy per AWARD-RESEARCH §3.5:
 * hero → problem → role → visual payload (Veo) → outcome metrics →
 * next-project funnel. Non-modal side dialog; real DOM; Esc closes; focus
 * managed; ?project= deep links. Reveals fire once per block (scoped to
 * the panel's own scroll root) and animate transform/opacity only — the
 * panel is a fixed overlay, so document CLS is structurally zero.
 */

const STATUS_LABEL: Record<ProjectStatus, string> = {
  production: "In production",
  live: "Live",
  "under-review": "PR under review",
  active: "Active",
  archived: "Archived",
};

const KNOWN_IDS = new Set(projects.map((p) => p.id));
const FLAGSHIP_ORDER = projects
  .filter((p) => p.tier === "flagship")
  .map((p) => p.id);

// Close defers the unmount + scroll-unlock past the next paint — the
// full-document relayout otherwise lands inside the interaction (Phase-8
// INP finding: 300–550ms close frames on an unthrottled CPU).
function closeCaseStudy() {
  afterNextPaint(() => setGraphState({ selectedProjectId: null }));
}

function nextFlagshipId(currentId: string): string {
  const index = FLAGSHIP_ORDER.indexOf(currentId);
  return FLAGSHIP_ORDER[(index + 1) % FLAGSHIP_ORDER.length];
}

function syncUrl(projectId: string | null) {
  const url = new URL(window.location.href);
  if (projectId) url.searchParams.set("project", projectId);
  else url.searchParams.delete("project");
  window.history.replaceState(null, "", url);
}

function SectionHeading({ children }: { children: string }) {
  return (
    <h3
      className="mt-8 text-[length:var(--text-xs)] tracking-[var(--tracking-terminal)] uppercase"
      style={{ color: "var(--accent)" }}
    >
      {children}
    </h3>
  );
}

export function CaseStudyPanel() {
  const { selectedProjectId } = useGraphState();
  const panelRef = useRef<HTMLDivElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const restoreFocusRef = useRef<HTMLElement | null>(null);

  // The page must not scroll behind the panel (Lenis listens on window —
  // wheel over the panel would move the document, not the case study).
  useScrollLock(Boolean(selectedProjectId));

  // Deep link in: ?project=nm-gpt opens the panel on load.
  useEffect(() => {
    const requested = new URLSearchParams(window.location.search).get("project");
    if (requested && KNOWN_IDS.has(requested)) {
      setGraphState({ selectedProjectId: requested });
    }
  }, []);

  // URL + focus + camera alignment + panel scroll reset on open/switch.
  useEffect(() => {
    syncUrl(selectedProjectId);
    if (selectedProjectId) {
      if (!restoreFocusRef.current) {
        restoreFocusRef.current = document.activeElement as HTMLElement;
      }
      panelRef.current?.focus();
      scrollRef.current?.scrollTo({ top: 0 });
      // Camera rest-pose alignment: the glowing node lives in the projects
      // region — fly there so the scene context matches the case study.
      // force: the scroll lock has stopped Lenis; the flight still runs.
      if (!isSectionInView("projects")) {
        scrollToSection("projects", false, { force: true });
      }
    } else {
      restoreFocusRef.current?.focus();
      restoreFocusRef.current = null;
    }
  }, [selectedProjectId]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") closeCaseStudy();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  if (!selectedProjectId) return null;
  const project = getProject(selectedProjectId);
  const skillLabels = getSkillsForProject(project.id).map(
    (id) => getSkill(id).label,
  );
  const isFlagship = project.tier === "flagship";

  return (
    <>
      {/* Backdrop: recedes the page so panel and content never read as
          colliding layers; click closes. */}
      <div
        role="presentation"
        onClick={closeCaseStudy}
        className="fixed inset-0"
        style={{
          zIndex: "var(--z-overlay)",
          background: "oklch(8% 0.02 250 / 0.65)",
        }}
      />
      {/* div, not aside: ARIA-in-HTML forbids role="dialog" on aside (axe). */}
      <div
        ref={panelRef}
        role="dialog"
        aria-modal="true"
        aria-label={`${project.title} case study`}
        tabIndex={-1}
        className="fixed inset-y-0 right-0 w-full max-w-xl border-l outline-none"
        style={{
          zIndex: "var(--z-overlay)",
          background: "var(--bg-elevated)",
          borderColor: "var(--border)",
        }}
      >
        <div
          ref={scrollRef}
          data-lenis-prevent
          className="h-full overflow-y-auto overscroll-contain p-6 md:p-8"
        >
        {/* ---- Hero ---- */}
        <div className="flex items-start justify-between gap-4">
          <p
            className="text-[length:var(--text-xs)] tracking-[var(--tracking-terminal)] uppercase"
            style={{ color: "var(--accent)" }}
          >
            {STATUS_LABEL[project.status]}
          </p>
          <button
            type="button"
            onClick={closeCaseStudy}
            aria-label="Close case study"
            className="cursor-pointer rounded border px-2 py-0.5 text-[length:var(--text-sm)] transition-colors duration-[var(--duration-fast)] hover:border-[var(--border-active)]"
            style={{ borderColor: "var(--border)", color: "var(--text-muted)" }}
          >
            esc
          </button>
        </div>
        <h2
          className="mt-3 font-bold leading-[var(--leading-tight)]"
          style={{ fontSize: "var(--text-xl)" }}
        >
          {project.title}
        </h2>
        <p className="mt-3" style={{ color: "var(--text-muted)" }}>
          {project.oneLiner}
        </p>

        {/* ---- Problem ---- */}
        {project.problem ? (
          <Reveal root={scrollRef} delay={60}>
            <SectionHeading>The problem</SectionHeading>
            <p
              className="mt-2 text-[length:var(--text-sm)] leading-[var(--leading-body)]"
              style={{ color: "var(--text-muted)" }}
            >
              {project.problem}
            </p>
          </Reveal>
        ) : null}

        {/* ---- What it does ---- */}
        {project.narrative ? (
          <Reveal root={scrollRef} delay={120}>
            <SectionHeading>What it does</SectionHeading>
            <p
              className="mt-2 text-[length:var(--text-sm)] leading-[var(--leading-body)]"
              style={{ color: "var(--text-muted)" }}
            >
              {project.narrative}
            </p>
          </Reveal>
        ) : null}

        {/* ---- My role ---- */}
        {project.contribution ? (
          <Reveal root={scrollRef} delay={160}>
            <SectionHeading>My role</SectionHeading>
            <p
              className="mt-2 text-[length:var(--text-sm)] leading-[var(--leading-body)]"
              style={{ color: "var(--text-muted)" }}
            >
              {project.contribution}
            </p>
          </Reveal>
        ) : null}

        {/* ---- Visual payload (Veo) ---- */}
        {project.clip ? (
          <Reveal root={scrollRef}>
            <div className="mt-8">
              <VeoClip
                sources={veoSources(project.clip)}
                poster={veoPoster(project.clip)}
                label={`${project.title} — cinematic visualization`}
              />
            </div>
          </Reveal>
        ) : null}

        {/* ---- Outcome metrics ---- */}
        {project.metrics?.length ? (
          <Reveal root={scrollRef}>
            <SectionHeading>Numbers</SectionHeading>
            <ul className="mt-3 flex flex-col gap-3" role="list">
              {project.metrics.map((metric) => (
                <li
                  key={metric}
                  className="border-l-2 pl-3 font-medium"
                  style={{
                    borderColor: "var(--accent-dim)",
                    color: "var(--accent-bright)",
                  }}
                >
                  {metric}
                </li>
              ))}
            </ul>
          </Reveal>
        ) : null}

        {/* ---- Stack + wired skills ---- */}
        <Reveal root={scrollRef}>
          <SectionHeading>Stack</SectionHeading>
          <ul className="mt-2 flex flex-wrap gap-2" role="list">
            {project.stack.map((item) => (
              <li
                key={item}
                className="rounded-full border px-3 py-0.5 text-[length:var(--text-xs)]"
                style={{ borderColor: "var(--border)", color: "var(--text-muted)" }}
              >
                {item}
              </li>
            ))}
          </ul>
          {skillLabels.length ? (
            <>
              <SectionHeading>Wired skills</SectionHeading>
              <ul className="mt-2 flex flex-wrap gap-2" role="list">
                {skillLabels.map((label) => (
                  <li
                    key={label}
                    className="rounded-full border px-3 py-0.5 text-[length:var(--text-xs)]"
                    style={{
                      borderColor: "var(--accent-dim)",
                      color: "var(--text-muted)",
                    }}
                  >
                    {label}
                  </li>
                ))}
              </ul>
            </>
          ) : null}
        </Reveal>

        {/* ---- Links ---- */}
        {(project.repoUrl || project.externalUrl) && (
          <p className="mt-8">
            <a
              href={project.repoUrl ?? project.externalUrl}
              target="_blank"
              rel="noreferrer"
              className="underline decoration-[var(--accent-dim)] underline-offset-4 hover:decoration-[var(--accent)]"
            >
              {project.repoUrl ? "View source ↗" : "View ↗"}
            </a>
          </p>
        )}

        {/* ---- Next-project funnel ---- */}
        <div
          className="mt-10 border-t pt-6"
          style={{ borderColor: "var(--border)" }}
        >
          <button
            type="button"
            onClick={() =>
              setGraphState({
                selectedProjectId: nextFlagshipId(project.id),
              })
            }
            className="group flex w-full cursor-pointer items-baseline justify-between gap-4 text-left"
          >
            <span
              className="text-[length:var(--text-xs)] tracking-[var(--tracking-wide)] uppercase"
              style={{ color: "var(--text-faint)" }}
            >
              Next case study
            </span>
            <span
              className="font-medium underline-offset-4 group-hover:underline"
              style={{ color: "var(--accent-bright)" }}
            >
              {getProject(nextFlagshipId(project.id)).title} →
            </span>
          </button>
        </div>

        {!isFlagship ? (
          <p
            className="mt-4 text-[length:var(--text-xs)]"
            style={{ color: "var(--text-faint)" }}
          >
            Archive project — part of the 30+ experiment corpus behind the
            flagships.
          </p>
        ) : null}
        </div>
      </div>
    </>
  );
}
