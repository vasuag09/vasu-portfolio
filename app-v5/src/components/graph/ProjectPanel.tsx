"use client";

import { useEffect, useRef } from "react";
import { getProject, projects } from "@/data/projects-v5";
import { getSkillsForProject } from "@/lib/graph-adjacency";
import { getSkill } from "@/data/skills-graph";
import { setGraphState } from "@/lib/graph-store";
import { useGraphState } from "@/hooks/useGraphState";
import type { ProjectStatus } from "@/data/types";

/**
 * Click-to-expand project panel (Phase 3). Non-modal side dialog: real DOM,
 * Esc closes, focus moves in on open and back out on close. Deep-linkable
 * via ?project=<id> (URL-as-state).
 */

const STATUS_LABEL: Record<ProjectStatus, string> = {
  production: "In production",
  live: "Live",
  "under-review": "PR under review",
  active: "Active",
  archived: "Archived",
};

const KNOWN_IDS = new Set(projects.map((p) => p.id));

function syncUrl(projectId: string | null) {
  const url = new URL(window.location.href);
  if (projectId) url.searchParams.set("project", projectId);
  else url.searchParams.delete("project");
  window.history.replaceState(null, "", url);
}

export function ProjectPanel() {
  const { selectedProjectId } = useGraphState();
  const panelRef = useRef<HTMLElement>(null);
  const restoreFocusRef = useRef<HTMLElement | null>(null);

  // Deep link in: ?project=nm-gpt opens the panel on load.
  useEffect(() => {
    const requested = new URLSearchParams(window.location.search).get("project");
    if (requested && KNOWN_IDS.has(requested)) {
      setGraphState({ selectedProjectId: requested });
    }
  }, []);

  // URL out + focus management on open/close.
  useEffect(() => {
    syncUrl(selectedProjectId);
    if (selectedProjectId) {
      restoreFocusRef.current = document.activeElement as HTMLElement;
      panelRef.current?.focus();
    } else {
      restoreFocusRef.current?.focus();
      restoreFocusRef.current = null;
    }
  }, [selectedProjectId]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") setGraphState({ selectedProjectId: null });
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  if (!selectedProjectId) return null;
  const project = getProject(selectedProjectId);
  const skillLabels = getSkillsForProject(project.id).map(
    (id) => getSkill(id).label,
  );

  return (
    <aside
      ref={panelRef}
      role="dialog"
      aria-label={`${project.title} details`}
      tabIndex={-1}
      className="fixed inset-y-0 right-0 w-full max-w-md overflow-y-auto border-l p-6 outline-none md:p-8"
      style={{
        zIndex: "var(--z-overlay)",
        background: "var(--bg-elevated)",
        borderColor: "var(--border)",
      }}
    >
      <div className="flex items-start justify-between gap-4">
        <p
          className="text-[length:var(--text-xs)] tracking-[var(--tracking-terminal)] uppercase"
          style={{ color: "var(--accent)" }}
        >
          {STATUS_LABEL[project.status]}
        </p>
        <button
          type="button"
          onClick={() => setGraphState({ selectedProjectId: null })}
          aria-label="Close project details"
          className="cursor-pointer rounded border px-2 py-0.5 text-[length:var(--text-sm)] transition-colors duration-[var(--duration-fast)] hover:border-[var(--border-active)]"
          style={{ borderColor: "var(--border)", color: "var(--text-muted)" }}
        >
          esc
        </button>
      </div>

      <h2
        className="mt-3 font-bold leading-[var(--leading-tight)]"
        style={{ fontSize: "var(--text-lg)" }}
      >
        {project.title}
      </h2>
      <p className="mt-3" style={{ color: "var(--text-muted)" }}>
        {project.oneLiner}
      </p>

      {project.narrative ? (
        <p
          className="mt-4 text-[length:var(--text-sm)]"
          style={{ color: "var(--text-muted)" }}
        >
          {project.narrative}
        </p>
      ) : null}

      {project.contribution ? (
        <>
          <h3
            className="mt-6 text-[length:var(--text-xs)] tracking-[var(--tracking-wide)] uppercase"
            style={{ color: "var(--text-faint)" }}
          >
            My part
          </h3>
          <p
            className="mt-2 text-[length:var(--text-sm)]"
            style={{ color: "var(--text-muted)" }}
          >
            {project.contribution}
          </p>
        </>
      ) : null}

      {project.metrics?.length ? (
        <>
          <h3
            className="mt-6 text-[length:var(--text-xs)] tracking-[var(--tracking-wide)] uppercase"
            style={{ color: "var(--text-faint)" }}
          >
            Numbers
          </h3>
          <ul className="mt-2 flex flex-col gap-1" role="list">
            {project.metrics.map((metric) => (
              <li
                key={metric}
                className="text-[length:var(--text-sm)]"
                style={{ color: "var(--accent-bright)" }}
              >
                {metric}
              </li>
            ))}
          </ul>
        </>
      ) : null}

      <h3
        className="mt-6 text-[length:var(--text-xs)] tracking-[var(--tracking-wide)] uppercase"
        style={{ color: "var(--text-faint)" }}
      >
        Wired skills
      </h3>
      <ul className="mt-2 flex flex-wrap gap-2" role="list">
        {skillLabels.map((label) => (
          <li
            key={label}
            className="rounded-full border px-3 py-0.5 text-[length:var(--text-xs)]"
            style={{ borderColor: "var(--border)", color: "var(--text-muted)" }}
          >
            {label}
          </li>
        ))}
      </ul>

      {(project.repoUrl || project.externalUrl) && (
        <p className="mt-6">
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
    </aside>
  );
}
