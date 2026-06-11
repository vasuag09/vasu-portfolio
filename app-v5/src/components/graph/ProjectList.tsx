"use client";

import { projects } from "@/data/projects-v5";
import { setGraphState } from "@/lib/graph-store";
import { useGraphState } from "@/hooks/useGraphState";

/**
 * Interactive flagship list for the projects region (Phase 3): hovering a
 * project glows its skill connections in the scene; click/Enter opens the
 * project panel.
 */

const flagships = projects.filter((p) => p.tier === "flagship");

export function ProjectList() {
  const { hoveredProjectId, selectedProjectId } = useGraphState();

  return (
    <ul className="mt-6 flex flex-col gap-3" role="list">
      {flagships.map((project) => {
        const isActive =
          project.id === hoveredProjectId || project.id === selectedProjectId;
        return (
          <li key={project.id}>
            <button
              type="button"
              onClick={() => setGraphState({ selectedProjectId: project.id })}
              onMouseEnter={() =>
                setGraphState({ hoveredProjectId: project.id })
              }
              onMouseLeave={() => setGraphState({ hoveredProjectId: null })}
              onFocus={() => setGraphState({ hoveredProjectId: project.id })}
              onBlur={() => setGraphState({ hoveredProjectId: null })}
              aria-haspopup="dialog"
              className="group block w-full cursor-pointer border-l-2 pl-4 text-left transition-colors duration-[var(--duration-fast)]"
              style={{
                borderColor: isActive ? "var(--accent)" : "var(--border)",
              }}
            >
              <span
                className="font-medium"
                style={{
                  fontSize: "var(--text-base)",
                  color: isActive ? "var(--accent-bright)" : "var(--text)",
                }}
              >
                {project.title}
              </span>
              <span
                className="mt-1 block max-w-xl"
                style={{
                  color: "var(--text-muted)",
                  fontSize: "var(--text-sm)",
                }}
              >
                {project.oneLiner}
              </span>
            </button>
          </li>
        );
      })}
    </ul>
  );
}
