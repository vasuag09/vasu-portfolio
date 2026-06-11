"use client";

import { useRef } from "react";
import { skills } from "@/data/skills-graph";
import { getProject } from "@/data/projects-v5";
import { getProjectsForSkill } from "@/lib/graph-adjacency";
import { setGraphState } from "@/lib/graph-store";
import { useGraphState } from "@/hooks/useGraphState";
import type { SkillCategory } from "@/data/types";

/**
 * HTML overlay for the skills region (Phase 3). Real DOM chips — the canvas
 * is pointer-inert (ADR-4), so this list IS the interaction surface:
 * hover/focus glows the connected projects in the scene, click pins a skill
 * and lists its projects, Enter on a project opens the case panel.
 */

const CATEGORY_LABEL: Record<SkillCategory, string> = {
  genai: "GenAI / LLM",
  ml: "AI / ML",
  fullstack: "Full-Stack",
  lang: "Languages",
  infra: "Infra & Tools",
};

const CATEGORY_ORDER: readonly SkillCategory[] = [
  "genai",
  "ml",
  "fullstack",
  "lang",
  "infra",
];

export function SkillsOverlay() {
  const { hoveredSkillId, selectedSkillId } = useGraphState();
  const listRef = useRef<HTMLDivElement>(null);
  const activeSkillId = hoveredSkillId ?? selectedSkillId;
  const pinnedProjects = selectedSkillId
    ? getProjectsForSkill(selectedSkillId)
    : [];

  // Arrow-key roving across all chips (Tab enters/leaves the widget).
  const onKeyDown = (event: React.KeyboardEvent) => {
    const keys = ["ArrowRight", "ArrowLeft", "ArrowDown", "ArrowUp", "Home", "End"];
    if (!keys.includes(event.key)) return;
    const chips = Array.from(
      listRef.current?.querySelectorAll<HTMLButtonElement>("[data-skill-chip]") ??
        [],
    );
    const current = chips.indexOf(document.activeElement as HTMLButtonElement);
    if (current === -1) return;
    event.preventDefault();
    const delta =
      event.key === "ArrowRight" || event.key === "ArrowDown" ? 1 : -1;
    const next =
      event.key === "Home"
        ? 0
        : event.key === "End"
          ? chips.length - 1
          : (current + delta + chips.length) % chips.length;
    chips[next]?.focus();
  };

  return (
    <div>
      <div ref={listRef} onKeyDown={onKeyDown} className="flex flex-col gap-5">
        {CATEGORY_ORDER.map((category) => (
          <div key={category}>
            <p
              className="text-[length:var(--text-xs)] tracking-[var(--tracking-wide)] uppercase"
              style={{ color: "var(--text-faint)" }}
            >
              {CATEGORY_LABEL[category]}
            </p>
            <ul className="mt-2 flex flex-wrap gap-2" role="list">
              {skills
                .filter((skill) => skill.category === category)
                .map((skill) => {
                  const isActive = skill.id === activeSkillId;
                  const isPinned = skill.id === selectedSkillId;
                  return (
                    <li key={skill.id}>
                      <button
                        type="button"
                        data-skill-chip
                        aria-pressed={isPinned}
                        onMouseEnter={() =>
                          setGraphState({ hoveredSkillId: skill.id })
                        }
                        onMouseLeave={() =>
                          setGraphState({ hoveredSkillId: null })
                        }
                        onFocus={() =>
                          setGraphState({ hoveredSkillId: skill.id })
                        }
                        onBlur={() => setGraphState({ hoveredSkillId: null })}
                        onClick={() =>
                          setGraphState({
                            selectedSkillId: isPinned ? null : skill.id,
                          })
                        }
                        className="cursor-pointer rounded-full border px-3 py-1 text-[length:var(--text-sm)] transition-colors duration-[var(--duration-fast)]"
                        style={{
                          borderColor: isActive
                            ? "var(--border-active)"
                            : "var(--border)",
                          color: isActive ? "var(--accent-bright)" : "var(--text-muted)",
                          background: isActive
                            ? "var(--bg-overlay)"
                            : "transparent",
                        }}
                      >
                        {skill.label}
                      </button>
                    </li>
                  );
                })}
            </ul>
          </div>
        ))}
      </div>

      {selectedSkillId ? (
        <div
          className="mt-6 border-l-2 pl-4"
          style={{ borderColor: "var(--accent-dim)" }}
        >
          <p
            className="text-[length:var(--text-xs)] tracking-[var(--tracking-wide)] uppercase"
            style={{ color: "var(--accent)" }}
          >
            wired into
          </p>
          <ul className="mt-2 flex flex-col gap-1" role="list">
            {pinnedProjects.map((projectId) => {
              const project = getProject(projectId);
              return (
                <li key={projectId}>
                  <button
                    type="button"
                    onClick={() =>
                      setGraphState({ selectedProjectId: projectId })
                    }
                    onMouseEnter={() =>
                      setGraphState({ hoveredProjectId: projectId })
                    }
                    onMouseLeave={() =>
                      setGraphState({ hoveredProjectId: null })
                    }
                    className="cursor-pointer underline decoration-[var(--accent-dim)] underline-offset-4 hover:decoration-[var(--accent)] text-[length:var(--text-sm)]"
                    style={{ color: "var(--text)" }}
                  >
                    {project.title}
                  </button>
                </li>
              );
            })}
          </ul>
        </div>
      ) : null}
    </div>
  );
}
