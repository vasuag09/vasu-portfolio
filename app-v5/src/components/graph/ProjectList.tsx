"use client";

import { useState } from "react";
import { projects } from "@/data/projects-v5";
import { setGraphState } from "@/lib/graph-store";
import { useGraphState } from "@/hooks/useGraphState";
import { veoPoster, veoSources } from "@/lib/veo-sources";

/**
 * Interactive flagship list for the projects region (Phase 3): hovering a
 * project glows its skill connections in the scene; click/Enter opens the
 * project panel.
 *
 * Design elevation P1.8: flagships with a Veo clip surface their poster on
 * the journey itself — hover swaps in the (already ≤4MB, lazy) loop. The
 * most expensive pixels on the site are no longer two clicks deep.
 */

const flagships = projects.filter((p) => p.tier === "flagship");

function VeoTeaser({ clip, active }: { clip: string; active: boolean }) {
  // Mount the <video> only after the first hover — zero cost until then.
  const [wanted, setWanted] = useState(false);
  if (active && !wanted) setWanted(true);

  return (
    <span
      aria-hidden="true"
      className="relative hidden shrink-0 overflow-hidden rounded border md:block"
      style={{
        width: "9.5rem",
        aspectRatio: "16 / 9",
        borderColor: active ? "var(--border-active)" : "var(--border)",
        transition: "border-color var(--duration-fast) linear",
      }}
    >
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={veoPoster(clip)}
        alt=""
        loading="lazy"
        className="h-full w-full object-cover"
      />
      {wanted ? (
        <video
          muted
          loop
          playsInline
          autoPlay
          preload="none"
          className="absolute inset-0 h-full w-full object-cover transition-opacity duration-[var(--duration-normal)]"
          style={{ opacity: active ? 1 : 0 }}
        >
          {veoSources(clip).map((s) => (
            <source key={s.src} src={s.src} type={s.type} />
          ))}
        </video>
      ) : null}
    </span>
  );
}

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
              className="group flex w-full cursor-pointer items-center gap-5 border-l-2 pl-4 text-left transition-colors duration-[var(--duration-fast)]"
              style={{
                borderColor: isActive ? "var(--accent)" : "var(--border)",
              }}
            >
              <span className="min-w-0 flex-1">
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
              </span>
              {project.clip ? (
                <VeoTeaser clip={project.clip} active={isActive} />
              ) : null}
            </button>
          </li>
        );
      })}
    </ul>
  );
}
