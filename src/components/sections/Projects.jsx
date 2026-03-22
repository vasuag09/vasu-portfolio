import React from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { Filter, Code2 } from "lucide-react";
import { getFilteredProjects } from "../../data/projects";
import { useDocumentTitle } from "../../hooks/useDocumentTitle";
import { useUI } from "../../hooks/useUI";
import ProjectCard from "./ProjectCard";
import ScrollReveal from "../effects/ScrollReveal";

export default function Projects() {
  useDocumentTitle("Trained Models");
  const { setPreviewProject } = useUI();
  const [searchParams, setSearchParams] = useSearchParams();
  const navigate = useNavigate();

  const filterTech = searchParams.get("tech") || null;

  const setFilterTech = (tech) => {
    if (tech) {
      setSearchParams({ tech });
    } else {
      setSearchParams({});
    }
  };

  const filteredProjects = getFilteredProjects(filterTech);

  return (
    <section id="projects" className="space-y-6 pb-20 md:pb-8">
      {filterTech && (
        <div className="flex items-center gap-4 glass-card-static p-3 text-sm">
          <Filter size={16} className="text-cyan-500" />
          <span className="text-slate-300">
            Filtering by:{" "}
            <span className="text-white font-semibold">{filterTech}</span>
          </span>
          <button
            onClick={() => setFilterTech(null)}
            className="ml-auto text-xs bg-[rgba(0,212,255,0.06)] hover:bg-[rgba(0,212,255,0.12)] px-3 py-1 rounded-lg text-cyan-400/70 hover:text-cyan-300 cursor-pointer font-mono transition-colors"
          >
            Clear
          </button>
        </div>
      )}

      {/* Bento grid — S-tier cards span 2 columns */}
      <div className="grid md:grid-cols-2 xl:grid-cols-3 gap-4 md:gap-5 auto-rows-auto">
        {filteredProjects.length > 0 ? (
          filteredProjects.map((project, index) => (
            <ProjectCard
              key={project.id}
              project={project}
              index={index}
              filterTech={filterTech}
              onFilterTech={setFilterTech}
              onExpand={(p) => navigate(`/projects/${p.alias}`)}
              onPreview={(p) => setPreviewProject(p)}
            />
          ))
        ) : (
          <div className="col-span-full flex flex-col items-center justify-center py-20 text-slate-500">
            <Code2 size={48} className="mb-4 opacity-15" />
            <p className="font-mono text-sm">
              No models found matching: &quot;{filterTech}&quot;
            </p>
            <button
              onClick={() => setFilterTech(null)}
              className="mt-4 text-cyan-500 hover:text-cyan-400 font-mono text-sm cursor-pointer transition-colors"
            >
              Clear Filter
            </button>
          </div>
        )}
      </div>
    </section>
  );
}
