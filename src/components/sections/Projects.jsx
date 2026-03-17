import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Filter, Code2 } from "lucide-react";
import { getFilteredProjects } from "../../data/projects";
import { useDocumentTitle } from "../../hooks/useDocumentTitle";
import { useUI } from "../../hooks/useUI";
import ProjectCard from "./ProjectCard";

import SectionWrapper from "../layout/SectionWrapper";

export default function Projects() {
  useDocumentTitle("Deployments");
  const { setPreviewProject } = useUI();
  const [filterTech, setFilterTech] = useState(null);
  const navigate = useNavigate();

  const filteredProjects = getFilteredProjects(filterTech);

  return (
    <SectionWrapper id="projects" className="space-y-6">
      {filterTech && (
        <div className="flex items-center gap-4 bg-emerald-900/20 border border-emerald-500/30 p-3 rounded mb-4 text-sm">
          <Filter size={16} className="text-emerald-500" />
          <span className="text-slate-300">
            Filtering by:{" "}
            <span className="text-white font-bold">{filterTech}</span>
          </span>
          <button
            onClick={() => setFilterTech(null)}
            className="ml-auto text-xs bg-slate-800 hover:bg-slate-700 px-2 py-1 rounded text-slate-300 cursor-pointer"
          >
            Clear Filter
          </button>
        </div>
      )}
      <div className="grid md:grid-cols-2 xl:grid-cols-3 gap-6">
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
          <div className="col-span-full flex flex-col items-center justify-center py-12 text-slate-500">
            <Code2 size={48} className="mb-4 opacity-20" />
            <p>No projects found with tech stack: &quot;{filterTech}&quot;</p>
            <button
              onClick={() => setFilterTech(null)}
              className="mt-4 text-emerald-500 hover:underline cursor-pointer"
            >
              Clear Filters
            </button>
          </div>
        )}
      </div>
    </SectionWrapper>
  );
}
