import type { ReactNode } from "react";
import type { SectionId } from "@/data/types";

interface SectionProps {
  id: SectionId;
  label: string;
  /** Chapter length in svh (AWARD-RESEARCH §4: tall chapters, camera rests mid-section). */
  heightSvh: number;
  children: ReactNode;
}

/**
 * A chapter region: real DOM, real text (ADR-4). The tall wrapper provides
 * scroll runway for the camera flight; content sits at the section's vertical
 * center, exactly where the camera rest-pose aligns.
 */
export function Section({ id, label, heightSvh, children }: SectionProps) {
  const headingId = `${id}-heading`;
  return (
    <section
      id={id}
      data-chapter={id}
      data-chapter-label={label}
      aria-labelledby={headingId}
      className="relative flex items-center"
      style={{ minHeight: `${heightSvh}svh` }}
    >
      <div className="mx-auto w-full max-w-3xl px-6">
        <p
          className="text-[length:var(--text-xs)] tracking-[var(--tracking-terminal)] uppercase"
          style={{ color: "var(--accent)" }}
        >
          {label}
        </p>
        <div id={headingId}>{children}</div>
      </div>
    </section>
  );
}
