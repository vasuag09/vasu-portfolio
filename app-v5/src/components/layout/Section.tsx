import type { ReactNode } from "react";
import { Reveal } from "@/components/ui/Reveal";
import type { SectionId } from "@/data/types";

/**
 * Editorial composition per chapter (design elevation P0.2): three distinct
 * variants instead of one centered column, an oversized numeral, a display-
 * scale title, and a 3-stage reveal stagger (label/numeral → title →
 * content). Hero stays bespoke in page.tsx (custom children, no title).
 */
type SectionVariant = "center" | "split" | "end";

interface SectionProps {
  id: SectionId;
  label: string;
  /** Chapter length in svh (AWARD-RESEARCH §4: tall chapters, camera rests mid-section). */
  heightSvh: number;
  /** 1-based chapter number, rendered as the 01–05 numeral. */
  index: number;
  /** Display-scale chapter title. Omit for fully bespoke chapters (hero). */
  title?: string;
  /**
   * Composition: "split" = title block left / content right (editorial),
   * "end" = title pushed right, content full width, "center" = stacked.
   */
  variant?: SectionVariant;
  /**
   * "start" pins content into the FIRST viewport of the chapter (hero: the
   * name must be visible at scroll 0, not buried mid-runway). Default
   * centers content at the camera rest-pose.
   */
  align?: "start" | "center";
  children: ReactNode;
}

const STAGGER = { header: 0, title: 110, content: 230 } as const;

function Numeral({ index }: { index: number }) {
  return (
    <span aria-hidden="true" className="chapter-numeral block">
      {String(index).padStart(2, "0")}
    </span>
  );
}

function Label({ label }: { label: string }) {
  return (
    <p
      className="text-[length:var(--text-xs)] tracking-[var(--tracking-terminal)] uppercase"
      style={{ color: "var(--accent)" }}
    >
      {label}
    </p>
  );
}

export function Section({
  id,
  label,
  heightSvh,
  index,
  title,
  variant = "center",
  align = "center",
  children,
}: SectionProps) {
  const headingId = `${id}-heading`;
  return (
    <section
      id={id}
      data-chapter={id}
      data-chapter-label={label}
      data-align={align}
      aria-labelledby={headingId}
      className={`relative flex ${align === "start" ? "items-start" : "items-center"}`}
      style={{
        minHeight: `${heightSvh}svh`,
        paddingTop: align === "start" ? "max(24svh, 7rem)" : undefined,
      }}
    >
      <div className="relative mx-auto w-full max-w-6xl px-6 md:px-10">
        <div
          aria-hidden="true"
          className="absolute -inset-x-12 -inset-y-14 -z-10"
          style={{
            background:
              "radial-gradient(110% 100% at 35% 50%, oklch(8% 0.02 250 / 0.92) 0%, oklch(8% 0.02 250 / 0.55) 55%, transparent 80%)",
          }}
        />

        {title === undefined ? (
          /* Bespoke chapter (hero): page.tsx owns the whole composition. */
          <div id={headingId}>{children}</div>
        ) : variant === "split" ? (
          <div>
            <Reveal delay={STAGGER.header} className="flex items-end gap-6">
              <Numeral index={index} />
              <div className="pb-2">
                <Label label={label} />
              </div>
            </Reveal>
            <Reveal delay={STAGGER.title}>
              <h2
                id={headingId}
                className="mt-3 font-bold leading-[var(--leading-tight)]"
                style={{ fontSize: "var(--text-display)" }}
              >
                {title}
              </h2>
            </Reveal>
            {/* Editorial offset: content sits right-of-center, the title's
                left rag stays empty — asymmetry without collision. */}
            <div className="md:grid md:grid-cols-12">
              <Reveal
                delay={STAGGER.content}
                className="mt-8 md:col-span-7 md:col-start-6"
              >
                {children}
              </Reveal>
            </div>
          </div>
        ) : variant === "end" ? (
          <div>
            <Reveal delay={STAGGER.header} className="flex items-end justify-between gap-6">
              <Numeral index={index} />
              <Label label={label} />
            </Reveal>
            <Reveal delay={STAGGER.title}>
              <h2
                id={headingId}
                className="mt-3 text-right font-bold leading-[var(--leading-tight)]"
                style={{ fontSize: "var(--text-display)" }}
              >
                {title}
              </h2>
            </Reveal>
            <Reveal delay={STAGGER.content}>{children}</Reveal>
          </div>
        ) : (
          <div className="mx-auto max-w-3xl">
            <Reveal delay={STAGGER.header} className="flex items-end gap-6">
              <Numeral index={index} />
              <div className="pb-2">
                <Label label={label} />
              </div>
            </Reveal>
            <Reveal delay={STAGGER.title}>
              <h2
                id={headingId}
                className="mt-3 font-bold leading-[var(--leading-tight)]"
                style={{ fontSize: "var(--text-display)" }}
              >
                {title}
              </h2>
            </Reveal>
            <Reveal delay={STAGGER.content}>{children}</Reveal>
          </div>
        )}
      </div>
    </section>
  );
}
