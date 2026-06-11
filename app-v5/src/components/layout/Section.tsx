import type { ReactNode } from "react";
import { Reveal } from "@/components/ui/Reveal";
import type { SectionId } from "@/data/types";

interface SectionProps {
  id: SectionId;
  label: string;
  /** Chapter length in svh (AWARD-RESEARCH §4: tall chapters, camera rests mid-section). */
  heightSvh: number;
  /**
   * "start" pins content into the FIRST viewport of the chapter (hero: the
   * name must be visible at scroll 0, not buried mid-runway). Default
   * centers content at the camera rest-pose.
   */
  align?: "start" | "center";
  children: ReactNode;
}

/**
 * A chapter region: real DOM, real text (ADR-4). The tall wrapper provides
 * scroll runway for the camera flight; content sits at the section's vertical
 * center, exactly where the camera rest-pose aligns.
 *
 * Phase 5: a gradient scrim sits behind the content so text reads over the
 * busy scene (pure gradient — backdrop blur over WebGL is a frame-time tax),
 * and the content block reveals once on first scroll-in.
 */
export function Section({
  id,
  label,
  heightSvh,
  align = "center",
  children,
}: SectionProps) {
  const headingId = `${id}-heading`;
  return (
    <section
      id={id}
      data-chapter={id}
      data-chapter-label={label}
      aria-labelledby={headingId}
      className={`relative flex ${align === "start" ? "items-start" : "items-center"}`}
      style={{
        minHeight: `${heightSvh}svh`,
        paddingTop: align === "start" ? "max(24svh, 7rem)" : undefined,
      }}
    >
      <div className="relative mx-auto w-full max-w-3xl px-6">
        <div
          aria-hidden="true"
          className="absolute -inset-x-10 -inset-y-12 -z-10"
          style={{
            background:
              "radial-gradient(110% 100% at 35% 50%, oklch(8% 0.02 250 / 0.92) 0%, oklch(8% 0.02 250 / 0.55) 55%, transparent 80%)",
          }}
        />
        <Reveal>
          <p
            className="text-[length:var(--text-xs)] tracking-[var(--tracking-terminal)] uppercase"
            style={{ color: "var(--accent)" }}
          >
            {label}
          </p>
          <div id={headingId}>{children}</div>
        </Reveal>
      </div>
    </section>
  );
}
