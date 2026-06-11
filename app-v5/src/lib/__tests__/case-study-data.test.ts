import { describe, expect, it } from "vitest";
import { existsSync } from "node:fs";
import { join } from "node:path";
import { projects } from "@/data/projects-v5";

const flagships = projects.filter((p) => p.tier === "flagship");
const VEO_DIR = join(__dirname, "../../../public/veo");

describe("case-study data contract (Phase 5)", () => {
  it("every flagship tells the full story: problem, narrative, contribution", () => {
    flagships.forEach((p) => {
      expect(p.problem, `${p.id} missing problem`).toBeTruthy();
      expect(p.narrative, `${p.id} missing narrative`).toBeTruthy();
      expect(p.contribution, `${p.id} missing contribution`).toBeTruthy();
    });
  });

  it("copy is metric-driven: every flagship has at least two metrics", () => {
    flagships.forEach((p) => {
      expect(p.metrics?.length ?? 0, `${p.id} needs >=2 metrics`).toBeGreaterThanOrEqual(2);
    });
  });

  it("the three locked Veo winners are wired to their flagships", () => {
    const withClips = flagships.filter((p) => p.clip);
    expect(withClips.map((p) => p.id).sort()).toEqual([
      "fundlymart",
      "geovision",
      "nm-gpt",
    ]);
  });

  it("every referenced clip actually exists in public/veo (all codecs + poster)", () => {
    projects
      .filter((p) => p.clip)
      .forEach((p) => {
        for (const suffix of [".av1.mp4", ".hevc.mp4", ".h264.mp4", "-poster.avif"]) {
          const file = join(VEO_DIR, `${p.clip}${suffix}`);
          expect(existsSync(file), `missing ${file}`).toBe(true);
        }
      });
  });

  it("banned LinkedIn-fluff words never appear in any copy (voice rules §7)", () => {
    const banned = /excited to announce|grateful for|humbled by|leverage|synergy|game-changer/i;
    projects.forEach((p) => {
      const text = [p.oneLiner, p.problem, p.narrative, p.contribution].join(" ");
      expect(banned.test(text), `${p.id} contains banned fluff`).toBe(false);
    });
  });
});
