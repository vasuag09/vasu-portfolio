import { describe, expect, it } from "vitest";
import { veoSources, veoPoster } from "../veo-sources";

describe("veoSources (clip naming + codec negotiation)", () => {
  it("orders sources best-compression-first: AV1 → HEVC → H.264", () => {
    const sources = veoSources("fundlymart");
    expect(sources.map((s) => s.src)).toEqual([
      "/veo/fundlymart.av1.mp4",
      "/veo/fundlymart.hevc.mp4",
      "/veo/fundlymart.h264.mp4",
    ]);
  });

  it("declares codecs so browsers can skip what they cannot play", () => {
    const sources = veoSources("nm-gpt");
    expect(sources[0].type).toContain("av01");
    expect(sources[1].type).toContain("hvc1"); // Safari needs the hvc1 tag
    expect(sources[2].type).toContain("avc1");
  });

  it("poster path matches the VEO-BRIEF naming convention", () => {
    expect(veoPoster("geovision")).toBe("/veo/geovision-poster.avif");
  });
});
