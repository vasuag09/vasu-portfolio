import { describe, it, expect } from "vitest";
import { render } from "@testing-library/react";
import TierBadge from "../TierBadge";

describe("TierBadge", () => {
  it("renders TIER S with sparkle icon", () => {
    const { container } = render(<TierBadge tier="S" />);
    expect(container.textContent).toContain("TIER S");
    // S-tier has the Sparkles icon (an SVG)
    expect(container.querySelector("svg")).toBeTruthy();
  });

  it("renders TIER A without icon", () => {
    const { container } = render(<TierBadge tier="A" />);
    expect(container.textContent).toContain("TIER A");
    expect(container.querySelector("svg")).toBeNull();
  });

  it("renders TIER B without icon", () => {
    const { container } = render(<TierBadge tier="B" />);
    expect(container.textContent).toContain("TIER B");
    expect(container.querySelector("svg")).toBeNull();
  });

  it("returns null for unknown tier", () => {
    const { container } = render(<TierBadge tier="Z" />);
    expect(container.innerHTML).toBe("");
  });

  it("applies large size classes when size='lg'", () => {
    const { container } = render(<TierBadge tier="S" size="lg" />);
    const span = container.querySelector("span");
    expect(span.className).toContain("text-xs");
    expect(span.className).toContain("py-1");
  });

  it("applies small size classes by default", () => {
    const { container } = render(<TierBadge tier="A" />);
    const span = container.querySelector("span");
    expect(span.className).toContain("text-[10px]");
    expect(span.className).toContain("py-0.5");
  });

  it("applies glow class for S-tier", () => {
    const { container } = render(<TierBadge tier="S" />);
    const span = container.querySelector("span");
    expect(span.className).toContain("shadow-");
  });
});
