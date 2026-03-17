import { describe, it, expect } from "vitest";
import { render } from "@testing-library/react";
import StatusBadge from "../StatusBadge";

describe("StatusBadge", () => {
  it("renders the status text", () => {
    const { container } = render(<StatusBadge status="LIVE" />);
    expect(container.textContent).toContain("LIVE");
  });

  it("renders BUILDING with a spinner icon", () => {
    const { container } = render(<StatusBadge status="BUILDING" />);
    expect(container.textContent).toContain("BUILDING");
    const svg = container.querySelector("svg");
    expect(svg).toBeTruthy();
    expect(svg.className.baseVal || svg.getAttribute("class")).toContain(
      "animate-spin",
    );
  });

  it("renders LIVE without a spinner icon", () => {
    const { container } = render(<StatusBadge status="LIVE" />);
    expect(container.querySelector("svg")).toBeNull();
  });

  it("renders RESEARCH status", () => {
    const { container } = render(<StatusBadge status="RESEARCH" />);
    expect(container.textContent).toContain("RESEARCH");
    const span = container.querySelector("span");
    expect(span.className).toContain("blue");
  });

  it("renders CODE status", () => {
    const { container } = render(<StatusBadge status="CODE" />);
    expect(container.textContent).toContain("CODE");
    const span = container.querySelector("span");
    expect(span.className).toContain("purple");
  });

  it("falls back to CODE styling for unknown status", () => {
    const { container } = render(<StatusBadge status="UNKNOWN" />);
    expect(container.textContent).toContain("UNKNOWN");
    const span = container.querySelector("span");
    expect(span.className).toContain("purple");
  });
});
