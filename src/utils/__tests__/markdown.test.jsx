import { describe, it, expect } from "vitest";
import { render } from "@testing-library/react";
import { renderMarkdown } from "../markdown";

/** Helper: renders markdown into a container and returns the container element */
function renderInto(markdown) {
  const { container } = render(<div>{renderMarkdown(markdown)}</div>);
  return container;
}

describe("renderMarkdown", () => {
  it("renders ### headings as h3 elements", () => {
    const container = renderInto("### My Heading");
    const h3 = container.querySelector("h3");
    expect(h3).toBeTruthy();
    expect(h3.textContent).toContain("My Heading");
  });

  it("renders plain text as paragraphs", () => {
    const container = renderInto("Hello world");
    const p = container.querySelector("p");
    expect(p).toBeTruthy();
    expect(p.textContent).toBe("Hello world");
  });

  it("renders **bold** text as <strong>", () => {
    const container = renderInto("This is **important** text");
    const strong = container.querySelector("strong");
    expect(strong).toBeTruthy();
    expect(strong.textContent).toBe("important");
  });

  it("renders `code` as <code>", () => {
    const container = renderInto("Use `npm install` to install");
    const code = container.querySelector("code");
    expect(code).toBeTruthy();
    expect(code.textContent).toBe("npm install");
  });

  it("renders [text](url) as <a> links", () => {
    const container = renderInto(
      "Visit [Google](https://google.com) today",
    );
    const link = container.querySelector("a");
    expect(link).toBeTruthy();
    expect(link.textContent).toBe("Google");
    expect(link.getAttribute("href")).toBe("https://google.com");
    expect(link.getAttribute("target")).toBe("_blank");
    expect(link.getAttribute("rel")).toBe("noreferrer");
  });

  it("renders numbered lists with border-left styling", () => {
    const container = renderInto("1. First item\n2. Second item");
    const items = container.querySelectorAll(".border-l");
    expect(items.length).toBe(2);
  });

  it("skips empty lines (returns null)", () => {
    const container = renderInto("Line one\n\nLine two");
    const paragraphs = container.querySelectorAll("p");
    expect(paragraphs.length).toBe(2);
  });

  it("handles multiple inline formats in one line", () => {
    const container = renderInto(
      "Use **bold** and `code` together",
    );
    expect(container.querySelector("strong")).toBeTruthy();
    expect(container.querySelector("code")).toBeTruthy();
  });

  it("renders multi-line content correctly", () => {
    const md = "### Title\nSome body text\n1. A list item";
    const container = renderInto(md);
    expect(container.querySelector("h3")).toBeTruthy();
    expect(container.querySelector("p")).toBeTruthy();
    expect(container.querySelector(".border-l")).toBeTruthy();
  });
});
