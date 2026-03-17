import { describe, it, expect, afterEach } from "vitest";
import { renderHook } from "@testing-library/react";
import { useDocumentTitle } from "../useDocumentTitle";

describe("useDocumentTitle", () => {
  afterEach(() => {
    document.title = "";
  });

  it("sets document title with base suffix", () => {
    renderHook(() => useDocumentTitle("Projects"));
    expect(document.title).toBe("Projects | VASU_OS");
  });

  it("uses only base when title is empty", () => {
    renderHook(() => useDocumentTitle(""));
    expect(document.title).toBe("VASU_OS");
  });

  it("uses only base when title is null", () => {
    renderHook(() => useDocumentTitle(null));
    expect(document.title).toBe("VASU_OS");
  });

  it("updates when title changes", () => {
    const { rerender } = renderHook(({ title }) => useDocumentTitle(title), {
      initialProps: { title: "Projects" },
    });

    expect(document.title).toBe("Projects | VASU_OS");

    rerender({ title: "Blog" });
    expect(document.title).toBe("Blog | VASU_OS");
  });

  it("resets title on unmount", () => {
    const { unmount } = renderHook(() => useDocumentTitle("Projects"));
    expect(document.title).toBe("Projects | VASU_OS");

    unmount();
    expect(document.title).toBe("VASU_OS");
  });
});
