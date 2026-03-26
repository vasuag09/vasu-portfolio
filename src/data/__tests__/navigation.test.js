import { describe, it, expect } from "vitest";
import {
  NAVIGATION_ITEMS,
  getActiveLayer,
  getActiveItem,
  SHORTCUT_ROUTES,
} from "../navigation";

describe("navigation data", () => {
  it("has at least 5 navigation items", () => {
    expect(NAVIGATION_ITEMS.length).toBeGreaterThanOrEqual(5);
  });

  it("every item has required fields", () => {
    NAVIGATION_ITEMS.forEach((item) => {
      expect(item.path).toBeTruthy();
      expect(item.label).toBeTruthy();
      expect(item.shortLabel).toBeTruthy();
      expect(item.layerTag).toBeTruthy();
      expect(item.layerTitle).toBeTruthy();
      expect(item.shortcut).toBeTruthy();
      expect(item.icon).toBeTruthy();
    });
  });

  it("has unique paths", () => {
    const paths = NAVIGATION_ITEMS.map((i) => i.path);
    expect(new Set(paths).size).toBe(paths.length);
  });

  it("has unique shortcuts", () => {
    const shortcuts = NAVIGATION_ITEMS.map((i) => i.shortcut);
    expect(new Set(shortcuts).size).toBe(shortcuts.length);
  });

  it("first item is root path", () => {
    expect(NAVIGATION_ITEMS[0].path).toBe("/");
  });
});

describe("getActiveLayer", () => {
  it("returns 0 for root path", () => {
    expect(getActiveLayer("/")).toBe(0);
  });

  it("returns correct index for /projects", () => {
    const idx = NAVIGATION_ITEMS.findIndex((i) => i.path === "/projects");
    expect(getActiveLayer("/projects")).toBe(idx);
  });

  it("matches sub-paths correctly", () => {
    const idx = NAVIGATION_ITEMS.findIndex((i) => i.path === "/projects");
    expect(getActiveLayer("/projects/some-alias")).toBe(idx);
  });

  it("returns -1 for unknown paths", () => {
    expect(getActiveLayer("/unknown-route")).toBe(-1);
  });
});

describe("getActiveItem", () => {
  it("returns first item for root path", () => {
    const item = getActiveItem("/");
    expect(item.path).toBe("/");
  });

  it("returns first item as fallback for unknown paths", () => {
    const item = getActiveItem("/nonexistent");
    expect(item.path).toBe("/");
  });
});

describe("SHORTCUT_ROUTES", () => {
  it("maps shortcut keys to paths", () => {
    NAVIGATION_ITEMS.forEach((item) => {
      expect(SHORTCUT_ROUTES[item.shortcut]).toBe(item.path);
    });
  });
});
