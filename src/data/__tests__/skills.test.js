import { describe, it, expect } from "vitest";
import { skills, certifications } from "../skills";

describe("skills data", () => {
  it("has 4 skill categories", () => {
    expect(skills).toHaveLength(4);
  });

  it("every category has a name and non-empty items array", () => {
    skills.forEach((s) => {
      expect(s.category).toBeTypeOf("string");
      expect(s.category.length).toBeGreaterThan(0);
      expect(s.items).toBeInstanceOf(Array);
      expect(s.items.length).toBeGreaterThan(0);
    });
  });

  it("all skill items are non-empty strings", () => {
    skills.forEach((s) => {
      s.items.forEach((item) => {
        expect(item).toBeTypeOf("string");
        expect(item.length).toBeGreaterThan(0);
      });
    });
  });

  it("has unique category names", () => {
    const categories = skills.map((s) => s.category);
    expect(new Set(categories).size).toBe(categories.length);
  });

  it("includes expected categories", () => {
    const categories = skills.map((s) => s.category);
    expect(categories).toContain("CORE ML / COMPUTER VISION");
    expect(categories).toContain("APPLIED ML ENGINEERING");
    expect(categories).toContain("FULL-STACK + ML SYSTEMS");
    expect(categories).toContain("GEN AI (FOUNDATIONAL)");
  });
});

describe("certifications data", () => {
  it("has 8 certifications", () => {
    expect(certifications).toHaveLength(4);
  });

  it("all certifications are non-empty strings", () => {
    certifications.forEach((cert) => {
      expect(cert).toBeTypeOf("string");
      expect(cert.length).toBeGreaterThan(0);
    });
  });

  it("has no duplicate certifications", () => {
    expect(new Set(certifications).size).toBe(certifications.length);
  });
});
