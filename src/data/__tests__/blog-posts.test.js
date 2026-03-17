import { describe, it, expect } from "vitest";
import { engineeringLogs } from "../blog-posts";

describe("blog posts data", () => {
  it("has 5 engineering logs", () => {
    expect(engineeringLogs).toHaveLength(5);
  });

  it("every post has required fields", () => {
    engineeringLogs.forEach((post) => {
      expect(post.id).toBeTypeOf("number");
      expect(post.slug).toBeTypeOf("string");
      expect(post.title).toBeTypeOf("string");
      expect(post.date).toBeTypeOf("string");
      expect(post.tags).toBeInstanceOf(Array);
      expect(post.tags.length).toBeGreaterThan(0);
      expect(post.readTime).toBeTypeOf("string");
      expect(post.content).toBeTypeOf("string");
      expect(post.content.length).toBeGreaterThan(0);
    });
  });

  it("has unique ids", () => {
    const ids = engineeringLogs.map((p) => p.id);
    expect(new Set(ids).size).toBe(ids.length);
  });

  it("has unique slugs", () => {
    const slugs = engineeringLogs.map((p) => p.slug);
    expect(new Set(slugs).size).toBe(slugs.length);
  });

  it("slugs are URL-safe (lowercase, hyphens, no spaces)", () => {
    engineeringLogs.forEach((post) => {
      expect(post.slug).toMatch(/^[a-z0-9-]+$/);
    });
  });

  it("every post has markdown content with at least one heading", () => {
    engineeringLogs.forEach((post) => {
      expect(post.content).toContain("###");
    });
  });

  it("readTime includes 'min'", () => {
    engineeringLogs.forEach((post) => {
      expect(post.readTime).toMatch(/\d+ min/);
    });
  });
});
