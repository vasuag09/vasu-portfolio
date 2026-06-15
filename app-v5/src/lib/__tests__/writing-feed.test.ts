import { describe, expect, it } from "vitest";

import {
  excerptFromHtml,
  mapRawItem,
  normalizeTitle,
  rankPosts,
  type WritingPost,
} from "../writing-feed";

describe("normalizeTitle", () => {
  it("lowercases, trims, and collapses whitespace", () => {
    expect(normalizeTitle("  The  Model  ")).toBe("the model");
  });

  it("ignores trailing punctuation and case for dedupe equality", () => {
    expect(normalizeTitle("RAG Didn’t Solve Hallucinations.")).toBe(
      normalizeTitle("rag didn’t solve hallucinations"),
    );
  });
});

describe("excerptFromHtml", () => {
  it("strips tags and collapses whitespace", () => {
    expect(excerptFromHtml("<p>Hello <b>world</b></p>\n<p>again</p>")).toBe(
      "Hello world again",
    );
  });

  it("decodes the common named/numeric entities", () => {
    expect(excerptFromHtml("a &amp; b &#39;c&#39; &lt;d&gt;")).toBe("a & b 'c' <d>");
  });

  it("truncates to the limit on a word boundary with an ellipsis", () => {
    const long = "word ".repeat(80).trim();
    const out = excerptFromHtml(long, 50);
    expect(out.length).toBeLessThanOrEqual(51); // +1 for the ellipsis char
    expect(out.endsWith("…")).toBe(true);
  });
});

describe("mapRawItem", () => {
  it("maps a Medium-shaped item to a WritingPost", () => {
    const post = mapRawItem(
      {
        title: "The Model That Routes Around Itself",
        link: "https://medium.com/p/abc",
        pubDate: "Thu, 11 Jun 2026 22:01:01 GMT",
        "content:encoded": "<p>Routing is the new architecture.</p>",
      },
      "Medium",
    );
    expect(post).not.toBeNull();
    expect(post?.title).toBe("The Model That Routes Around Itself");
    expect(post?.url).toBe("https://medium.com/p/abc");
    expect(post?.source).toBe("Medium");
    expect(post?.publishedAt).toBe(new Date("Thu, 11 Jun 2026 22:01:01 GMT").toISOString());
    expect(post?.excerpt).toContain("Routing is the new architecture");
  });

  it("returns null when the essential fields are missing", () => {
    expect(mapRawItem({ link: "https://x.com" }, "Medium")).toBeNull();
    expect(mapRawItem({ title: "No link" }, "Substack")).toBeNull();
  });
});

const post = (over: Partial<WritingPost>): WritingPost => ({
  title: "T",
  url: "https://u",
  source: "Medium",
  publishedAt: "2026-06-01T00:00:00.000Z",
  excerpt: "",
  ...over,
});

describe("rankPosts", () => {
  it("sorts newest first and respects the limit", () => {
    const out = rankPosts(
      [
        post({ title: "A", publishedAt: "2026-06-01T00:00:00.000Z", url: "a" }),
        post({ title: "B", publishedAt: "2026-06-10T00:00:00.000Z", url: "b" }),
        post({ title: "C", publishedAt: "2026-06-05T00:00:00.000Z", url: "c" }),
      ],
      2,
    );
    expect(out.map((p) => p.title)).toEqual(["B", "C"]);
  });

  it("dedupes a cross-posted title, keeping the newer copy", () => {
    const out = rankPosts(
      [
        post({
          title: "I Caught My LLM Agent Lying Mid-Tool-Call",
          source: "Substack",
          publishedAt: "2026-06-06T10:40:37.000Z",
          url: "sub",
        }),
        post({
          title: "I Caught My LLM Agent Lying Mid-Tool-Call.",
          source: "Medium",
          publishedAt: "2026-06-07T13:01:02.000Z",
          url: "med",
        }),
      ],
      5,
    );
    expect(out).toHaveLength(1);
    expect(out[0].source).toBe("Medium"); // newer copy wins
  });
});
