/**
 * Writing feed (Signal chapter) — PURE logic only. The RSS fetch + XML parse
 * live in writing-fetch.ts; this module maps an already-parsed item shape to
 * a WritingPost and ranks the merged set. Same pure/impure split as
 * cursor-state.ts vs CustomCursor. Vasu cross-posts between Medium and
 * Substack, so dedupe by title is load-bearing, not cosmetic.
 */

export type WritingSource = "Medium" | "Substack";

export interface WritingPost {
  title: string;
  url: string;
  source: WritingSource;
  /** ISO 8601 — sortable, locale-stable. */
  publishedAt: string;
  excerpt: string;
}

/** Per-<item> shape after XML parsing (fields optional — feeds vary). */
export interface RawFeedItem {
  title?: string;
  link?: string;
  pubDate?: string;
  description?: string;
  "content:encoded"?: string;
}

/** Normalized key for dedupe: case/space/punctuation-insensitive. */
export function normalizeTitle(title: string): string {
  return title
    .toLowerCase()
    .replace(/[\p{P}\p{S}]/gu, "") // drop punctuation + symbols
    .replace(/\s+/g, " ")
    .trim();
}

const ENTITIES: Record<string, string> = {
  "&amp;": "&",
  "&lt;": "<",
  "&gt;": ">",
  "&quot;": '"',
  "&#39;": "'",
  "&apos;": "'",
  "&nbsp;": " ",
};

/** Strip HTML to plain text, decode common entities, truncate on a word edge. */
export function excerptFromHtml(html: string, limit = 180): string {
  const text = html
    .replace(/<[^>]*>/g, " ")
    .replace(/&#(\d+);/g, (_, n) => String.fromCodePoint(Number(n)))
    .replace(/&[a-z]+;/gi, (m) => ENTITIES[m.toLowerCase()] ?? m)
    .replace(/\s+/g, " ")
    .trim();
  if (text.length <= limit) return text;
  const cut = text.slice(0, limit);
  const lastSpace = cut.lastIndexOf(" ");
  return `${cut.slice(0, lastSpace > 0 ? lastSpace : limit)}…`;
}

/** Map one parsed feed item to a WritingPost, or null if it lacks essentials. */
export function mapRawItem(
  item: RawFeedItem,
  source: WritingSource,
): WritingPost | null {
  const title = item.title?.trim();
  const url = item.link?.trim();
  if (!title || !url) return null;

  const date = item.pubDate ? new Date(item.pubDate) : null;
  const publishedAt =
    date && !Number.isNaN(date.getTime())
      ? date.toISOString()
      : new Date(0).toISOString();

  const body = item["content:encoded"] ?? item.description ?? "";
  return { title, url, source, publishedAt, excerpt: excerptFromHtml(body) };
}

/**
 * Merge → dedupe by normalized title (newer copy wins; Medium breaks a tie) →
 * sort newest-first → take `limit`.
 */
export function rankPosts(posts: WritingPost[], limit: number): WritingPost[] {
  const byKey = new Map<string, WritingPost>();
  for (const post of posts) {
    const key = normalizeTitle(post.title);
    const existing = byKey.get(key);
    if (!existing || isPreferred(post, existing)) byKey.set(key, post);
  }
  return Array.from(byKey.values())
    .sort((a, b) => b.publishedAt.localeCompare(a.publishedAt))
    .slice(0, limit);
}

/** A is preferred over B when it is newer, or same-day and from Medium. */
function isPreferred(a: WritingPost, b: WritingPost): boolean {
  if (a.publishedAt !== b.publishedAt) return a.publishedAt > b.publishedAt;
  return a.source === "Medium" && b.source !== "Medium";
}
