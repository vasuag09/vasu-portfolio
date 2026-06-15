import { XMLParser } from "fast-xml-parser";

import {
  mapRawItem,
  rankPosts,
  type RawFeedItem,
  type WritingPost,
  type WritingSource,
} from "./writing-feed";

/**
 * Writing feed fetch (Signal chapter) — server-only. Pulls Medium + Substack
 * RSS, parses, and hands the raw items to the pure ranker in writing-feed.ts.
 * Fail-soft (ADR mirror of /api/synapse): one dead feed → render the other;
 * both dead → empty list, the section hides itself. Never throws to the page.
 */

export const FEEDS: { url: string; source: WritingSource }[] = [
  { url: "https://medium.com/feed/@vasuagrawal1040", source: "Medium" },
  { url: "https://vasuagrawal.substack.com/feed", source: "Substack" },
];

export const LINKEDIN_URL = "https://www.linkedin.com/in/vasu-agrawal20/";

/** ISR window — the section self-refreshes ~hourly with no redeploy. */
const REVALIDATE_SECONDS = 3600;

const parser = new XMLParser({
  ignoreAttributes: true,
  trimValues: true,
});

function itemsFrom(xml: string): RawFeedItem[] {
  const parsed = parser.parse(xml) as {
    rss?: { channel?: { item?: RawFeedItem | RawFeedItem[] } };
  };
  const item = parsed.rss?.channel?.item;
  if (!item) return [];
  return Array.isArray(item) ? item : [item];
}

async function fetchFeed(url: string, source: WritingSource): Promise<WritingPost[]> {
  const res = await fetch(url, {
    headers: { "User-Agent": "vasu-portfolio/1.0 (+https://github.com/vasuag09)" },
    next: { revalidate: REVALIDATE_SECONDS },
  });
  if (!res.ok) throw new Error(`${source} feed ${res.status}`);
  const xml = await res.text();
  return itemsFrom(xml)
    .map((item) => mapRawItem(item, source))
    .filter((post): post is WritingPost => post !== null);
}

/**
 * Latest posts across both feeds, deduped + newest-first. Returns [] if every
 * feed fails — the caller renders nothing rather than an error.
 */
export async function fetchWritingPosts(limit = 5): Promise<WritingPost[]> {
  const results = await Promise.allSettled(
    FEEDS.map(({ url, source }) => fetchFeed(url, source)),
  );
  const posts = results.flatMap((r) =>
    r.status === "fulfilled" ? r.value : [],
  );
  return rankPosts(posts, limit);
}
