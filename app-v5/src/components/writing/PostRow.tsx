import type { WritingPost } from "@/lib/writing-feed";

/**
 * One writing entry as an editorial row (not a boxed card): a meta line
 * (source · date), a display-scale title that is the link, and a muted
 * excerpt. The whole row is the hit target; hover lifts the title to accent
 * and slides the arrow — an intentional state, per the design bar.
 */

function formatDate(iso: string): string {
  return new Date(iso).toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
}

export function PostRow({ post }: { post: WritingPost }) {
  return (
    <li className="group border-t border-[var(--hairline,oklch(100%_0_0/0.1))]">
      <a
        href={post.url}
        target="_blank"
        rel="noopener noreferrer"
        className="block py-6 transition-colors duration-[var(--duration-normal)] ease-[var(--ease-out-expo)]"
        data-cursor="interactive"
      >
        <div className="flex items-center gap-3 text-[length:var(--text-xs)] tracking-[var(--tracking-wide)] uppercase">
          <span style={{ color: "var(--accent)" }}>{post.source}</span>
          <span aria-hidden="true" style={{ color: "var(--text-faint)" }}>
            ·
          </span>
          <span style={{ color: "var(--text-faint)" }}>
            {formatDate(post.publishedAt)}
          </span>
        </div>
        <h3
          className="mt-2 flex items-start gap-3 font-bold leading-[var(--leading-tight)] transition-transform duration-[var(--duration-normal)] ease-[var(--ease-out-expo)] group-hover:translate-x-1"
          style={{ fontSize: "var(--text-lg)" }}
        >
          <span className="transition-colors duration-[var(--duration-normal)] group-hover:text-[var(--accent)]">
            {post.title}
          </span>
          <span
            aria-hidden="true"
            className="opacity-0 transition-opacity duration-[var(--duration-normal)] group-hover:opacity-100"
            style={{ color: "var(--accent)" }}
          >
            ↗
          </span>
        </h3>
        {post.excerpt ? (
          <p
            className="mt-2 max-w-2xl"
            style={{ color: "var(--text-muted)", fontSize: "var(--text-sm)" }}
          >
            {post.excerpt}
          </p>
        ) : null}
      </a>
    </li>
  );
}
