import { fetchWritingPosts, LINKEDIN_URL } from "@/lib/writing-fetch";
import { PostRow } from "./PostRow";

/**
 * Signal chapter content — async Server Component. Fetches the merged
 * Medium + Substack feed at build/ISR time (writing-fetch sets revalidate),
 * so the list self-updates with zero client JS. Fail-soft: no posts (every
 * feed down) → render only the "follow on LinkedIn" line, never an error.
 */
export async function WritingPosts() {
  const posts = await fetchWritingPosts(5);

  return (
    <div>
      {posts.length > 0 ? (
        <ul className="border-b border-[var(--hairline,oklch(100%_0_0/0.1))]">
          {posts.map((post) => (
            <PostRow key={post.url} post={post} />
          ))}
        </ul>
      ) : (
        <p style={{ color: "var(--text-muted)", fontSize: "var(--text-lg)" }}>
          Latest essays on{" "}
          <a
            className="underline decoration-[var(--accent-dim)] underline-offset-4 hover:decoration-[var(--accent)]"
            href="https://medium.com/@vasuagrawal1040"
            target="_blank"
            rel="noopener noreferrer"
          >
            Medium
          </a>{" "}
          and{" "}
          <a
            className="underline decoration-[var(--accent-dim)] underline-offset-4 hover:decoration-[var(--accent)]"
            href="https://vasuagrawal.substack.com"
            target="_blank"
            rel="noopener noreferrer"
          >
            Substack
          </a>
          .
        </p>
      )}

      <p
        className="mt-8 text-[length:var(--text-xs)] tracking-[var(--tracking-wide)] uppercase"
        style={{ color: "var(--text-faint)" }}
      >
        Also writing on{" "}
        <a
          className="underline decoration-[var(--accent-dim)] underline-offset-4 transition-colors hover:text-[var(--accent)] hover:decoration-[var(--accent)]"
          href={LINKEDIN_URL}
          target="_blank"
          rel="noopener noreferrer"
          data-cursor="interactive"
        >
          LinkedIn ↗
        </a>
      </p>
    </div>
  );
}
