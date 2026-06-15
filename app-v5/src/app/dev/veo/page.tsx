import { VeoClip } from "@/components/media/VeoClip";
import { veoSources, veoPoster } from "@/lib/veo-sources";

/**
 * DEV-ONLY harness for the VeoClip component (Phase 4). Not linked from
 * anywhere; remove at /ship (Phase 12 checklist). The spacer forces the
 * clip below the fold so lazy mounting is actually exercised.
 */
export default function VeoTestPage() {
  return (
    <main className="mx-auto max-w-3xl px-6 py-12">
      <h1 style={{ fontSize: "var(--text-lg)" }}>VeoClip test harness</h1>
      <p className="mt-2" style={{ color: "var(--text-muted)" }}>
        Scroll down — the video must mount lazily, play on entry, pause when
        scrolled away. Under prefers-reduced-motion only the poster renders.
      </p>
      <div style={{ height: "150svh" }} aria-hidden="true" />
      <VeoClip
        sources={veoSources("placeholder")}
        poster={veoPoster("placeholder")}
        label="Placeholder gradient clip for component testing"
      />
      <div style={{ height: "120svh" }} aria-hidden="true" />
      <p style={{ color: "var(--text-faint)" }}>end of harness</p>
    </main>
  );
}
