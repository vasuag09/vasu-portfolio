/**
 * Veo clip naming convention (docs/v5/VEO-BRIEF.md): files live in
 * /public/veo/ as {name}.{codec}.mp4 + {name}-poster.avif. Sources are
 * ordered best-compression-first — the browser picks the first <source>
 * type it can play (Safari skips AV1 to hvc1; everything plays H.264).
 */

export interface VeoClipSource {
  src: string;
  type: string;
}

export function veoSources(name: string): readonly VeoClipSource[] {
  const base = `/veo/${name}`;
  return [
    { src: `${base}.av1.mp4`, type: 'video/mp4; codecs="av01.0.08M.10"' },
    { src: `${base}.hevc.mp4`, type: 'video/mp4; codecs="hvc1"' },
    { src: `${base}.h264.mp4`, type: 'video/mp4; codecs="avc1.640028"' },
  ];
}

export function veoPoster(name: string): string {
  return `/veo/${name}-poster.avif`;
}
