/**
 * Deterministic PRNG (ADR-3): node positions must be identical on server and
 * client — no runtime Math.random() — or React hydration fails.
 */

/** FNV-1a 32-bit hash: stable numeric seed from a node id. */
export function hashString(input: string): number {
  let hash = 0x811c9dc5;
  for (let i = 0; i < input.length; i += 1) {
    hash ^= input.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193);
  }
  return hash >>> 0;
}

/** Mulberry32: tiny, fast, deterministic generator returning floats in [0, 1). */
export function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) | 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Convenience: a generator keyed directly on a node id. */
export function seededRandom(id: string): () => number {
  return mulberry32(hashString(id));
}
