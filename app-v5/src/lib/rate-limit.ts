/**
 * Fixed-window in-memory rate limiter — ported from v4 api/ai.js.
 *
 * Honest limitation (fine for a portfolio endpoint): state lives in the
 * function instance. Vercel Fluid Compute reuses instances across requests
 * so it works in practice, but a cold start or parallel instance resets the
 * window. Upgrade path if it ever matters: Vercel KV / Upstash.
 */

interface RateLimiterOptions {
  windowMs: number;
  max: number;
}

interface WindowEntry {
  windowStart: number;
  count: number;
}

export interface RateLimiter {
  isLimited(clientId: string): boolean;
}

export function createRateLimiter(options: RateLimiterOptions): RateLimiter {
  const windows = new Map<string, WindowEntry>();

  return {
    isLimited(clientId: string): boolean {
      const now = Date.now();
      const entry = windows.get(clientId);

      if (!entry || now - entry.windowStart > options.windowMs) {
        windows.set(clientId, { windowStart: now, count: 1 });
        return false;
      }

      entry.count += 1;
      return entry.count > options.max;
    },
  };
}
