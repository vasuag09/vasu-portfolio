import express from "express";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const app = express();
const PORT = process.env.PORT || 3000;

// Rate limiting: simple in-memory per-IP tracker
const rateLimiter = new Map();
const RATE_LIMIT_WINDOW = 60_000; // 1 minute
const RATE_LIMIT_MAX = 15; // 15 requests per minute

function isRateLimited(ip) {
  const now = Date.now();
  const entry = rateLimiter.get(ip);

  if (!entry || now - entry.windowStart > RATE_LIMIT_WINDOW) {
    rateLimiter.set(ip, { windowStart: now, count: 1 });
    return false;
  }

  entry.count++;
  return entry.count > RATE_LIMIT_MAX;
}

// Clean up stale entries every 5 minutes
setInterval(() => {
  const now = Date.now();
  for (const [ip, entry] of rateLimiter) {
    if (now - entry.windowStart > RATE_LIMIT_WINDOW) {
      rateLimiter.delete(ip);
    }
  }
}, 300_000);

app.use(express.json());

// API proxy endpoint
app.post("/api/ai", async (req, res) => {
  const clientIp = req.headers["x-forwarded-for"] || req.socket.remoteAddress;

  if (isRateLimited(clientIp)) {
    return res.status(429).json({ error: "Rate limit exceeded. Try again in a minute." });
  }

  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    return res.status(500).json({ error: "AI backend not configured." });
  }

  try {
    const response = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=${apiKey}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(req.body),
      },
    );

    const data = await response.json();
    res.json(data);
  } catch (error) {
    console.error("Gemini proxy error:", error);
    res.status(502).json({ error: "AI service unavailable." });
  }
});

// Serve static files from dist/
app.use(express.static(path.join(__dirname, "dist")));

// SPA fallback — serve index.html for all non-API routes
app.get("*", (req, res) => {
  res.sendFile(path.join(__dirname, "dist", "index.html"));
});

app.listen(PORT, () => {
  console.log(`VASU_OS server running on port ${PORT}`);
});
