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

// Health check — confirms which server version is running
app.get("/api/health", (req, res) => {
  res.json({ status: "ok", model: "gemini-2.5-flash", version: "4" });
});

// Input validation for AI requests
const MAX_TEXT_LENGTH = 2000;
function validateAiRequest(body) {
  if (!body || typeof body !== "object") return "Invalid request body.";
  if (!Array.isArray(body.contents) || body.contents.length === 0)
    return "Missing 'contents' array.";

  for (const content of body.contents) {
    if (!content.parts || !Array.isArray(content.parts))
      return "Each content must have a 'parts' array.";
    for (const part of content.parts) {
      if (typeof part.text !== "string") return "Each part must have a 'text' string.";
      if (part.text.length > MAX_TEXT_LENGTH)
        return `Text exceeds maximum length of ${MAX_TEXT_LENGTH} characters.`;
    }
  }
  return null;
}

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

  // Validate input
  const validationError = validateAiRequest(req.body);
  if (validationError) {
    return res.status(400).json({ error: validationError });
  }

  // Build sanitized payload — only forward contents and systemInstruction
  const sanitizedBody = { contents: req.body.contents };
  if (req.body.systemInstruction) {
    sanitizedBody.systemInstruction = req.body.systemInstruction;
  }

  try {
    const response = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=${apiKey}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(sanitizedBody),
      },
    );

    const text = await response.text();
    if (!text) {
      return res.status(502).json({ error: "AI service returned empty response." });
    }

    let data;
    try {
      data = JSON.parse(text);
    } catch {
      console.error("Gemini non-JSON response:", text.slice(0, 200));
      return res.status(502).json({ error: "AI service returned invalid response." });
    }

    res.json(data);
  } catch (error) {
    console.error("Gemini proxy error:", error);
    res.status(502).json({ error: "AI service unavailable." });
  }
});

// Serve static files from dist/
app.use(express.static(path.join(__dirname, "dist")));

// SPA fallback — serve index.html for all non-API routes
app.get("/{*path}", (req, res) => {
  res.sendFile(path.join(__dirname, "dist", "index.html"));
});

app.listen(PORT, () => {
  console.log(`VASU_OS server running on port ${PORT}`);
});
