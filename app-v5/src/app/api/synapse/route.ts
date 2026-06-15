import { createRateLimiter } from "@/lib/rate-limit";
import {
  validateSynapseRequest,
  type SynapseRequest,
} from "@/lib/synapse-validation";
import { buildSynapseContext } from "@/lib/synapse-context";

/**
 * Synapse proxy — ported from v4 api/ai.js with a narrower contract:
 * the client sends ONLY a chat history; the system instruction is built
 * server-side from the site's own data. The Gemini key never leaves the
 * server (Phase-6 exit criterion).
 */

const GEMINI_URL =
  "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent";

const limiter = createRateLimiter({ windowMs: 60_000, max: 15 });

function jsonError(message: string, status: number): Response {
  return Response.json({ error: message }, { status });
}

export async function POST(request: Request): Promise<Response> {
  const clientIp =
    request.headers.get("x-forwarded-for")?.split(",")[0]?.trim() ?? "unknown";
  if (limiter.isLimited(clientIp)) {
    return jsonError("Rate limit exceeded. Try again in a minute.", 429);
  }

  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    return jsonError("AI backend not configured.", 500);
  }

  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return jsonError("Invalid JSON body.", 400);
  }

  const validationError = validateSynapseRequest(body);
  if (validationError) {
    return jsonError(validationError, 400);
  }

  const { messages } = body as SynapseRequest;
  const geminiBody = {
    contents: messages.map((message) => ({
      role: message.role,
      parts: [{ text: message.text }],
    })),
    systemInstruction: { parts: [{ text: buildSynapseContext() }] },
  };

  try {
    const response = await fetch(`${GEMINI_URL}?key=${apiKey}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(geminiBody),
    });

    const text = await response.text();
    if (!text) {
      return jsonError("AI service returned empty response.", 502);
    }

    let data: unknown;
    try {
      data = JSON.parse(text);
    } catch {
      console.error("Gemini non-JSON response:", text.slice(0, 200));
      return jsonError("AI service returned invalid response.", 502);
    }

    const reply = (
      data as {
        candidates?: { content?: { parts?: { text?: string }[] } }[];
        error?: { message?: string };
      }
    );
    if (reply.error) {
      console.error("Gemini error:", reply.error.message);
      return jsonError("AI service rejected the request.", 502);
    }

    const answer = reply.candidates?.[0]?.content?.parts?.[0]?.text;
    if (!answer) {
      return jsonError("AI service returned no answer.", 502);
    }

    // Narrow response surface: the client gets the answer, nothing else.
    return Response.json({ answer });
  } catch (error) {
    console.error("Gemini proxy error:", error);
    return jsonError("AI service unavailable.", 502);
  }
}
