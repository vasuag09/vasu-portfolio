/**
 * Synapse request schema. Deliberately narrower than v4's /api/ai: the
 * client may ONLY send a chat history — the system instruction is built
 * server-side (buildSynapseContext), so the proxy can't be repurposed as a
 * free general-purpose Gemini gateway.
 */

export const MAX_TEXT_LENGTH = 2000;
export const MAX_MESSAGES = 20;

export interface SynapseMessage {
  role: "user" | "model";
  text: string;
}

export interface SynapseRequest {
  messages: SynapseMessage[];
}

/** Returns an error string, or null when the body is valid. */
export function validateSynapseRequest(body: unknown): string | null {
  if (!body || typeof body !== "object") return "Invalid request body.";

  const record = body as Record<string, unknown>;
  if ("systemInstruction" in record) {
    return "systemInstruction is not accepted.";
  }

  const messages = record.messages;
  if (!Array.isArray(messages) || messages.length === 0) {
    return "Missing 'messages' array.";
  }
  if (messages.length > MAX_MESSAGES) {
    return `History exceeds ${MAX_MESSAGES} messages.`;
  }

  for (const message of messages) {
    if (!message || typeof message !== "object") return "Invalid message.";
    const { role, text } = message as Record<string, unknown>;
    if (role !== "user" && role !== "model") {
      return "Message role must be 'user' or 'model'.";
    }
    if (typeof text !== "string" || text.length === 0) {
      return "Each message needs a non-empty 'text' string.";
    }
    if (text.length > MAX_TEXT_LENGTH) {
      return `Text exceeds maximum length of ${MAX_TEXT_LENGTH} characters.`;
    }
  }
  return null;
}
