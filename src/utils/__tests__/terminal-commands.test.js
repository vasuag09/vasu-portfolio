import { describe, it, expect, vi } from "vitest";
import { processCommand, generateAiResponse } from "../terminal-commands";

// Mock callbacks for processCommand
const makeMocks = () => ({
  setExpandedProject: vi.fn(),
  setIsTerminalOpen: vi.fn(),
  isRetro: false,
  setIsRetro: vi.fn(),
});

describe("processCommand", () => {
  it("./help returns list of available commands", () => {
    const result = processCommand("./help", makeMocks());
    expect(result).toBeInstanceOf(Array);
    expect(result.length).toBeGreaterThan(0);
    expect(result[0]).toContain("Available commands");
  });

  it("./whoami returns user context", () => {
    const result = processCommand("./whoami", makeMocks());
    expect(result).toContain("User: GUEST");
    expect(result).toContain("Role: RECRUITER / ENGINEER");
  });

  it("./list-projects returns all projects", () => {
    const result = processCommand("./list-projects", makeMocks());
    expect(result).toBeInstanceOf(Array);
    expect(result.length).toBe(8);
    result.forEach((line) => {
      expect(line).toMatch(/\[\d+\]/); // [id]
      expect(line).toMatch(/\[[SAB]\]/); // [tier]
    });
  });

  it("./ls is an alias for ./list-projects", () => {
    const result = processCommand("./ls", makeMocks());
    expect(result).toHaveLength(8);
  });

  it("./ship-log returns engineering logs", () => {
    const result = processCommand("./ship-log", makeMocks());
    expect(result).toBeInstanceOf(Array);
    expect(result.length).toBe(5);
  });

  it("./retro toggles retro mode", () => {
    const mocks = makeMocks();
    const result = processCommand("./retro", mocks);
    expect(mocks.setIsRetro).toHaveBeenCalledOnce();
    expect(result).toHaveLength(1);
    expect(result[0]).toContain("Retro Mode");
  });

  it("./open without argument returns usage error", () => {
    const result = processCommand("./open", makeMocks());
    expect(result[0]).toContain("Error: Missing argument");
  });

  it("./open with invalid project returns not found", () => {
    const result = processCommand("./open nonexistent", makeMocks());
    expect(result[0]).toContain("not found");
  });

  it("./stats without argument returns usage error", () => {
    const result = processCommand("./stats", makeMocks());
    expect(result[0]).toContain("Error: Missing argument");
  });

  it("./stats with invalid project returns not found", () => {
    const result = processCommand("./stats nonexistent", makeMocks());
    expect(result[0]).toContain("not found");
  });

  it("returns null for unrecognized commands (delegate to AI)", () => {
    const result = processCommand("what is react?", makeMocks());
    expect(result).toBeNull();
  });
});

describe("generateAiResponse", () => {
  it("returns error when API key is empty", async () => {
    const result = await generateAiResponse("hello", "");
    expect(result).toContain("Error");
    expect(result).toContain("API Key");
  });

  it("returns error when API key is undefined", async () => {
    const result = await generateAiResponse("hello", undefined);
    expect(result).toContain("Error");
  });

  it("handles fetch failure gracefully", async () => {
    const originalFetch = globalThis.fetch;
    globalThis.fetch = vi.fn().mockRejectedValue(new Error("Network error"));

    const result = await generateAiResponse("hello", "test-key");
    expect(result).toContain("Error");
    expect(result).toContain("Network error");

    globalThis.fetch = originalFetch;
  });

  it("handles API error response", async () => {
    const originalFetch = globalThis.fetch;
    globalThis.fetch = vi.fn().mockResolvedValue({
      json: () =>
        Promise.resolve({ error: { message: "Invalid API key" } }),
    });

    const result = await generateAiResponse("hello", "bad-key");
    expect(result).toContain("Error");
    expect(result).toContain("Invalid API key");

    globalThis.fetch = originalFetch;
  });

  it("returns AI response on success", async () => {
    const originalFetch = globalThis.fetch;
    globalThis.fetch = vi.fn().mockResolvedValue({
      json: () =>
        Promise.resolve({
          candidates: [
            { content: { parts: [{ text: "Hello from AI" }] } },
          ],
        }),
    });

    const result = await generateAiResponse("hello", "valid-key");
    expect(result).toBe("Hello from AI");

    globalThis.fetch = originalFetch;
  });
});
