import { projects } from "../data/projects";
import { engineeringLogs } from "../data/blog-posts";
import { profile } from "../data/profile";
import { skills } from "../data/skills";

/**
 * Processes built-in terminal commands. Returns an array of output lines,
 * or null if the command is not recognized (delegate to AI).
 */
export function processCommand(command, { setExpandedProject, setIsTerminalOpen, isRetro, setIsRetro }) {
  const args = command.split(" ");
  const cmd = args[0];

  switch (cmd) {
    case "./help":
      return [
        "Available commands:",
        "./list-projects - List all projects",
        "./open [alias/id] - Open project deep dive",
        "./stats [alias/id] - View quick stats",
        "./ship-log - List recent engineering logs",
        "./whoami - Display user context",
        "./clear - Clear terminal",
        "./retro - Toggle Retro Mode",
        "[Any Question] - Ask VASU_OS AI anything",
      ];

    case "./whoami":
      return ["User: GUEST", "Role: RECRUITER / ENGINEER", "Access: READ_ONLY"];

    case "./retro":
      setIsRetro((prev) => !prev);
      return [`Retro Mode: ${!isRetro ? "ON" : "OFF"}`];

    case "./ls":
    case "./list-projects":
      return projects.map(
        (p) => `[${p.id}] [${p.tier}] ${p.alias} -> ${p.title}`,
      );

    case "./ship-log":
      return engineeringLogs.map((l) => `[${l.date}] ${l.title}`);

    case "./project":
    case "./open": {
      const target = args[1];
      if (target) {
        const proj = !isNaN(parseInt(target))
          ? projects.find((p) => p.id === parseInt(target))
          : projects.find((p) => p.alias === target);
        if (proj) {
          setTimeout(() => {
            setExpandedProject(proj);
            setIsTerminalOpen(false);
          }, 800);
          return [`Opening Deep Dive for: ${proj.title}...`];
        }
        return [`Error: Project '${target}' not found.`];
      }
      return ["Error: Missing argument. Usage: ./open insightify"];
    }

    case "./stats": {
      const statTarget = args[1];
      if (statTarget) {
        const proj = projects.find(
          (p) => p.alias === statTarget || p.id === parseInt(statTarget),
        );
        if (proj && proj.details) {
          return [
            `STATS FOR: ${proj.title}`,
            `--------------------------------`,
            `Tier: ${proj.tier}`,
            `Stack: ${proj.tech.join(", ")}`,
            `Metrics: ${proj.details.metrics}`,
            `Status: ${proj.status}`,
          ];
        }
        return [`Error: Project '${statTarget}' not found.`];
      }
      return ["Error: Missing argument. Usage: ./stats covid-xray"];
    }

    default:
      return null; // Not a built-in command, delegate to AI
  }
}

// Cached system context — built once at module load (perf fix #17)
const SYSTEM_CONTEXT = `You are VASU_OS, the AI assistant for Vasu Agrawal's engineering portfolio. Your persona: Technical, concise, slightly witty, "hacker" aesthetic. Here is Vasu's data: Profile: ${JSON.stringify(
  profile,
)}, Skills: ${JSON.stringify(skills)}, Projects: ${JSON.stringify(
  projects.map((p) => ({
    title: p.title,
    tech: p.tech,
    description: p.description,
    details: p.details,
  })),
)} Engineering Logs: ${JSON.stringify(
  engineeringLogs.map((l) => ({
    title: l.title,
    summary: l.content.substring(0, 100),
  })),
)}. Answer the user's question based STRICTLY on this data. Keep answers short (under 3 sentences) to fit the terminal style.`;

/**
 * Generates an AI response via the backend proxy (which calls Gemini).
 * Always uses /api/ai proxy — API key is never exposed client-side.
 */
export async function generateAiResponse(userQuery) {
  const requestBody = {
    contents: [{ parts: [{ text: userQuery }] }],
    systemInstruction: { parts: [{ text: SYSTEM_CONTEXT }] },
  };

  try {
    const response = await fetch("/api/ai", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(requestBody),
    });
    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.error || "AI service unavailable");
    }
    const data = await response.json();

    if (data.error) throw new Error(data.error.message);
    return (
      data.candidates?.[0]?.content?.parts?.[0]?.text ||
      "Error: Neural link unstable."
    );
  } catch (error) {
    console.error("AI Error:", error);
    return `Error: AI Core Offline. (${error.message})`;
  }
}
