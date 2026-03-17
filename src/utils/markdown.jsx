import React from "react";
import { Hash } from "lucide-react";

/**
 * Parses inline markdown: **bold**, `code`, [text](url)
 */
function parseInline(text) {
  const parts = [];
  let remaining = text;
  let key = 0;

  while (remaining.length > 0) {
    // Bold: **text**
    const boldMatch = remaining.match(/\*\*(.+?)\*\*/);
    // Code: `text`
    const codeMatch = remaining.match(/`(.+?)`/);
    // Link: [text](url)
    const linkMatch = remaining.match(/\[(.+?)\]\((.+?)\)/);

    // Find earliest match
    const matches = [
      boldMatch ? { type: "bold", match: boldMatch } : null,
      codeMatch ? { type: "code", match: codeMatch } : null,
      linkMatch ? { type: "link", match: linkMatch } : null,
    ]
      .filter(Boolean)
      .sort((a, b) => a.match.index - b.match.index);

    if (matches.length === 0) {
      parts.push(remaining);
      break;
    }

    const first = matches[0];
    const idx = first.match.index;

    // Add text before match
    if (idx > 0) {
      parts.push(remaining.slice(0, idx));
    }

    if (first.type === "bold") {
      parts.push(
        <strong key={key++} className="text-white font-semibold">
          {first.match[1]}
        </strong>,
      );
      remaining = remaining.slice(idx + first.match[0].length);
    } else if (first.type === "code") {
      parts.push(
        <code
          key={key++}
          className="bg-slate-800 text-emerald-300 px-1.5 py-0.5 rounded text-xs font-mono"
        >
          {first.match[1]}
        </code>,
      );
      remaining = remaining.slice(idx + first.match[0].length);
    } else if (first.type === "link") {
      parts.push(
        <a
          key={key++}
          href={first.match[2]}
          target="_blank"
          rel="noreferrer"
          className="text-emerald-400 hover:text-emerald-300 underline underline-offset-2"
        >
          {first.match[1]}
        </a>,
      );
      remaining = remaining.slice(idx + first.match[0].length);
    }
  }

  return parts;
}

/**
 * Renders markdown content string into React elements.
 * Supports: ### headings, numbered lists, **bold**, `code`, [links](url)
 */
export function renderMarkdown(content) {
  return content.split("\n").map((line, i) => {
    if (line.trim().startsWith("###")) {
      return (
        <h3
          key={i}
          className="text-xl font-bold text-white mt-8 mb-4 flex items-center gap-2"
        >
          <Hash size={16} className="text-emerald-500" />
          {line.replace("###", "").trim()}
        </h3>
      );
    }
    if (line.trim().match(/^\d\./)) {
      return (
        <div
          key={i}
          className="ml-4 mb-2 text-slate-300 pl-4 border-l border-slate-700"
        >
          {parseInline(line.trim())}
        </div>
      );
    }
    if (line.trim() === "") {
      return null;
    }
    return (
      <p key={i} className="text-slate-300 leading-relaxed mb-4">
        {parseInline(line)}
      </p>
    );
  });
}
