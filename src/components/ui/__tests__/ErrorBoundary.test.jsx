import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import { ErrorBoundary } from "../ErrorBoundary";

// A component that throws on render
function Thrower({ shouldThrow }) {
  if (shouldThrow) throw new Error("Test explosion");
  return <div>All good</div>;
}

describe("ErrorBoundary", () => {
  beforeEach(() => {
    // Suppress React error boundary console output during tests
    vi.spyOn(console, "error").mockImplementation(() => {});
  });

  it("renders children when no error", () => {
    render(
      <ErrorBoundary>
        <div>Child content</div>
      </ErrorBoundary>,
    );
    expect(screen.getByText("Child content")).toBeTruthy();
  });

  it("renders error UI when child throws", () => {
    render(
      <ErrorBoundary>
        <Thrower shouldThrow={true} />
      </ErrorBoundary>,
    );
    expect(screen.getByText("SYSTEM ERROR")).toBeTruthy();
    expect(screen.getByText(/Test explosion/)).toBeTruthy();
  });

  it("shows REBOOT SYSTEM button on error", () => {
    render(
      <ErrorBoundary>
        <Thrower shouldThrow={true} />
      </ErrorBoundary>,
    );
    expect(screen.getByText("REBOOT SYSTEM")).toBeTruthy();
  });

  it("displays a fallback message when error has no message", () => {
    function EmptyThrower() {
      throw new Error();
    }

    render(
      <ErrorBoundary>
        <EmptyThrower />
      </ErrorBoundary>,
    );
    expect(screen.getByText("SYSTEM ERROR")).toBeTruthy();
    // Should show fallback text
    expect(
      screen.getByText("An unexpected error occurred."),
    ).toBeTruthy();
  });
});
