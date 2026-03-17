import React from "react";

export class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error("ErrorBoundary caught:", error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen bg-slate-950 flex items-center justify-center p-8">
          <div className="max-w-md text-center">
            <div className="text-6xl mb-6 font-mono text-red-500">!</div>
            <h1 className="text-2xl font-bold text-white mb-4 font-mono">
              SYSTEM ERROR
            </h1>
            <p className="text-slate-400 mb-6 font-mono text-sm">
              {this.state.error?.message || "An unexpected error occurred."}
            </p>
            <button
              onClick={() => {
                this.setState({ hasError: false, error: null });
                window.location.href = "/";
              }}
              className="bg-emerald-600 hover:bg-emerald-500 text-white font-mono px-6 py-3 rounded transition-colors cursor-pointer"
            >
              REBOOT SYSTEM
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
