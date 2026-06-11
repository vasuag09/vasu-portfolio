import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // ADR-5: app-v5 is a self-contained workspace inside the v4 repo; the v4
  // lockfile at the repo root must not be treated as this project's root.
  turbopack: {
    root: __dirname,
  },
};

export default nextConfig;
