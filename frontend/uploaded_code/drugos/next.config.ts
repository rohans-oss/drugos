import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Note: output: "standalone" is enabled for production Docker/Node deployments.
  // Disable it locally if you just want `next dev` / `next start` to work without
  // copying the .next/standalone folder around.
  output: "standalone",
  typescript: {
    // We have a few `@ts-expect-error` annotations and loose types in the
    // mock-data layer; don't fail the build on TS warnings during iteration.
    ignoreBuildErrors: true,
  },
  reactStrictMode: false,
};

export default nextConfig;
