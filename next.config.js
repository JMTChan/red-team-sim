/** @type {import('next').NextConfig} */

// Project pages on GitHub are served from https://<user>.github.io/<repo>/,
// so production builds need a basePath. Change `repo` if you rename the repo.
const repo = "red-team-sim";
const isProd = process.env.NODE_ENV === "production";
const basePath = isProd ? `/${repo}` : "";

const nextConfig = {
  output: "export", // 100% static HTML/CSS/JS -- no SSR, no API routes
  basePath,
  assetPrefix: basePath ? `${basePath}/` : "",
  images: { unoptimized: true }, // next/image optimizer needs a server; disable it
  trailingSlash: true, // friendlier paths on static hosts
  env: {
    NEXT_PUBLIC_BASE_PATH: basePath, // read by the client to locate the .onnx file
  },
};

module.exports = nextConfig;
