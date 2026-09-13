/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  eslint: { ignoreDuringBuilds: true },
  experimental: {
    // Markdown content is read from disk at request/build time.
    serverActions: { bodySizeLimit: '2mb' },
  },
};

export default nextConfig;
