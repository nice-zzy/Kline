/** @type {import('next').NextConfig} */
const nextConfig = {
  async rewrites() {
    return [
      // 仅将分析接口转发到 Python；/api/chats 等由 Next.js 自身处理（Supabase）
      {
        source: "/api/analyze",
        destination: "http://localhost:8000/analyze",
      },
    ];
  },
};

export default nextConfig;
