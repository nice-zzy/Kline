import "./styles/globals.css";
import { AuthProvider } from "@/contexts/AuthContext";

export const metadata = {
  title: "K-line Analyst",
  description: "输入 OHLC 数据或上传 K 线图，获取分析报告。",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="zh-CN">
      <body>
        <AuthProvider>{children}</AuthProvider>
      </body>
    </html>
  );
}


