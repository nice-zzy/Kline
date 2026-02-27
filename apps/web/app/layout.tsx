import "./styles/globals.css";
import { AuthProvider } from "@/contexts/AuthContext";

export const metadata = {
  title: "K-line Analyst",
  description: "Enter OHLC data or upload a candlestick chart to get an analysis report.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>
        <AuthProvider>{children}</AuthProvider>
      </body>
    </html>
  );
}


