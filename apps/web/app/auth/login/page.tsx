"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { useAuth } from "@/contexts/AuthContext";

export default function LoginPage() {
  const { user, loading, signIn } = useAuth();
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    if (!loading && user) {
      router.replace("/");
    }
  }, [user, loading, router]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setSubmitting(true);
    try {
      const timeoutMs = 15000;
      await Promise.race([
        signIn(email.trim(), password),
        new Promise<never>((_, reject) =>
          setTimeout(() => reject(new Error("连接超时，请检查网络或使用可访问国际网络的代理后重试")), timeoutMs)
        ),
      ]);
      router.replace("/");
    } catch (err) {
      setError(err instanceof Error ? err.message : "登录失败");
    } finally {
      setSubmitting(false);
    }
  };

  if (loading) {
    return (
      <div className="authPage">
        <div className="authSpinner" aria-hidden />
      </div>
    );
  }

  if (user) return null;

  return (
    <div className="authPage">
      <div className="authCard">
        <h1 className="authTitle">登录</h1>
        <p className="authSubtitle">登录后对话将同步到云端</p>
        <p className="authHint">若长时间无响应，可能是无法连接 Supabase，请检查网络或使用可访问国际网络的代理。</p>
        <form onSubmit={handleSubmit} className="authForm">
          <label className="authLabel">
            邮箱
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="authInput"
              placeholder="your@email.com"
              required
              autoComplete="email"
            />
          </label>
          <label className="authLabel">
            密码
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="authInput"
              placeholder="••••••••"
              required
              autoComplete="current-password"
            />
          </label>
          {error && <p className="authError">{error}</p>}
          <button type="submit" className="authSubmit" disabled={submitting}>
            {submitting ? "登录中…" : "登录"}
          </button>
        </form>
        <p className="authFooter">
          还没有账号？ <Link href="/auth/signup">注册</Link>
        </p>
      </div>
    </div>
  );
}
