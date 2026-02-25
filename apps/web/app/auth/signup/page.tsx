"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { useAuth } from "@/contexts/AuthContext";

export default function SignupPage() {
  const { user, loading, signUp } = useAuth();
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
      await signUp(email.trim(), password);
      router.replace("/");
    } catch (err) {
      setError(err instanceof Error ? err.message : "注册失败");
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
        <h1 className="authTitle">注册</h1>
        <p className="authSubtitle">注册后对话将同步到云端</p>
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
              placeholder="至少 6 位"
              required
              minLength={6}
              autoComplete="new-password"
            />
          </label>
          {error && <p className="authError">{error}</p>}
          <button type="submit" className="authSubmit" disabled={submitting}>
            {submitting ? "注册中…" : "注册"}
          </button>
        </form>
        <p className="authFooter">
          已有账号？ <Link href="/auth/login">登录</Link>
        </p>
      </div>
    </div>
  );
}
