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
          setTimeout(() => reject(new Error("Connection timed out. Please check your network or try with a VPN that can access international services.")), timeoutMs)
        ),
      ]);
      router.replace("/");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Login failed.");
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
        <h1 className="authTitle">Log in</h1>
        <p className="authSubtitle">Sign in to sync your chats to the cloud.</p>
        <p className="authHint">If there is no response for a long time, you may be unable to reach Supabase. Please check your network or use a VPN that can access international services.</p>
        <form onSubmit={handleSubmit} className="authForm">
          <label className="authLabel">
            Email
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
            Password
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
            {submitting ? "Signing in…" : "Log in"}
          </button>
        </form>
        <p className="authFooter">
          Don&apos;t have an account? <Link href="/auth/signup">Sign up</Link>
        </p>
      </div>
    </div>
  );
}
