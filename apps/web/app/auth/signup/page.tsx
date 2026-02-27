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
      setError(err instanceof Error ? err.message : "Sign up failed.");
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
        <h1 className="authTitle">Sign up</h1>
        <p className="authSubtitle">Create an account to sync your chats to the cloud.</p>
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
              placeholder="At least 6 characters"
              required
              minLength={6}
              autoComplete="new-password"
            />
          </label>
          {error && <p className="authError">{error}</p>}
          <button type="submit" className="authSubmit" disabled={submitting}>
            {submitting ? "Signing up…" : "Sign up"}
          </button>
        </form>
        <p className="authFooter">
          Already have an account? <Link href="/auth/login">Log in</Link>
        </p>
      </div>
    </div>
  );
}
