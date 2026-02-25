/**
 * 服务端鉴权：与 portfoliopilot-main lib/auth/server.ts 一致
 * - getAuthenticatedUser：cookie 客户端 getUser()，失败返回 null
 * - requireAuth：无用户时抛 "Authentication required"，供 API 统一 createAuthError(401)
 */
import { createClient } from "@/lib/supabase/server";

export async function getAuthenticatedUser() {
  const supabase = await createClient();
  try {
    let {
      data: { user },
      error,
    } = await supabase.auth.getUser();
    if (!error && user) return user;
    if (error) console.error("Auth error:", error);

    const { data: { session } } = await supabase.auth.getSession();
    if (session) {
      const { error: refreshError } = await supabase.auth.refreshSession();
      if (!refreshError) {
        const { data: { user: u } } = await supabase.auth.getUser();
        return u ?? null;
      }
    }
    return null;
  } catch (err) {
    console.error("Failed to get authenticated user:", err);
    return null;
  }
}

export async function requireAuth() {
  const user = await getAuthenticatedUser();
  if (!user) {
    throw new Error("Authentication required");
  }
  return user;
}

export function createAuthError(message = "Unauthorized", status = 401) {
  return Response.json({ error: message }, { status });
}
