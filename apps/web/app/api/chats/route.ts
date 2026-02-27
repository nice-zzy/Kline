import { NextRequest, NextResponse } from "next/server";
import { requireAuth, createAuthError } from "@/lib/auth/server";
import { getChatService } from "@/lib/services/chats";

export async function GET() {
  try {
    const user = await requireAuth();
    const chatService = getChatService();
    const list = await chatService.listChats(user.id);
    return NextResponse.json({ chats: list });
  } catch (err) {
    if (err instanceof Error && err.message === "Authentication required") {
      return createAuthError();
    }
    console.error("GET /api/chats failed:", err);
    return NextResponse.json(
      { error: err instanceof Error ? err.message : "Internal error" },
      { status: 500 }
    );
  }
}

export async function POST(req: NextRequest) {
  try {
    const user = await requireAuth();
    const body = await req.json().catch(() => ({}));
    const title = typeof body.title === "string" ? body.title.trim() || "New chat" : "New chat";
    const chatService = getChatService();
    const chat = await chatService.createChat(user.id, title);
    return NextResponse.json(chat, { status: 201 });
  } catch (err) {
    if (err instanceof Error && err.message === "Authentication required") {
      return createAuthError();
    }
    console.error("POST /api/chats failed:", err);
    return NextResponse.json(
      { error: err instanceof Error ? err.message : "Internal error" },
      { status: 500 }
    );
  }
}
