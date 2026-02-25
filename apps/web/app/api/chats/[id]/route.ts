import { NextRequest, NextResponse } from "next/server";
import { requireAuth, createAuthError } from "@/lib/auth/server";
import { getChatService } from "@/lib/services/chats";
import type { Message } from "@/lib/types/chat";

export async function GET(
  _req: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const user = await requireAuth();
    const { id } = await params;
    const chatService = getChatService();
    const chat = await chatService.getChat(id, user.id);
    if (!chat) {
      console.warn("[GET /api/chats/[id]] Not found:", { id, userId: user.id });
      return NextResponse.json({ error: "Not found" }, { status: 404 });
    }
    return NextResponse.json(chat);
  } catch (err) {
    if (err instanceof Error && err.message === "Authentication required") {
      return createAuthError();
    }
    console.error("GET /api/chats/[id] failed:", err);
    return NextResponse.json(
      { error: err instanceof Error ? err.message : "Internal error" },
      { status: 500 }
    );
  }
}

export async function PATCH(
  req: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const user = await requireAuth();
    const { id } = await params;
    const body = await req.json().catch(() => ({}));
    const updates: { title?: string; messages?: Message[] } = {};
    if (typeof body.title === "string") updates.title = body.title.trim();
    if (Array.isArray(body.messages)) updates.messages = body.messages;
    const chatService = getChatService();
    const chat = await chatService.updateChat(id, user.id, updates);
    if (!chat) {
      return NextResponse.json({ error: "Not found" }, { status: 404 });
    }
    return NextResponse.json(chat);
  } catch (err) {
    if (err instanceof Error && err.message === "Authentication required") {
      return createAuthError();
    }
    console.error("PATCH /api/chats/[id] failed:", err);
    return NextResponse.json(
      { error: err instanceof Error ? err.message : "Internal error" },
      { status: 500 }
    );
  }
}

export async function DELETE(
  _req: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const user = await requireAuth();
    const { id } = await params;
    const chatService = getChatService();
    await chatService.deleteChat(id, user.id);
    return new NextResponse(null, { status: 204 });
  } catch (err) {
    if (err instanceof Error && err.message === "Authentication required") {
      return createAuthError();
    }
    console.error("DELETE /api/chats/[id] failed:", err);
    return NextResponse.json(
      { error: err instanceof Error ? err.message : "Internal error" },
      { status: 500 }
    );
  }
}
