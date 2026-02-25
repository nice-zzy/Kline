/**
 * 对话服务：与 portfoliopilot-main 的 StrategyService / TrainingLogService 模式一致
 * - 全部使用 Service Role 客户端，按 user_id 显式过滤（鉴权在 API 层 requireAuth 完成）
 * - 大对象（消息列表）存 Storage，表只存元数据与 messages_url
 */

import { createServiceRoleClient } from "@/lib/supabase/server";
import { getStorageService } from "./storage";
import type { Chat, Message } from "@/lib/types/chat";

const TABLE = "user_chats";
const BUCKET_FOLDER = "chats";

export interface ChatRow {
  id: string;
  user_id: string;
  title: string;
  messages_url: string | null;
  created_at: string;
  updated_at: string;
}

export interface MessagesPayload {
  messages: Message[];
}

export class ChatService {
  private supabase = createServiceRoleClient();
  private storage = getStorageService(true);

  /** 列表：仅元数据，不拉取消息内容 */
  async listChats(userId: string): Promise<ChatRow[]> {
    const { data, error } = await this.supabase
      .from(TABLE)
      .select("id, user_id, title, messages_url, created_at, updated_at")
      .eq("user_id", userId)
      .order("updated_at", { ascending: false });

    if (error) throw new Error(`Failed to list chats: ${error.message}`);
    return (data ?? []) as ChatRow[];
  }

  /** 单条：从 Storage 拉取消息并组装为 Chat */
  async getChat(id: string, userId: string): Promise<Chat | null> {
    const { data: row, error } = await this.supabase
      .from(TABLE)
      .select("*")
      .eq("id", id)
      .eq("user_id", userId)
      .single();

    if (error || !row) {
      if (error?.code === "PGRST116") return null;
      throw new Error(`Failed to get chat: ${error?.message ?? "not found"}`);
    }

    const typed = row as ChatRow;
    let messages: Message[] = [];
    if (typed.messages_url) {
      const payload = await this.storage.downloadJson<MessagesPayload>(typed.messages_url);
      messages = payload?.messages ?? [];
    }

    return {
      id: row.id,
      title: row.title,
      messages,
      createdAt: new Date(row.created_at).getTime(),
    };
  }

  /** 新建：先上传空消息文件再写表，失败则删 Storage（与 portfoliopilot createLog/saveStrategy 一致） */
  async createChat(userId: string, title: string = "新对话"): Promise<Chat> {
    const chatId = crypto.randomUUID();
    const uploadResult = await this.storage.uploadJson(userId, BUCKET_FOLDER, chatId, { messages: [] });

    const now = new Date().toISOString();
    const { data, error } = await this.supabase
      .from(TABLE)
      .insert({
        id: chatId,
        user_id: userId,
        title,
        messages_url: uploadResult.url,
        created_at: now,
        updated_at: now,
      })
      .select()
      .single();

    if (error) {
      await this.storage.deleteFile(uploadResult.url);
      throw new Error(`Failed to create chat: ${error.message}`);
    }

    return {
      id: data.id,
      title: data.title,
      messages: [],
      createdAt: new Date(data.created_at).getTime(),
    };
  }

  /** 更新：若有 messages 先写 Storage 再更新表（与 portfoliopilot updateStrategy 一致） */
  async updateChat(
    id: string,
    userId: string,
    updates: { title?: string; messages?: Message[] }
  ): Promise<Chat | null> {
    const { data: row, error: fetchError } = await this.supabase
      .from(TABLE)
      .select("*")
      .eq("id", id)
      .eq("user_id", userId)
      .single();

    if (fetchError || !row) {
      if (fetchError?.code === "PGRST116") return null;
      throw new Error(`Chat not found: ${fetchError?.message ?? ""}`);
    }

    if (updates.messages !== undefined) {
      await this.storage.uploadJson(userId, BUCKET_FOLDER, id, { messages: updates.messages });
    }

    const updatePayload: Partial<ChatRow> = { updated_at: new Date().toISOString() };
    if (updates.title !== undefined) updatePayload.title = updates.title;

    const { data: updated, error } = await this.supabase
      .from(TABLE)
      .update(updatePayload)
      .eq("id", id)
      .eq("user_id", userId)
      .select()
      .single();

    if (error) throw new Error(`Failed to update chat: ${error.message}`);

    const messages =
      updates.messages ?? (await this.getChat(id, userId))?.messages ?? [];
    return {
      id: updated.id,
      title: updated.title,
      messages,
      createdAt: new Date(updated.created_at).getTime(),
    };
  }

  /** 删除：先删表再删 Storage（与 portfoliopilot deleteStrategy 一致） */
  async deleteChat(id: string, userId: string): Promise<void> {
    const { data: row } = await this.supabase
      .from(TABLE)
      .select("messages_url")
      .eq("id", id)
      .eq("user_id", userId)
      .single();

    const { error } = await this.supabase
      .from(TABLE)
      .delete()
      .eq("id", id)
      .eq("user_id", userId);

    if (error) throw new Error(`Failed to delete chat: ${error.message}`);
    if (row?.messages_url) {
      await this.storage.deleteFile(row.messages_url);
    }
  }
}

let chatService: ChatService | null = null;

export function getChatService(): ChatService {
  if (!chatService) {
    chatService = new ChatService();
  }
  return chatService;
}
