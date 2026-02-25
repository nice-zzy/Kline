/**
 * Storage 服务：与 portfoliopilot-main lib/services/storage.ts 一致
 * - 服务端用 Service Role，浏览器用 Anon
 * - 路径格式：{userId}/{folder}/{id}.json，folder 本项目中为 chats
 */
import { createServiceRoleClient } from "@/lib/supabase/server";
import { createClient as createBrowserClient } from "@/lib/supabase/client";

const BUCKET_NAME = "user-data";

export interface StorageUploadResult {
  url: string; // Storage 路径，非完整 URL
}

export class StorageService {
  private supabase: ReturnType<typeof createServiceRoleClient> | ReturnType<typeof createBrowserClient>;

  constructor(isServer = false) {
    if (isServer) {
      this.supabase = createServiceRoleClient();
    } else {
      this.supabase = createBrowserClient();
    }
  }

  async uploadJson(
    userId: string,
    folder: "chats",
    id: string,
    data: Record<string, unknown>
  ): Promise<StorageUploadResult> {
    const path = `${userId}/${folder}/${id}.json`;
    const jsonString = JSON.stringify(data);
    const blob = new Blob([jsonString], { type: "application/json" });

    const { error } = await this.supabase.storage
      .from(BUCKET_NAME)
      .upload(path, blob, {
        contentType: "application/json",
        upsert: true,
      });

    if (error) {
      console.error("Storage upload error:", error);
      throw new Error(`Failed to upload: ${error.message}`);
    }
    return { url: path };
  }

  async downloadJson<T = Record<string, unknown>>(path: string): Promise<T | null> {
    if (!path) return null;
    const { data, error } = await this.supabase.storage
      .from(BUCKET_NAME)
      .download(path);
    if (error) {
      if (
        error.message.includes("not found") ||
        error.message.includes("Object not found")
      ) {
        return null;
      }
      throw new Error(`Failed to download: ${error.message}`);
    }
    if (!data) return null;
    const text = await data.text();
    return JSON.parse(text) as T;
  }

  async deleteFile(path: string): Promise<void> {
    if (!path) return;
    const { error } = await this.supabase.storage
      .from(BUCKET_NAME)
      .remove([path]);
    if (error && !error.message.includes("not found")) {
      throw new Error(`Failed to delete: ${error.message}`);
    }
  }

  /**
   * 上传图片等二进制文件到 Storage（对话内 K 线图等）
   * 路径：{userId}/images/{fileId}.{ext}
   */
  async uploadFile(
    userId: string,
    fileId: string,
    blob: Blob | File,
    contentType: string
  ): Promise<StorageUploadResult> {
    const ext = contentType.startsWith("image/") ? contentType.replace("image/", "") : "bin";
    const safeExt = ["png", "jpeg", "jpg", "gif", "webp"].includes(ext) ? ext : "png";
    const path = `${userId}/images/${fileId}.${safeExt}`;
    const { error } = await this.supabase.storage
      .from(BUCKET_NAME)
      .upload(path, blob, {
        contentType: contentType || "image/png",
        upsert: true,
      });
    if (error) {
      console.error("Storage upload file error:", error);
      throw new Error(`Failed to upload file: ${error.message}`);
    }
    return { url: path };
  }

  /**
   * 获取私有文件的临时访问 URL（1 年有效，用于写入 message.imageUrl）
   * 仅服务端调用（Service Role）
   */
  getSignedUrl(path: string, expiresIn = 31536000): Promise<string | null> {
    if (!path) return Promise.resolve(null);
    return this.supabase.storage
      .from(BUCKET_NAME)
      .createSignedUrl(path, expiresIn)
      .then(({ data, error }) => {
        if (error) {
          console.error("Storage signed URL error:", error);
          return null;
        }
        return data?.signedUrl ?? null;
      });
  }
}

let serverStorage: StorageService | null = null;
let browserStorage: StorageService | null = null;

export function getStorageService(isServer = false): StorageService {
  if (isServer) {
    if (!serverStorage) serverStorage = new StorageService(true);
    return serverStorage;
  }
  if (!browserStorage) browserStorage = new StorageService(false);
  return browserStorage;
}
