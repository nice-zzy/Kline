import { NextRequest, NextResponse } from "next/server";
import { requireAuth, createAuthError } from "@/lib/auth/server";
import { getStorageService } from "@/lib/services/storage";

/** 上传 K 线图到 Storage，返回可访问 URL（signed URL，1 年有效），用于写入 message.imageUrl */
export async function POST(req: NextRequest) {
  try {
    const user = await requireAuth();
    const formData = await req.formData();
    const file = formData.get("file") as File | null;
    if (!file || !(file instanceof File)) {
      return NextResponse.json({ error: "缺少文件" }, { status: 400 });
    }
    const contentType = file.type || "image/png";
    const storage = getStorageService(true);
    const fileId = crypto.randomUUID();
    const { url: path } = await storage.uploadFile(user.id, fileId, file, contentType);
    const signedUrl = await storage.getSignedUrl(path, 31536000);
    if (!signedUrl) {
      return NextResponse.json({ error: "生成访问链接失败" }, { status: 500 });
    }
    return NextResponse.json({ url: signedUrl });
  } catch (err) {
    if (err instanceof Error && err.message === "Authentication required") {
      return createAuthError();
    }
    console.error("POST /api/chats/upload-image failed:", err);
    return NextResponse.json(
      { error: err instanceof Error ? err.message : "Internal error" },
      { status: 500 }
    );
  }
}
