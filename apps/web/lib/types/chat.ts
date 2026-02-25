/** 图文报告中一条相似历史案例（含配图与标注） */
export interface SimilarCase {
  rank: number;
  symbol?: string | null;
  start_date: string;
  end_date: string;
  similarity: number;
  summary: string;
  /** 当前窗口 K 线图 */
  image_base64?: string | null;
  /** 往后第 1 个窗口（步长 3，与向量一致） */
  image_next1_base64?: string | null;
  /** 往后第 2 个窗口 */
  image_next2_base64?: string | null;
}

/** 分析报告附带的相似案例列表，用于图文展示 */
export interface ReportPayload {
  similar_cases: SimilarCase[];
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  type?: "text" | "image";
  imageUrl?: string;
  /** 助理消息的图文报告：相似历史 K 线图列表（注明股票与日期） */
  report?: ReportPayload;
}

export interface Chat {
  id: string;
  title: string;
  messages: Message[];
  createdAt: number;
}

export interface ChatListItem {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
}

export interface CreateChatBody {
  title?: string;
}

export interface UpdateChatBody {
  title?: string;
  messages?: Message[];
}
