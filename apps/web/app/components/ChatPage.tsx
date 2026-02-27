"use client";

import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { useAuth } from "@/contexts/AuthContext";
import type { Message, Chat } from "@/lib/types/chat";
import { parseOHLC, drawCandlesToCanvas } from "@/lib/utils/ohlc";
import { genId, formatChatTitle } from "@/lib/utils/chat-helpers";
import { ChatSidebar } from "./ChatSidebar";
import { ChatMessages } from "./ChatMessages";
import { ChatInput } from "./ChatInput";

const STORAGE_KEY_LOCAL = "kline-chats-local";
const SIDEBAR_STORAGE_KEY = "kline-sidebar";
const INPUT_AREA_STORAGE_KEY = "kline-input-area-height";
const API_BASE = process.env.NEXT_PUBLIC_API_URL || "";
const ANALYZE_ENDPOINT = API_BASE
  ? `${API_BASE.replace(/\/$/, "")}/analyze`
  : "/api/analyze";
const CHATS_API = "/api/chats";
const CHATS_UPLOAD_IMAGE = "/api/chats/upload-image";

const SIDEBAR_WIDTH_MIN = 200;
const SIDEBAR_WIDTH_MAX = 480;
const SIDEBAR_WIDTH_DEFAULT = 260;
const INPUT_AREA_HEIGHT_MIN = 120;
const INPUT_AREA_HEIGHT_MAX = 400;
const INPUT_AREA_HEIGHT_DEFAULT = 180;

export default function ChatPage() {
  const { user, loading: authLoading, signOut } = useAuth();
  const [chats, setChats] = useState<Record<string, Chat>>({});
  const [currentChatId, setCurrentChatId] = useState<string | null>(null);
  const [inputText, setInputText] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [successBanner, setSuccessBanner] = useState(false);
  const [uploadFileName, setUploadFileName] = useState<string | null>(null);
  const [sidebarWidth, setSidebarWidth] = useState(SIDEBAR_WIDTH_DEFAULT);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [resizing, setResizing] = useState(false);
  const [inputAreaHeight, setInputAreaHeight] = useState<number | null>(null);
  const [inputResizing, setInputResizing] = useState(false);
  const [chatsLoading, setChatsLoading] = useState(false);

  const dirtyChatIdRef = useRef<string | null>(null);
  const serverChatIdsRef = useRef<Set<string>>(new Set());
  const chatsRef = useRef<Record<string, Chat>>({});
  const resizeStartX = useRef(0);
  const resizeStartWidth = useRef(0);
  const resizeStartY = useRef(0);
  const resizeStartInputHeight = useRef(0);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const chatScrollRef = useRef<HTMLDivElement>(null);
  const inputAreaRef = useRef<HTMLDivElement>(null);

  const currentChat = currentChatId ? chats[currentChatId] : null;
  const messages = currentChat?.messages ?? [];
  const historyList = useMemo(
    () =>
      Object.values(chats)
        .filter((c) => (user ? true : c.messages.length > 0))
        .sort((a, b) => b.createdAt - a.createdAt),
    [chats, user]
  );

  const lastFetchedUserIdRef = useRef<string | null>(null);
  useEffect(() => {
    if (authLoading) return;
    if (user) {
      if (lastFetchedUserIdRef.current === user.id) return;
      lastFetchedUserIdRef.current = user.id;
      loadedChatIdsRef.current.clear();
      setChatsLoading(true);
      fetch(CHATS_API, { credentials: "include" })
        .then((r) =>
          r.ok ? r.json() : Promise.reject(new Error("Unauthorized"))
        )
        .then((data) => {
          const list = data.chats ?? [];
          const fromList: Record<string, Chat> = {};
          list.forEach(
            (row: {
              id: string;
              title: string;
              created_at: string;
              updated_at: string;
            }) => {
              fromList[row.id] = {
                id: row.id,
                title: row.title,
                messages: [],
                createdAt: new Date(
                  row.updated_at || row.created_at
                ).getTime(),
              };
            }
          );
          setChats((prev) => {
            const merged = { ...fromList };
            for (const id of Object.keys(prev)) {
              if (
                merged[id] &&
                prev[id].messages &&
                prev[id].messages.length > 0
              ) {
                merged[id] = {
                  ...merged[id],
                  messages: prev[id].messages,
                  title: prev[id].title || merged[id].title,
                };
              }
            }
            return merged;
          });
          const ids = Object.keys(fromList);
          serverChatIdsRef.current = new Set(ids);
          setCurrentChatId((prev) => {
            if (ids.length === 0) return null;
            if (prev && fromList[prev]) return prev;
            return ids[0];
          });
        })
        .catch(() => setChats({}))
        .finally(() => setChatsLoading(false));
    } else {
      lastFetchedUserIdRef.current = null;
      loadedChatIdsRef.current.clear();
      try {
        const raw = localStorage.getItem(STORAGE_KEY_LOCAL);
        if (raw) {
          const parsed = JSON.parse(raw) as Record<string, Chat>;
          setChats(parsed);
          const ids = Object.keys(parsed);
          if (ids.length > 0) setCurrentChatId(ids[0]);
        }
      } catch {
        // ignore
      }
    }
  }, [user, authLoading]);

  const loadedChatIdsRef = useRef<Set<string>>(new Set());
  const justCreatedChatIdRef = useRef<string | null>(null);
  const createdInSessionRef = useRef<Set<string>>(new Set());
  useEffect(() => {
    if (!user || !currentChatId || chatsLoading) return;
    if (!serverChatIdsRef.current.has(currentChatId)) return;
    if (createdInSessionRef.current.has(currentChatId)) return;
    if (currentChatId === justCreatedChatIdRef.current) return;
    const cur = chats[currentChatId];
    if (!cur) return;
    if (cur.messages.length || loadedChatIdsRef.current.has(currentChatId))
      return;
    loadedChatIdsRef.current.add(currentChatId);
    const idToFetch = currentChatId;
    fetch(`${CHATS_API}/${currentChatId}`, { credentials: "include" })
      .then((r) => {
        if (!r.ok) {
          if (r.status === 404) {
            serverChatIdsRef.current.delete(idToFetch);
            loadedChatIdsRef.current.add(idToFetch);
            setCurrentChatId((prev) => {
              if (prev !== idToFetch) return prev;
              const allIds = Object.keys(chatsRef.current);
              return allIds.filter((k) => k !== idToFetch)[0] ?? null;
            });
          }
          return null;
        }
        return r.json();
      })
      .then((chat) => {
        if (chat) {
          setChats((prev) => ({
            ...prev,
            [idToFetch]: {
              id: chat.id,
              title: chat.title,
              messages: chat.messages ?? [],
              createdAt: chat.createdAt ?? Date.now(),
            },
          }));
        }
      })
      .catch(() => {});
  }, [user, currentChatId, chats, chatsLoading]);

  useEffect(() => {
    try {
      const raw = sessionStorage.getItem(SIDEBAR_STORAGE_KEY);
      if (raw) {
        const { w, collapsed } = JSON.parse(raw) as {
          w?: number;
          collapsed?: boolean;
        };
        if (
          typeof w === "number" &&
          w >= SIDEBAR_WIDTH_MIN &&
          w <= SIDEBAR_WIDTH_MAX
        )
          setSidebarWidth(w);
        if (typeof collapsed === "boolean") setSidebarCollapsed(collapsed);
      }
      const hRaw = sessionStorage.getItem(INPUT_AREA_STORAGE_KEY);
      if (hRaw != null) {
        const h = Number(hRaw);
        if (
          Number.isFinite(h) &&
          h >= INPUT_AREA_HEIGHT_MIN &&
          h <= INPUT_AREA_HEIGHT_MAX
        )
          setInputAreaHeight(h);
      }
    } catch {
      // ignore
    }
  }, []);

  const prevResizing = useRef(false);
  useEffect(() => {
    if (!resizing) return;
    const onMove = (e: MouseEvent) => {
      const delta = e.clientX - resizeStartX.current;
      const next = Math.max(
        SIDEBAR_WIDTH_MIN,
        Math.min(SIDEBAR_WIDTH_MAX, resizeStartWidth.current + delta)
      );
      setSidebarWidth(next);
    };
    const onUp = () => setResizing(false);
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
  }, [resizing]);

  useEffect(() => {
    if (prevResizing.current && !resizing) {
      try {
        sessionStorage.setItem(
          SIDEBAR_STORAGE_KEY,
          JSON.stringify({ w: sidebarWidth, collapsed: sidebarCollapsed })
        );
      } catch {
        // ignore
      }
    }
    prevResizing.current = resizing;
  }, [resizing, sidebarWidth, sidebarCollapsed]);

  useEffect(() => {
    try {
      sessionStorage.setItem(
        SIDEBAR_STORAGE_KEY,
        JSON.stringify({ w: sidebarWidth, collapsed: sidebarCollapsed })
      );
    } catch {
      // ignore
    }
  }, [sidebarCollapsed]);

  const prevInputResizing = useRef(false);
  useEffect(() => {
    if (!inputResizing) return;
    const onMove = (e: MouseEvent) => {
      const delta = resizeStartY.current - e.clientY;
      const current = resizeStartInputHeight.current;
      const next = Math.max(
        INPUT_AREA_HEIGHT_MIN,
        Math.min(INPUT_AREA_HEIGHT_MAX, current + delta)
      );
      setInputAreaHeight(next);
    };
    const onUp = () => setInputResizing(false);
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
  }, [inputResizing]);

  useEffect(() => {
    if (
      prevInputResizing.current &&
      !inputResizing &&
      inputAreaHeight != null
    ) {
      try {
        sessionStorage.setItem(
          INPUT_AREA_STORAGE_KEY,
          String(inputAreaHeight)
        );
      } catch {
        // ignore
      }
    }
    prevInputResizing.current = inputResizing;
  }, [inputResizing, inputAreaHeight]);

  useEffect(() => {
    if (user || Object.keys(chats).length === 0) return;
    try {
      localStorage.setItem(STORAGE_KEY_LOCAL, JSON.stringify(chats));
    } catch {
      // ignore
    }
  }, [user, chats]);

  useEffect(() => {
    chatsRef.current = chats;
  }, [chats]);

  useEffect(() => {
    const id = dirtyChatIdRef.current;
    if (!user || !id) return;
    if (!serverChatIdsRef.current.has(id)) {
      dirtyChatIdRef.current = null;
      return;
    }
    const cur = chats[id];
    if (!cur || cur.messages.length === 0) return;
    dirtyChatIdRef.current = null;
    const payload = { title: cur.title, messages: cur.messages };
    fetch(`${CHATS_API}/${id}`, {
      method: "PATCH",
      credentials: "include",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }).catch(() => {});
  }, [user, chats]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    if (successBanner) {
      const t = setTimeout(() => setSuccessBanner(false), 4000);
      return () => clearTimeout(t);
    }
  }, [successBanner]);

  const ensureCurrentChat = useCallback(async () => {
    if (currentChatId && chats[currentChatId]) return currentChatId;
    if (user) {
      const res = await fetch(CHATS_API, {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title: "New chat" }),
      });
      if (!res.ok) {
        if (res.status === 401) {
          await signOut();
          setError("Session expired. Please sign in again.");
        } else {
          const msg = await res.json().catch(() => ({}));
          const detail = (msg as { error?: string })?.error ?? `HTTP ${res.status}`;
          setError(`Failed to create chat: ${detail}`);
        }
        throw new Error(
          res.status === 401 ? "Session expired. Please sign in again." : "Failed to create chat."
        );
      }
      const chat = (await res.json()) as Chat;
      serverChatIdsRef.current.add(chat.id);
      createdInSessionRef.current.add(chat.id);
      justCreatedChatIdRef.current = chat.id;
      const createdId = chat.id;
      setTimeout(() => {
        if (justCreatedChatIdRef.current === createdId)
          justCreatedChatIdRef.current = null;
      }, 3000);
      setChats((prev) => ({ ...prev, [chat.id]: chat }));
      setCurrentChatId(chat.id);
      loadedChatIdsRef.current.add(chat.id);
      return chat.id;
    }
    const id = genId();
    setChats((prev) => ({
      ...prev,
      [id]: { id, title: "New chat", messages: [], createdAt: Date.now() },
    }));
    setCurrentChatId(id);
    return id;
  }, [currentChatId, chats, user, signOut]);

  const addMessage = useCallback(
    (chatId: string, msg: Message) => {
      if (user) dirtyChatIdRef.current = chatId;
      setChats((prev) => {
        const chat = prev[chatId];
        if (!chat) return prev;
        const title =
          chat.messages.length === 0 && msg.role === "user"
            ? formatChatTitle(msg.content, msg.type === "image")
            : chat.title;
        return {
          ...prev,
          [chatId]: {
            ...chat,
            title,
            messages: [...chat.messages, msg],
          },
        };
      });
    },
    [user]
  );

  const syncChatToServer = useCallback(
    (chatId: string, messages: Message[], title: string) => {
      if (
        !user ||
        !serverChatIdsRef.current.has(chatId) ||
        messages.length === 0
      )
        return;
      fetch(`${CHATS_API}/${chatId}`, {
        method: "PATCH",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title, messages }),
      }).catch(() => {});
    },
    [user]
  );

  const sendFile = useCallback(
    async (file: File, extraMessage?: string) => {
      const chatId = await ensureCurrentChat();
      const currentMessages = chatsRef.current[chatId]?.messages ?? [];
      const newTitle =
        currentMessages.length === 0
          ? formatChatTitle(`Uploaded K-line chart: ${file.name}`, true)
          : (chatsRef.current[chatId]?.title ?? "New chat");

      let imageUrl: string | undefined;
      if (user) {
        try {
          const uploadForm = new FormData();
          uploadForm.append("file", file);
          const uploadRes = await fetch(CHATS_UPLOAD_IMAGE, {
            method: "POST",
            credentials: "include",
            body: uploadForm,
          });
          if (uploadRes.ok) {
            const { url } = await uploadRes.json();
            imageUrl = typeof url === "string" ? url : undefined;
          }
        } catch {
          // 上传图片失败仍继续发分析请求
        }
      }
      const userContent =
        extraMessage?.trim() ?
          `Uploaded K-line chart: ${file.name}\n${extraMessage.trim()}`
        : `Uploaded K-line chart: ${file.name}`;
      const userMsg: Message = {
        id: genId(),
        role: "user",
        content: userContent,
        type: "image",
        ...(imageUrl && { imageUrl }),
      };
      addMessage(chatId, userMsg);
      const retrievingMsg: Message = {
        id: genId(),
        role: "assistant",
        content: "Retrieving similar historical K-lines…",
      };
      addMessage(chatId, retrievingMsg);
      setLoading(true);
      setError(null);
      setUploadFileName(null);

      const formData = new FormData();
      formData.append("file", file);
      if (typeof extraMessage === "string" && extraMessage.trim()) {
        formData.append("message", extraMessage.trim());
      }
      const history = (currentMessages as Message[])
        .slice(-20)
        .map((m) => ({ role: m.role, content: m.content }));
      formData.append("history", JSON.stringify(history));

      try {
        const resp = await fetch(ANALYZE_ENDPOINT, {
          method: "POST",
          body: formData,
        });
        if (!resp.ok) throw new Error(`Request failed: ${resp.status}`);
        const data = await resp.json();
        const text = data?.text ?? "No analysis content returned.";
        const ok = data?.success !== false;
        const similarCases =
          (data?.similar_cases?.length && data.similar_cases) ||
          (data?.report?.similar_cases?.length && data.report.similar_cases) ||
          [];

        const newMessages: Message[] = [...currentMessages, userMsg, retrievingMsg];

        if (similarCases.length) {
          const similarLines = similarCases.map((c, idx) => {
            const parts: string[] = [];
            if (c.symbol) parts.push(c.symbol);
            if (c.start_date || c.end_date) {
              const range = `${c.start_date || ""} ~ ${c.end_date || ""}`.trim();
              if (range) parts.push(range);
            }
            if (typeof c.similarity === "number") {
              parts.push(`similarity ${(c.similarity * 100).toFixed(2)}%`);
            }
            const lineBody = parts.join(" | ");
            return `${idx + 1}. ${lineBody}`;
          });
          const similarText =
            similarLines.length > 0 ? `\n\n${similarLines.join("\n")}` : "";

          const similarMsg: Message = {
            id: genId(),
            role: "assistant",
            content:
              "Found the following similar historical K-lines (current window + next two, for reference):" +
              similarText,
            report: { similar_cases: similarCases },
          };
          addMessage(chatId, similarMsg);
          newMessages.push(similarMsg);
        }

        const thinkingMsg: Message = {
          id: genId(),
          role: "assistant",
          content: "Generating analysis…",
        };
        addMessage(chatId, thinkingMsg);
        newMessages.push(thinkingMsg);

        const analysisMsg: Message = {
          id: genId(),
          role: "assistant",
          content: text,
        };
        addMessage(chatId, analysisMsg);
        newMessages.push(analysisMsg);

        if (ok) setSuccessBanner(true);
        else setError(data?.message ?? "Analysis failed.");
        syncChatToServer(
          chatId,
          newMessages,
          newTitle
        );
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : "Request error";
        setError(msg);
        const assistantMsg: Message = {
          id: genId(),
          role: "assistant",
          content: `Analysis failed: ${msg}`,
        };
        addMessage(chatId, assistantMsg);
        syncChatToServer(
          chatId,
          [...currentMessages, userMsg, retrievingMsg, assistantMsg],
          newTitle
        );
      } finally {
        setLoading(false);
      }
    },
    [ensureCurrentChat, addMessage, syncChatToServer, user]
  );

  const sendMessageOnly = useCallback(
    async (message: string) => {
      const chatId = await ensureCurrentChat();
      const currentMessages = chatsRef.current[chatId]?.messages ?? [];
      const newTitle =
        currentMessages.length === 0
          ? formatChatTitle(message.slice(0, 30), false)
          : (chatsRef.current[chatId]?.title ?? "New chat");

      const userMsg: Message = {
        id: genId(),
        role: "user",
        content: message,
      };
      addMessage(chatId, userMsg);
      setLoading(true);
      setError(null);

      const formData = new FormData();
      formData.append("message", message);
      const history = (currentMessages as Message[])
        .slice(-20)
        .map((m) => ({ role: m.role, content: m.content }));
      formData.append("history", JSON.stringify(history));

      try {
        const resp = await fetch(ANALYZE_ENDPOINT, {
          method: "POST",
          body: formData,
        });
        if (!resp.ok) throw new Error(`Request failed: ${resp.status}`);
        const data = await resp.json();
        const text = data?.text ?? "No analysis content returned.";
        const ok = data?.success !== false;
        const assistantMsg: Message = {
          id: genId(),
          role: "assistant",
          content: text,
          report:
            (data?.similar_cases?.length && {
              similar_cases: data.similar_cases,
            }) ||
            (data?.report?.similar_cases?.length && {
              similar_cases: data.report.similar_cases,
            }) ||
            undefined,
        };
        addMessage(chatId, assistantMsg);
        if (ok) setSuccessBanner(true);
        else setError(data?.message ?? "Analysis failed.");
        syncChatToServer(
          chatId,
          [...currentMessages, userMsg, assistantMsg],
          newTitle
        );
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : "Request error";
        setError(msg);
        const assistantMsg: Message = {
          id: genId(),
          role: "assistant",
          content: `Analysis failed: ${msg}`,
        };
        addMessage(chatId, assistantMsg);
        syncChatToServer(
          chatId,
          [...currentMessages, userMsg, assistantMsg],
          newTitle
        );
      } finally {
        setLoading(false);
      }
    },
    [ensureCurrentChat, addMessage, syncChatToServer, user]
  );

  const handleSubmit = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault();
      const text = inputText.trim();
      const ohlcData = parseOHLC(text);

      if (ohlcData.length > 0) {
        const canvas = drawCandlesToCanvas(ohlcData);
        const blob: Blob | null = await new Promise((resolve) =>
          canvas.toBlob((b) => resolve(b), "image/png")
        );
        if (!blob) {
          setError("Failed to convert OHLC data to image.");
          return;
        }
        const file = new File([blob], "ohlc.png", { type: "image/png" });
        setInputText("");
        await sendFile(file, text);
        return;
      }

      const input = fileInputRef.current;
      if (input?.files?.length) {
        await sendFile(input.files[0], text || undefined);
        input.value = "";
        setInputText("");
        return;
      }

      if (text) {
        await sendMessageOnly(text);
        setInputText("");
        return;
      }

      setError("Please enter a question, OHLC data, or upload a K-line chart.");
    },
    [inputText, sendFile, sendMessageOnly]
  );

  const handleNewChat = useCallback(async () => {
    setError(null);
    setSuccessBanner(false);
    if (user) {
      try {
        const res = await fetch(CHATS_API, {
          method: "POST",
          credentials: "include",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ title: "New chat" }),
        });
        if (!res.ok) {
          const msg = await res.json().catch(() => ({}));
          const detail = (msg as { error?: string })?.error ?? `HTTP ${res.status}`;
          setError(`Failed to create chat: ${detail}`);
          return;
        }
        const chat = (await res.json()) as Chat;
        serverChatIdsRef.current.add(chat.id);
        setChats((prev) => ({ ...prev, [chat.id]: chat }));
        setCurrentChatId(chat.id);
        loadedChatIdsRef.current.add(chat.id);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to create chat.");
      }
      return;
    }
    const id = genId();
    setChats((prev) => ({
      ...prev,
      [id]: { id, title: "New chat", messages: [], createdAt: Date.now() },
    }));
    setCurrentChatId(id);
  }, [user]);

  const handleSelectHistory = useCallback((id: string) => {
    setCurrentChatId(id);
    setError(null);
  }, []);

  const handleDeleteChat = useCallback(
    async (e: React.MouseEvent, id: string) => {
      e.preventDefault();
      e.stopPropagation();
      if (user) {
        try {
          const res = await fetch(`${CHATS_API}/${id}`, {
            method: "DELETE",
            credentials: "include",
          });
          if (!res.ok && res.status !== 404) return;
        } catch {
          return;
        }
        serverChatIdsRef.current.delete(id);
        loadedChatIdsRef.current.delete(id);
      }
      setChats((prev) => {
        const next = { ...prev };
        delete next[id];
        return next;
      });
      setCurrentChatId((prev) => {
        if (prev !== id) return prev;
        const remaining = Object.keys(chatsRef.current).filter((k) => k !== id);
        return remaining[0] ?? null;
      });
      setError(null);
    },
    [user]
  );

  const handleFileChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const f = e.target.files?.[0];
      if (f) setUploadFileName(f.name);
    },
    []
  );

  const handleToggleSidebar = useCallback(() => {
    setSidebarCollapsed((prev) => !prev);
  }, []);

  const handleResizeStart = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    resizeStartX.current = e.clientX;
    resizeStartWidth.current = sidebarWidth;
    setResizing(true);
  }, [sidebarWidth]);

  const handleInputResizeStart = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    resizeStartY.current = e.clientY;
    resizeStartInputHeight.current =
      inputAreaHeight ?? INPUT_AREA_HEIGHT_DEFAULT;
    setInputResizing(true);
  }, [inputAreaHeight]);

  return (
    <div
      className={`chatLayout ${resizing ? "resizing" : ""} ${inputResizing ? "inputResizing" : ""}`}
    >
      <ChatSidebar
        sidebarWidth={sidebarWidth}
        sidebarCollapsed={sidebarCollapsed}
        historyList={historyList}
        currentChatId={currentChatId}
        user={user}
        onNewChat={handleNewChat}
        onSelectHistory={handleSelectHistory}
        onDeleteChat={handleDeleteChat}
        onToggleSidebar={handleToggleSidebar}
        onResizeStart={handleResizeStart}
        signOut={signOut}
      />

      <main className="mainArea">
        <ChatMessages
          messages={messages}
          loading={loading}
          error={error}
          successBanner={successBanner}
          messagesEndRef={messagesEndRef}
          chatScrollRef={chatScrollRef}
        />

        <div
          className="inputResizeHandle"
          onMouseDown={handleInputResizeStart}
          role="separator"
          aria-orientation="horizontal"
          title="拖拽调节输入区高度"
        />
        <ChatInput
          inputText={inputText}
          setInputText={setInputText}
          uploadFileName={uploadFileName}
          setUploadFileName={setUploadFileName}
          fileInputRef={fileInputRef}
          loading={loading}
          onSubmit={handleSubmit}
          onFileChange={handleFileChange}
          inputAreaHeight={inputAreaHeight}
          inputAreaMinHeight={INPUT_AREA_HEIGHT_MIN}
        />
      </main>
    </div>
  );
}
