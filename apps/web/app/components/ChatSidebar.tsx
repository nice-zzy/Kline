"use client";

import React from "react";
import Link from "next/link";
import type { Chat } from "@/lib/types/chat";

const SIDEBAR_COLLAPSED_WIDTH = 48;

type User = { id: string; email?: string | null };

export type ChatSidebarProps = {
  sidebarWidth: number;
  sidebarCollapsed: boolean;
  historyList: Chat[];
  currentChatId: string | null;
  user: User | null;
  onNewChat: () => void;
  onSelectHistory: (id: string) => void;
  onDeleteChat: (e: React.MouseEvent, id: string) => void;
  onToggleSidebar: () => void;
  onResizeStart: (e: React.MouseEvent) => void;
  signOut: () => void;
};

export function ChatSidebar({
  sidebarWidth,
  sidebarCollapsed,
  historyList,
  currentChatId,
  user,
  onNewChat,
  onSelectHistory,
  onDeleteChat,
  onToggleSidebar,
  onResizeStart,
  signOut,
}: ChatSidebarProps) {
  return (
    <>
      <aside
        className={`sidebar ${sidebarCollapsed ? "collapsed" : ""}`}
        style={{ width: sidebarCollapsed ? SIDEBAR_COLLAPSED_WIDTH : sidebarWidth }}
      >
        <div className="sidebarHeader">
          {!sidebarCollapsed && (
            <>
              <div className="sidebarHeaderRow">
                <h1 className="appName">K-line Analyst</h1>
                <button
                  type="button"
                  className="btnCollapse"
                  onClick={onToggleSidebar}
                  aria-label="收起侧栏"
                  title="收起"
                >
                  ◀
                </button>
              </div>
              <button
                type="button"
                className="btnNew"
                onClick={onNewChat}
                aria-label="新建对话"
              >
                <span className="btnNewIcon">+</span>
                新建对话
              </button>
            </>
          )}
          {sidebarCollapsed && (
            <button
              type="button"
              className="btnExpand"
              onClick={onToggleSidebar}
              aria-label="展开侧栏"
              title="展开"
            >
              ▶
            </button>
          )}
        </div>
        {!sidebarCollapsed && (
          <>
            <div className="historySection">
              <h2 className="historyTitle">历史记录</h2>
              {historyList.length === 0 ? (
                <p className="historyEmpty">
                  暂无历史记录，发送一条消息开始分析
                </p>
              ) : (
                <ul className="historyList">
                  {historyList.map((c) => (
                    <li key={c.id} className="historyListItem">
                      <button
                        type="button"
                        className={`historyItem ${currentChatId === c.id ? "active" : ""}`}
                        onClick={() => onSelectHistory(c.id)}
                      >
                        {c.title}
                      </button>
                      <button
                        type="button"
                        className="historyItemDelete"
                        onClick={(e) => onDeleteChat(e, c.id)}
                        aria-label={`删除对话：${c.title}`}
                        title="删除此对话"
                      >
                        ×
                      </button>
                    </li>
                  ))}
                </ul>
              )}
            </div>
            <div className="sidebarFooter">
              {user ? (
                <>
                  <span
                    className="sidebarUserEmail"
                    title={user.email ?? undefined}
                  >
                    {user.email ?? user.id.slice(0, 8)}
                  </span>
                  <button
                    type="button"
                    className="btnSignOut"
                    onClick={() => signOut()}
                    aria-label="退出登录"
                  >
                    退出
                  </button>
                </>
              ) : (
                <Link href="/auth/login" className="sidebarLoginLink">
                  登录
                </Link>
              )}
            </div>
          </>
        )}
      </aside>

      {!sidebarCollapsed && (
        <div
          className="resizeHandle"
          onMouseDown={onResizeStart}
          role="separator"
          aria-orientation="vertical"
          title="拖拽调节宽度"
        />
      )}
    </>
  );
}
