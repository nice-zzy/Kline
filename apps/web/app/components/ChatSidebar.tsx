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
                  aria-label="Collapse sidebar"
                  title="Collapse"
                >
                  ◀
                </button>
              </div>
              <button
                type="button"
                className="btnNew"
                onClick={onNewChat}
                aria-label="New chat"
              >
                <span className="btnNewIcon">+</span>
                New chat
              </button>
            </>
          )}
          {sidebarCollapsed && (
            <button
              type="button"
              className="btnExpand"
              onClick={onToggleSidebar}
              aria-label="Expand sidebar"
              title="Expand"
            >
              ▶
            </button>
          )}
        </div>
        {!sidebarCollapsed && (
          <>
            <div className="historySection">
              <h2 className="historyTitle">History</h2>
              {historyList.length === 0 ? (
                <p className="historyEmpty">
                  No history yet. Send a message to start an analysis.
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
                        aria-label={`Delete chat: ${c.title}`}
                        title="Delete this chat"
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
                    aria-label="Sign out"
                  >
                    Sign out
                  </button>
                </>
              ) : (
                <Link href="/auth/login" className="sidebarLoginLink">
                  Log in
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
