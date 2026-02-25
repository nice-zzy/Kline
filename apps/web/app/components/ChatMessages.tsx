"use client";

import React from "react";
import ReactMarkdown from "react-markdown";
import type { Message, SimilarCase } from "@/lib/types/chat";

export type ChatMessagesProps = {
  messages: Message[];
  loading: boolean;
  error: string | null;
  successBanner: boolean;
  messagesEndRef: React.RefObject<HTMLDivElement>;
  chatScrollRef?: React.RefObject<HTMLDivElement>;
};

function SimilarCasesBlock({ cases }: { cases: SimilarCase[] }) {
  if (!cases?.length) return null;
  return (
    <div className="reportSimilarCases">
      <div className="reportSimilarCasesTitle">相似历史 K 线（当前 + 往后两窗口，供参考）</div>
      <div className="reportSimilarCasesGrid">
        {cases.map((c) => (
          <figure key={c.rank} className="reportSimilarCaseItem">
            <div className="reportSimilarCaseRow">
              {c.image_base64 ? (
                <img
                  src={`data:image/png;base64,${c.image_base64}`}
                  alt="当前"
                  className="reportSimilarCaseImage"
                />
              ) : (
                <div className="reportSimilarCasePlaceholder">暂无</div>
              )}
              {c.image_next1_base64 ? (
                <img
                  src={`data:image/png;base64,${c.image_next1_base64}`}
                  alt="后续1"
                  className="reportSimilarCaseImage"
                />
              ) : (
                <div className="reportSimilarCasePlaceholder">—</div>
              )}
              {c.image_next2_base64 ? (
                <img
                  src={`data:image/png;base64,${c.image_next2_base64}`}
                  alt="后续2"
                  className="reportSimilarCaseImage"
                />
              ) : (
                <div className="reportSimilarCasePlaceholder">—</div>
              )}
            </div>
            <figcaption className="reportSimilarCaseCaption">
              {[c.symbol, c.start_date, c.end_date].filter(Boolean).join(" · ")}
              {c.symbol || c.start_date || c.end_date ? " " : ""}
              <span className="reportSimilarCaseSim">相似度 {(c.similarity * 100).toFixed(1)}%</span>
            </figcaption>
          </figure>
        ))}
      </div>
    </div>
  );
}

export function ChatMessages({
  messages,
  loading,
  error,
  successBanner,
  messagesEndRef,
  chatScrollRef,
}: ChatMessagesProps) {
  return (
    <div className="chatPanel" ref={chatScrollRef}>
      {successBanner && (
        <div className="successBanner" role="status">
          <span className="successIcon">✓</span>
          分析报告已生成！
        </div>
      )}

      {messages.length === 0 && !loading && (
        <div className="welcome">
          <p>请输入股票 OHLC 数据，或上传 K 线图进行分析</p>
          <p className="welcomeHint">
            格式示例：股票代码, 日期, 开盘, 最高, 最低, 收盘, 成交量（每行一条）
          </p>
          <p className="welcomeHint">或上传 PNG/JPG 格式的 K 线图</p>
        </div>
      )}

      {messages.map((msg) => (
        <div key={msg.id} className={`messageRow ${msg.role}`}>
          <div className={`bubble ${msg.role}`}>
            {msg.type === "image" && msg.imageUrl && (
              <img src={msg.imageUrl} alt="上传" className="bubbleImage" />
            )}
            <div className="bubbleContent">
              {msg.role === "assistant" ? (
                <div className="reportMarkdown">
                  <ReactMarkdown
                    components={{
                    p: ({ children }) => <p style={{ margin: "0.35em 0", lineHeight: 1.55 }}>{children}</p>,
                    h1: ({ children }) => <h3 style={{ margin: "0.55em 0 0.2em", fontSize: "1.05em", lineHeight: 1.35 }}>{children}</h3>,
                    h2: ({ children }) => <h4 style={{ margin: "0.5em 0 0.18em", fontSize: "0.98em", lineHeight: 1.35 }}>{children}</h4>,
                    h3: ({ children }) => <h4 style={{ margin: "0.45em 0 0.15em", fontSize: "0.95em", lineHeight: 1.35 }}>{children}</h4>,
                    ul: ({ children }) => <ul style={{ margin: "0.25em 0", paddingLeft: "1em" }}>{children}</ul>,
                    ol: ({ children }) => <ol style={{ margin: "0.25em 0", paddingLeft: "1em" }}>{children}</ol>,
                    li: ({ children }) => <li style={{ margin: "0.12em 0", lineHeight: 1.5 }}>{children}</li>,
                    strong: ({ children }) => <strong style={{ fontWeight: 600 }}>{children}</strong>,
                  }}
                >
                    {(msg.content || "")
                    .replace(/\r\n?|\r/g, "\n")
                    .replace(/\n\s*\n/g, "\n")}
                  </ReactMarkdown>
                </div>
              ) : (
                msg.content
              )}
            </div>
            {msg.role === "assistant" && msg.report?.similar_cases?.length ? (
              <SimilarCasesBlock cases={msg.report.similar_cases} />
            ) : null}
          </div>
        </div>
      ))}

      {loading && (
        <div className="messageRow assistant">
          <div className="bubble assistant">
            <div className="bubbleContent">正在生成分析报告…</div>
          </div>
        </div>
      )}

      {error && <div className="errorBar">{error}</div>}

      <div ref={messagesEndRef} />
    </div>
  );
}
