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
      <div className="reportSimilarCasesTitle">
        Similar historical windows (current + next two, for reference)
      </div>
      <div className="reportSimilarCasesGrid">
        {cases.map((c) => (
          <figure key={c.rank} className="reportSimilarCaseItem">
            <div className="reportSimilarCaseRow">
              {c.image_base64 ? (
                <img
                  src={`data:image/png;base64,${c.image_base64}`}
                  alt="Current window"
                  className="reportSimilarCaseImage"
                />
              ) : (
                <div className="reportSimilarCasePlaceholder">N/A</div>
              )}
              {c.image_next1_base64 ? (
                <img
                  src={`data:image/png;base64,${c.image_next1_base64}`}
                  alt="Next window 1"
                  className="reportSimilarCaseImage"
                />
              ) : (
                <div className="reportSimilarCasePlaceholder">—</div>
              )}
              {c.image_next2_base64 ? (
                <img
                  src={`data:image/png;base64,${c.image_next2_base64}`}
                  alt="Next window 2"
                  className="reportSimilarCaseImage"
                />
              ) : (
                <div className="reportSimilarCasePlaceholder">—</div>
              )}
            </div>
            <figcaption className="reportSimilarCaseCaption">
              {[c.symbol, c.start_date, c.end_date].filter(Boolean).join(" · ")}
              {c.symbol || c.start_date || c.end_date ? " " : ""}
              <span className="reportSimilarCaseSim">
                Similarity {(c.similarity * 100).toFixed(1)}%
              </span>
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
          Analysis report generated!
        </div>
      )}

      {messages.length === 0 && !loading && (
        <div className="welcome">
          <p>Please enter stock OHLC data, or upload a candlestick chart for analysis.</p>
          <p className="welcomeHint">
            Example format: symbol, date, open, high, low, close, volume (one row per bar)
          </p>
          <p className="welcomeHint">Or upload a candlestick image in PNG/JPG format.</p>
        </div>
      )}

      {messages.map((msg) => (
        <div key={msg.id} className={`messageRow ${msg.role}`}>
          <div className={`bubble ${msg.role}`}>
            {msg.type === "image" && msg.imageUrl && (
              <img src={msg.imageUrl} alt="Uploaded chart" className="bubbleImage" />
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

      {error && <div className="errorBar">{error}</div>}

      <div ref={messagesEndRef} />
    </div>
  );
}
