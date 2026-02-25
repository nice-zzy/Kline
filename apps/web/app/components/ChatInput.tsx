"use client";

import React from "react";

export type ChatInputProps = {
  inputText: string;
  setInputText: (v: string) => void;
  uploadFileName: string | null;
  setUploadFileName: (v: string | null) => void;
  fileInputRef: React.RefObject<HTMLInputElement | null>;
  loading: boolean;
  onSubmit: (e: React.FormEvent) => void;
  onFileChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
  inputAreaHeight: number | null;
  inputAreaMinHeight: number;
};

export function ChatInput({
  inputText,
  setInputText,
  uploadFileName,
  setUploadFileName,
  fileInputRef,
  loading,
  onSubmit,
  onFileChange,
  inputAreaHeight,
  inputAreaMinHeight,
}: ChatInputProps) {
  return (
    <div
      className="inputAreaWrapper"
      style={
        inputAreaHeight != null
          ? { height: inputAreaHeight, minHeight: inputAreaMinHeight }
          : { minHeight: inputAreaMinHeight }
      }
    >
      <form className="inputArea" onSubmit={onSubmit}>
        {uploadFileName && (
          <div className="inputFileTag">
            <span className="inputFileTagLabel">{uploadFileName}</span>
            <button
              type="button"
              className="inputFileTagRemove"
              onClick={() => {
                setUploadFileName(null);
                if (fileInputRef.current) fileInputRef.current.value = "";
              }}
              aria-label="移除文件"
            >
              ×
            </button>
          </div>
        )}
        <textarea
          className="inputText"
          placeholder="请输入股票 OHLC 数据，或上传 K 线图进行分析"
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              onSubmit(e);
            }
          }}
          rows={3}
          disabled={loading}
        />
        <div className="inputActions">
          <label className="uploadBtn">
            <input
              ref={fileInputRef}
              type="file"
              accept="image/png,image/jpeg,image/jpg"
              onChange={onFileChange}
            />
            <span className="uploadIcon">↑</span>
            <span className="uploadLabel">上传 K 线图 (PNG/JPG)</span>
          </label>
          <button type="submit" className="submitBtn" disabled={loading}>
            {loading ? "分析中…" : "发送"}
          </button>
        </div>
      </form>
    </div>
  );
}
