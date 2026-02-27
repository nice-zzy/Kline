export function genId(): string {
  return Math.random().toString(36).slice(2, 12);
}

export function formatChatTitle(content: string, isImage: boolean): string {
  if (isImage)
    return `K-line chart ${new Date().toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit" })}`;
  const firstLine = content.split(/\r?\n/)[0]?.trim() || "";
  if (firstLine.length > 20) return firstLine.slice(0, 20) + "…";
  return firstLine || "OHLC data";
}
