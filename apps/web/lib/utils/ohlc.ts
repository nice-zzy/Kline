export type OHLC = [number, number, number, number];

export function parseOHLC(text: string): OHLC[] {
  const rows = text
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter(Boolean);
  const parsed: OHLC[] = [];
  for (const row of rows) {
    const parts = row.split(/,|\s+/).filter(Boolean);
    if (parts.length < 4) continue;
    const numericParts = parts.slice(-4).map(Number);
    const [O, H, L, C] = numericParts;
    if ([O, H, L, C].every((v) => Number.isFinite(v))) {
      parsed.push([O, H, L, C]);
    }
  }
  return parsed;
}

export function drawCandlesToCanvas(
  data: OHLC[],
  width = 560,
  height = 280
): HTMLCanvasElement {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx || data.length === 0) return canvas;

  ctx.fillStyle = "#0b0f19";
  ctx.fillRect(0, 0, width, height);

  const highs = data.map((d) => d[1]);
  const lows = data.map((d) => d[2]);
  const maxH = Math.max(...highs);
  const minL = Math.min(...lows);
  const pad = 10;

  const xStep = (width - pad * 2) / Math.max(1, data.length);
  function yScale(v: number) {
    const t = (v - minL) / Math.max(1e-9, maxH - minL);
    return height - pad - t * (height - pad * 2);
  }

  for (let i = 0; i < data.length; i++) {
    const [O, H, L, C] = data[i];
    const x = pad + i * xStep + xStep * 0.5;
    const yH = yScale(H);
    const yL = yScale(L);
    const yO = yScale(O);
    const yC = yScale(C);
    const up = C >= O;
    ctx.strokeStyle = up ? "#22d1ee" : "#ff5b6e";
    ctx.fillStyle = up ? "rgba(34,209,238,0.35)" : "rgba(255,91,110,0.35)";

    ctx.beginPath();
    ctx.moveTo(x, yH);
    ctx.lineTo(x, yL);
    ctx.stroke();

    const bw = Math.max(2, xStep * 0.5);
    const yTop = Math.min(yO, yC);
    const bh = Math.max(1, Math.abs(yC - yO));
    ctx.fillRect(x - bw / 2, yTop, bw, bh);
    ctx.strokeRect(x - bw / 2, yTop, bw, bh);
  }

  return canvas;
}
