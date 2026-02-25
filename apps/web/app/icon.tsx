import { ImageResponse } from "next/og";

export const size = { width: 32, height: 32 };
export const contentType = "image/png";

export default function Icon() {
  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          background: "linear-gradient(135deg, #0b0f19 0%, #1a2332 100%)",
          borderRadius: 6,
          fontSize: 18,
        }}
      >
        K
      </div>
    ),
    { ...size }
  );
}
