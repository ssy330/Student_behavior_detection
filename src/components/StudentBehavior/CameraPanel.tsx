import React from "react";

interface CameraPanelProps {
  videoRef: React.RefObject<HTMLVideoElement>;
  canvasRef: React.RefObject<HTMLCanvasElement>;
  studySeconds: number;
  isCameraOn: boolean;
  onStartCamera: () => void;
  onStopCamera: () => void;
}

const formatTime = (totalSeconds: number) => {
  const h = Math.floor(totalSeconds / 3600)
    .toString()
    .padStart(2, "0");
  const m = Math.floor((totalSeconds % 3600) / 60)
    .toString()
    .padStart(2, "0");
  const s = Math.floor(totalSeconds % 60)
    .toString()
    .padStart(2, "0");
  return `${h}:${m}:${s}`;
};

const CameraPanel: React.FC<CameraPanelProps> = ({
  videoRef,
  canvasRef,
  studySeconds,
  isCameraOn,
  onStartCamera,
  onStopCamera,
}) => {
  // 🔥 카메라 끄기 버튼 눌렀을 때: 스켈레톤 캔버스도 같이 지우기
  const handleStopCamera = () => {
    // 1) 캔버스 클리어
    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext("2d");
      if (ctx) {
        ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      }
    }

    // 2) 실제 카메라 종료 로직 호출 (부모에서 내려온 함수)
    onStopCamera();
  };

  return (
    <section
      style={{
        flex: "1 1 380px",
        display: "flex",
        flexDirection: "column",
        gap: "12px",
      }}
    >
      <div
        style={{
          backgroundColor: "#ffffff",
          borderRadius: "16px",
          padding: "16px",
          boxShadow: "0 4px 12px rgba(0,0,0,0.05)",
        }}
      >
        <h2
          style={{
            fontSize: "1.1rem",
            fontWeight: 600,
            marginBottom: "8px",
          }}
        >
          카메라
        </h2>

        <div style={{ position: "relative", width: "100%" }}>
          <video
            ref={videoRef}
            style={{
              width: "100%",
              maxWidth: "720px",
              aspectRatio: "16 / 9",
              backgroundColor: "#000",
              borderRadius: "12px",
              objectFit: "cover",
              transform: "scaleX(-1)", // 좌우 반전
            }}
            playsInline
            muted
          />

          {/* 🔥 Skeleton overlay */}
          <canvas
            ref={canvasRef}
            style={{
              position: "absolute",
              left: 0,
              top: 0,
              width: "100%",
              height: "100%",
              pointerEvents: "none",
              borderRadius: "12px",
            }}
          />
        </div>

        <div
          style={{
            marginTop: "12px",
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            gap: "8px",
          }}
        >
          <div
            style={{
              fontFamily: "monospace",
              fontSize: "1.3rem",
              fontWeight: 700,
            }}
          >
            ⏱ 공부 시간: {formatTime(studySeconds)}
          </div>

          <div style={{ display: "flex", gap: "8px" }}>
            <button
              onClick={onStartCamera}
              disabled={isCameraOn}
              style={{
                padding: "6px 12px",
                borderRadius: "999px",
                backgroundColor: isCameraOn ? "#e5e7eb" : "#ffffff",
                cursor: isCameraOn ? "default" : "pointer",
              }}
            >
              카메라 켜기
            </button>

            <button
              onClick={handleStopCamera} // 🔥 여기서 래핑된 함수 사용
              disabled={!isCameraOn}
              style={{
                padding: "6px 12px",
                borderRadius: "999px",
                border: "1px solid #d1d5db",
                backgroundColor: !isCameraOn ? "#e5e7eb" : "#ffffff",
                cursor: !isCameraOn ? "default" : "pointer",
              }}
            >
              카메라 끄기
            </button>
          </div>
        </div>
      </div>
    </section>
  );
};

export default CameraPanel;
