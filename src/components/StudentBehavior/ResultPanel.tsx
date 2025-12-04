import React from "react";

type FocusLevel = "낮음" | "중간" | "높음";

interface ResultPanelProps {
  currentAction: string;
  focusLevel: FocusLevel;
  getFocusColor: (level: FocusLevel) => string;
}

const ResultPanel: React.FC<ResultPanelProps> = ({
  currentAction,
  focusLevel,
  getFocusColor,
}) => {
  return (
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
          marginBottom: "12px",
        }}
      >
        실시간 감지 결과
      </h2>

      <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
          }}
        >
          <span style={{ color: "#6b7280" }}>현재 행동</span>
          <span style={{ fontWeight: 600, fontSize: "1rem" }}>
            {currentAction}
          </span>
        </div>

        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            marginTop: "4px",
          }}
        >
          <span style={{ color: "#6b7280" }}>집중도</span>
          <span
            style={{
              padding: "4px 10px",
              borderRadius: "999px",
              backgroundColor: getFocusColor(focusLevel),
              color: "#ffffff",
              fontSize: "0.9rem",
              fontWeight: 600,
            }}
          >
            {focusLevel}
          </span>
        </div>

        <div
          style={{
            marginTop: "10px",
            fontSize: "0.85rem",
            color: "#9ca3af",
          }}
        >
          * 카메라 포즈에서 추출된 스켈레톤을 기반으로 행동을 예측합니다.
        </div>
      </div>
    </div>
  );
};

export default ResultPanel;
