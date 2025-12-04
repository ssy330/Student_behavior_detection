import React from "react";

type FocusLevel = "낮음" | "중간" | "높음";

export interface DetectionResult {
  timestamp: string; // 🔥 문자열로 저장하는 것이 안전함
  action: string;
  focus: FocusLevel;
}

interface HistoryListProps {
  history: DetectionResult[];
  getFocusColor: (level: FocusLevel) => string;
}

const HistoryList: React.FC<HistoryListProps> = ({
  history,
  getFocusColor,
}) => {
  return (
    <div
      style={{
        backgroundColor: "#ffffff",
        borderRadius: "16px",
        padding: "16px",
        boxShadow: "0 4px 12px rgba(0,0,0,0.05)",
        flex: 1,
        minHeight: 0,
        display: "flex",
        flexDirection: "column",
      }}
    >
      <h3
        style={{
          fontSize: "1rem",
          fontWeight: 600,
          marginBottom: "8px",
        }}
      >
        최근 감지 기록
      </h3>

      <div
        style={{
          flex: 1,
          minHeight: 0,
          overflowY: "auto",
          fontSize: "0.85rem",
        }}
      >
        {history.length === 0 ? (
          <p style={{ color: "#9ca3af" }}>아직 기록이 없습니다.</p>
        ) : (
          <ul style={{ listStyle: "none", padding: 0, margin: 0 }}>
            {history.map((item, idx) => (
              <li
                key={idx}
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  padding: "6px 0",
                  borderBottom:
                    idx === history.length - 1
                      ? "none"
                      : "1px solid #f3f4f6",
                }}
              >
                {/* 🔥 timestamp 문자열도 안전하게 처리 */}
                <span style={{ color: "#6b7280" }}>
                  {new Date(item.timestamp).toLocaleTimeString("ko-KR", {
                    hour: "2-digit",
                    minute: "2-digit",
                    second: "2-digit",
                  })}
                </span>

                <span>{item.action}</span>

                <span
                  style={{
                    padding: "2px 8px",
                    borderRadius: "999px",
                    backgroundColor: getFocusColor(item.focus),
                    color: "#ffffff",
                    fontSize: "0.75rem",
                  }}
                >
                  {item.focus}
                </span>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
};

export default HistoryList;
