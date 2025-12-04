import React, { useEffect, useRef, useState } from "react";
import CameraPanel from "../components/StudentBehavior/CameraPanel";
import ResultPanel from "../components/StudentBehavior/ResultPanel";
import HistoryList, {
  DetectionResult,
} from "../components/StudentBehavior/HistoryList";

import { useCamera } from "../lib/pose/useCamera";
import { useSkeletonBuffer } from "../lib/pose/useSkeletonBuffer";
import { useMediapipePose } from "../lib/pose/useMediapipePose";
import { buildSkeleton } from "../lib/pose/buildSkeleton";
import { predictAction, ActionResponse } from "../api/predictAction";

type FocusLevel = "낮음" | "중간" | "높음";

// 🔹 행동 라벨 → 집중도 매핑
const mapActionToFocus = (actionLabel: string): FocusLevel => {
  if (actionLabel.includes("공부")) return "높음";
  if (actionLabel.includes("휴대폰") || actionLabel.includes("딴짓"))
    return "낮음";
  return "중간";
};

// 🔹 집중도 색상
const getFocusColor = (level: FocusLevel) => {
  switch (level) {
    case "낮음":
      return "#f97373";
    case "중간":
      return "#fbbf24";
    case "높음":
      return "#22c55e";
    default:
      return "#e5e7eb";
  }
};

const StudentBehaviorPage: React.FC = () => {
  // 카메라 제어
  const { videoRef, isCameraOn, startCamera, stopCamera } = useCamera();
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  // ST-GCN 입력 버퍼
  const { bufferRef, pushFrame } = useSkeletonBuffer(60);

  // Mediapipe Pose
  useMediapipePose({
    videoRef,
    isCameraOn,
    onFrame: pushFrame,
    canvasRef,
    visualize: true,
  });

  // 모델 상태
  const [currentAction, setCurrentAction] = useState<string>("대기 중");
  const [currentActionId, setCurrentActionId] = useState<number | null>(null);
  const [focusLevel, setFocusLevel] = useState<FocusLevel>("중간");

  // 이전 상태 저장 (동작 변화 감지용)
  const [prevAction, setPrevAction] = useState<string | null>(null);
  const [prevFocus, setPrevFocus] = useState<FocusLevel | null>(null);

  // 히스토리
  const [history, setHistory] = useState<DetectionResult[]>([]);

  // 공부로 인정되는 행동
  const studyActionIds = [4, 5, 9];

  // 공부 중인지
  const isStudying =
    currentActionId !== null &&
    studyActionIds.includes(currentActionId) &&
    isCameraOn;

  // ⭐ 카메라 ON/OFF 기록 추가
  const logEvent = (action: string) => {
    setHistory((prev) => [
      {
        timestamp: new Date().toISOString(),
        action,
        focus: "중간", // 기본값
      },
      ...prev.slice(0, 9),
    ]);
  };

  const handleStartCamera = () => {
    startCamera();
    logEvent("카메라 켜짐");
  };

  const handleStopCamera = () => {
    stopCamera();
    logEvent("카메라 꺼짐");
  };

  // ⭐ ST-GCN 추론 (5초마다)
  useEffect(() => {
    let timerId: number | undefined;

    const callModel = async () => {
      try {
        const skeleton = buildSkeleton(bufferRef.current);
        if (!skeleton) return;

        const actionData: ActionResponse = await predictAction(skeleton);

        const actionId = actionData.action_id;
        const actionLabel = actionData.action_label;
        const focus = mapActionToFocus(actionLabel);

        setCurrentAction(actionLabel);
        setCurrentActionId(actionId);
        setFocusLevel(focus);

        // 행동 or 집중도 변화 시 기록
        if (actionLabel !== prevAction || focus !== prevFocus) {
          setHistory((prev) => [
            {
              timestamp: new Date().toISOString(),
              action: actionLabel,
              focus,
            },
            ...prev.slice(0, 9),
          ]);

          setPrevAction(actionLabel);
          setPrevFocus(focus);
        }
      } catch (err) {
        console.error("Failed to call model API:", err);
      }
    };

    timerId = window.setInterval(callModel, 5000);

    return () => {
      if (timerId) window.clearInterval(timerId);
    };
  }, [prevAction, prevFocus, bufferRef]);

  // ⭐ 공부 타이머 (카메라 꺼져 있으면 증가 X)
  const [studySeconds, setStudySeconds] = useState(0);

  useEffect(() => {
    if (!isStudying) return;

    const timer = window.setInterval(() => {
      setStudySeconds((prev) => prev + 1);
    }, 1000);

    return () => window.clearInterval(timer);
  }, [isStudying]);

  return (
    <div
      style={{
        minHeight: "100vh",
        padding: "24px",
        backgroundColor: "#f3f4f6",
        display: "flex",
        flexDirection: "column",
        gap: "16px",
      }}
    >
      {/* 헤더 */}
      <header
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "baseline",
        }}
      >
        <div>
          <h1 style={{ fontSize: "1.6rem", fontWeight: 700 }}>학생 행동 감지</h1>
          <p style={{ color: "#6b7280", marginTop: "4px" }}>
            카메라 ON 상태에서 읽기·쓰기·타이핑 행동일 때만 공부 시간이 증가합니다.
          </p>
        </div>
      </header>

      {/* 메인 */}
      <main
        style={{
          display: "flex",
          flex: 1,
          gap: "16px",
          marginTop: "8px",
          flexWrap: "wrap",
        }}
      >
        {/* 왼쪽 패널 */}
        <CameraPanel
          videoRef={videoRef}
          canvasRef={canvasRef}
          studySeconds={studySeconds}
          isCameraOn={isCameraOn}
          onStartCamera={handleStartCamera}
          onStopCamera={handleStopCamera}
        />

        {/* 오른쪽 패널 */}
        <section
          style={{
            flex: "1 1 320px",
            display: "flex",
            flexDirection: "column",
            gap: "12px",
          }}
        >
          <ResultPanel
            currentAction={currentAction}
            focusLevel={focusLevel}
            getFocusColor={getFocusColor}
          />

          <HistoryList history={history} getFocusColor={getFocusColor} />
        </section>
      </main>
    </div>
  );
};

export default StudentBehaviorPage;
