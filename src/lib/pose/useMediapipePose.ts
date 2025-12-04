import { useEffect, RefObject } from "react";
import { Pose } from "@mediapipe/pose";
import type { FrameJoints } from "./useSkeletonBuffer";

const V = 25; // 관절 수

interface UseMediapipePoseParams {
  videoRef: RefObject<HTMLVideoElement | null>;
  isCameraOn: boolean;
  onFrame: (frame: FrameJoints) => void;
  canvasRef?: RefObject<HTMLCanvasElement | null>;
  visualize?: boolean;
}

export function useMediapipePose({
  videoRef,
  isCameraOn,
  onFrame,
  canvasRef,
  visualize = false,
}: UseMediapipePoseParams) {
  useEffect(() => {
    if (!isCameraOn || !videoRef.current) return;

    const pose = new Pose({
      locateFile: (file) =>
        `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`,
    });

    pose.setOptions({
      selfieMode: true,
      modelComplexity: 0,
      enableSegmentation: false,
      smoothLandmarks: true,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });

    let animationFrameId: number;

    pose.onResults((results: any) => {
      const landmarks =
        (results.poseLandmarks as { x: number; y: number; z?: number }[]) ??
        [];

      if (landmarks.length === 0) return;

      // 🔹 1) FrameJoints 생성해서 상위에서 관리하는 버퍼에 넣기
      const frame: FrameJoints = [];
      for (let i = 0; i < V; i++) {
        const lm = landmarks[i];
        if (!lm) {
          frame.push({ x: 0, y: 0, z: 0 });
        } else {
          frame.push({ x: lm.x, y: lm.y, z: lm.z ?? 0 });
        }
      }
      onFrame(frame);

      // 🔹 2) 선택적으로 skeleton 시각화 (canvas 위에 그리기)
      if (visualize && canvasRef?.current && videoRef.current) {
        const canvas = canvasRef.current;
        const ctx = canvas.getContext("2d");
        const videoEl = videoRef.current;

        if (!ctx || !videoEl) return;

        canvas.width = videoEl.videoWidth;
        canvas.height = videoEl.videoHeight;

        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // 점 그리기
        ctx.fillStyle = "rgb(0, 255, 0)";
        for (let i = 0; i < V; i++) {
          const lm = landmarks[i];
          if (!lm) continue;
          const x = lm.x * canvas.width;
          const y = lm.y * canvas.height;
          ctx.beginPath();
          ctx.arc(x, y, 5, 0, Math.PI * 2);
          ctx.fill();
        }

        // 선(관절 연결) 그리기
        ctx.strokeStyle = "rgb(0, 200, 255)";
        ctx.lineWidth = 3;

        const connections: [number, number][] = [
          [11, 13],
          [13, 15], // 왼팔
          [12, 14],
          [14, 16], // 오른팔
          [11, 12], // 어깨
          [23, 24], // 골반
          [11, 23],
          [12, 24], // 몸통
          [23, 25],
          [25, 27], // 왼다리
          [24, 26],
          [26, 28], // 오른다리
        ];

        for (const [sIdx, eIdx] of connections) {
          const s = landmarks[sIdx];
          const e = landmarks[eIdx];
          if (!s || !e) continue;

          ctx.beginPath();
          ctx.moveTo(s.x * canvas.width, s.y * canvas.height);
          ctx.lineTo(e.x * canvas.width, e.y * canvas.height);
          ctx.stroke();
        }
      }
    });

    const render = async () => {
      if (videoRef.current) {
        await pose.send({ image: videoRef.current });
      }
      animationFrameId = requestAnimationFrame(render);
    };

    render();

    return () => {
      cancelAnimationFrame(animationFrameId);
      pose.close();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [videoRef, isCameraOn, canvasRef, visualize]);
}
