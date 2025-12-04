import { useRef } from "react";

export type Joint = {
  x: number;
  y: number;
  z: number;
};

export type FrameJoints = Joint[]; // V개 joint

export function useSkeletonBuffer(maxFrames: number) {
  const bufferRef = useRef<FrameJoints[]>([]);

  const pushFrame = (frame: FrameJoints) => {
    const buffer = bufferRef.current;
    buffer.push(frame);
    if (buffer.length > maxFrames) {
      buffer.shift();
    }
  };

  return { bufferRef, pushFrame, maxFrames };
}
