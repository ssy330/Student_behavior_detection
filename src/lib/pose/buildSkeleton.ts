import type { FrameJoints } from "./useSkeletonBuffer";

const C = 3;   // x, y, z
const T = 60;  // 시간 프레임 수
const V = 25;  // 관절 수
// M = 1 (사람 수)

export function buildSkeleton(frames: FrameJoints[]) {
  if (!frames || frames.length < T) {
    return null;
  }

  const windowFrames = frames.slice(frames.length - T); // 최근 T개

  const skeleton: number[][][][] = [];

  for (let c = 0; c < C; c++) {
    const channel: number[][][] = [];

    for (let t = 0; t < T; t++) {
      const joints: number[][] = [];

      for (let v = 0; v < V; v++) {
        const joint = windowFrames[t][v];
        let value = 0;
        if (c === 0) value = joint.x;
        else if (c === 1) value = joint.y;
        else value = joint.z;

        joints.push([value]); // M=1 → [value]
      }
      channel.push(joints);
    }

    skeleton.push(channel);
  }

  return skeleton;
}
