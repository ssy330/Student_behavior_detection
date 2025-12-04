export interface ActionResponse {
  action_id: number;
  action_label: string;
  probs: number[];
}

export async function predictAction(
  skeleton: number[][][][]
): Promise<ActionResponse> {
  try {
    const res = await fetch("http://localhost:8000/predict_action", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ skeleton }),
    });

    if (!res.ok) {
      console.error("[predictAction] 서버 응답 오류:", res.status);
      return {
        action_id: -1,
        action_label: "unknown",
        probs: [],
      };
    }

    const data = (await res.json()) as ActionResponse;

    // 방어코드: 혹시라도 label이 undefined일 경우
    return {
      action_id: data.action_id ?? -1,
      action_label: data.action_label ?? "unknown",
      probs: data.probs ?? [],
    };
  } catch (err) {
    console.error("[predictAction] 요청 실패:", err);
    return {
      action_id: -1,
      action_label: "unknown",
      probs: [],
    };
  }
}
