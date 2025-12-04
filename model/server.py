# server.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

###############################################
# 0. 기본 설정
###############################################

# ✔ 행동 모델 체크포인트 (필수)
ACTION_STATE_PATH = r"C:\dev\cv\model\stgcn_onecyclelr.pth"

# ✔ 집중도 모델 체크포인트 (지금은 없음 → 나중에 모델 생기면 경로만 넣기)
# 예: r"C:\dev\cv\model\stgcn_focus.pth"
FOCUS_STATE_PATH: str | None = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# === 클래스 ID → 행동 라벨 매핑 (직접 채우면 됨) ===
ACTION_ID_TO_LABEL = {
    0: "물 마시기",
    1: "음식 먹기",
    2: "앉기",
    3: "일어나기",
    4: "읽기",
    5: "쓰기",
    6: "종이 찢기",
    7: "전화하기",
    8: "휴대폰 하기",
    9: "키보드 치기",
    10: "시계 확인하기",
    11: "기침하기",
}

# === 집중도 ID → 라벨 매핑 (예: 0=낮음, 1=중간, 2=높음) ===
# 나중에 집중도 모델 학습하면 실제 클래스에 맞게 수정
FOCUS_ID_TO_LABEL = {
    0: "낮음",
    1: "중간",
    2: "높음",
}


###############################################
# 1. 그래프 관련 함수들 (학습 코드 그대로)
###############################################

def get_edge():
    num_node = 25
    self_link = [(i, i) for i in range(num_node)]
    neighbor_1base = [
        (1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
        (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
        (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
        (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
        (22, 23), (23, 8), (24, 25), (25, 12)
    ]
    neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
    edge = self_link + neighbor_link
    center = 21 - 1
    return edge, center


def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def get_adjacency(hop_dis, center, num_node, max_hop, dilation):
    valid_hop = range(0, max_hop + 1, dilation)
    adjacency = np.zeros((num_node, num_node))
    for hop in valid_hop:
        adjacency[hop_dis == hop] = 1
    normalize_adjacency = adjacency
    A = []
    for hop in valid_hop:
        a_root = np.zeros((num_node, num_node))
        a_close = np.zeros((num_node, num_node))
        a_further = np.zeros((num_node, num_node))
        for i in range(num_node):
            for j in range(num_node):
                if hop_dis[j, i] == hop:
                    if hop_dis[j, center] == hop_dis[i, center]:
                        a_root[j, i] = normalize_adjacency[j, i]
                    elif hop_dis[j, center] > hop_dis[i, center]:
                        a_close[j, i] = normalize_adjacency[j, i]
                    else:
                        a_further[j, i] = normalize_adjacency[j, i]
        if hop == 0:
            A.append(a_root)
        else:
            A.append(a_root + a_close)
            A.append(a_further)
    A = np.stack(A)
    return A


num_node = 25
edge, center = get_edge()
hop_dis = get_hop_distance(num_node, edge, max_hop=1)
A_np = get_adjacency(hop_dis, center, num_node, max_hop=1, dilation=1)
A = torch.tensor(A_np, dtype=torch.float32, requires_grad=False)


###############################################
# 2. ST-GCN 모델 정의 (학습 코드 그대로)
###############################################

class ConvTemporalGraphical(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        t_kernel_size=1,
        t_stride=1,
        t_padding=0,
        t_dilation=1,
        bias=True
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias
        )

    def forward(self, x, A):
        assert A.size(0) == self.kernel_size
        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        x = torch.einsum("nkctv,kvw->nctw", (x, A))
        return x.contiguous(), A


class st_gcn(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dropout=0,
        residual=True
    ):
        super().__init__()
        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels, kernel_size[1])
        self.tcn = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.GroupNorm(8, out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1),
                ),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res
        return self.relu(x), A


class Model(nn.Module):
    def __init__(self, in_channels, num_class, A, edge_importance_weighting, dropout):
        super().__init__()
        self.register_buffer("A", A)
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        channels = [64, 64, 64, 128, 128, 256]

        self.st_gcn_networks = nn.ModuleList(
            (
                st_gcn(in_channels, channels[0], kernel_size, 1, dropout=0.1, residual=False),
                st_gcn(channels[0], channels[1], kernel_size, 1, dropout=0.2),
                st_gcn(channels[1], channels[2], kernel_size, 1, dropout=0.3),
                st_gcn(channels[2], channels[3], kernel_size, 2, dropout=0.3),
                st_gcn(channels[3], channels[4], kernel_size, 2, dropout=0.3),
                st_gcn(channels[4], channels[5], kernel_size, 2, dropout=0.3),
            )
        )

        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList(
                [nn.Parameter(torch.ones(self.A.size())) for _ in self.st_gcn_networks]
            )
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        last_channels = channels[-1]
        self.fcn = nn.Conv2d(last_channels, num_class, kernel_size=1)

    def forward(self, x):
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()   # N,M,V,C,T
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()   # N,M,C,T,V
        x = x.view(N * M, C, T, V)

        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, M, -1, 1, 1).mean(dim=1)
        x = self.fcn(x)
        x = x.view(x.size(0), -1)
        return x


###############################################
# 3. 모델 로드 + 추론 유틸
###############################################

def load_stgcn_model(state_path: str, in_channels: int = 3, dropout: float = 0.2):
    checkpoint = torch.load(state_path, map_location=device)

    # checkpoint 포맷 유연하게 처리
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    fcn_weight = state_dict["fcn.weight"]
    num_class = fcn_weight.shape[0]
    print(f"[{state_path}] num_class detected: {num_class}")

    model = Model(
        in_channels=in_channels,
        num_class=num_class,
        A=A,
        edge_importance_weighting=True,
        dropout=dropout,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"✅ ST-GCN model loaded from: {state_path}")
    return model, num_class


# ✔ 행동 모델은 항상 로드 (필수)
if not Path(ACTION_STATE_PATH).exists():
    raise FileNotFoundError(f"Action model checkpoint not found: {ACTION_STATE_PATH}")

action_model, ACTION_NUM_CLASS = load_stgcn_model(ACTION_STATE_PATH)

# ✔ 집중도 모델은 선택적으로 로드 (없으면 None)
if FOCUS_STATE_PATH is not None and Path(FOCUS_STATE_PATH).exists():
    focus_model, FOCUS_NUM_CLASS = load_stgcn_model(FOCUS_STATE_PATH)
else:
    focus_model = None
    FOCUS_NUM_CLASS = 0
    print("⚠ 집중도 모델 없음: /predict_focus 엔드포인트는 503 오류를 반환합니다.")


def predict_action_single(skel_array: np.ndarray):
    """행동 모델로 예측"""
    x = torch.from_numpy(skel_array).float().unsqueeze(0).to(device)  # (1,C,T,V,M)
    with torch.no_grad():
        logits = action_model(x)
        probs = torch.softmax(logits, dim=1)
        pred_idx = int(torch.argmax(probs, dim=1).item())
        probs_np = probs.cpu().numpy()[0]
    return pred_idx, probs_np


def predict_focus_single(skel_array: np.ndarray):
    """집중도 모델로 예측 (모델이 있을 때만 사용)"""
    if focus_model is None:
        raise RuntimeError("Focus model is not loaded.")

    x = torch.from_numpy(skel_array).float().unsqueeze(0).to(device)  # (1,C,T,V,M)
    with torch.no_grad():
        logits = focus_model(x)
        probs = torch.softmax(logits, dim=1)
        pred_idx = int(torch.argmax(probs, dim=1).item())
        probs_np = probs.cpu().numpy()[0]
    return pred_idx, probs_np


###############################################
# 4. FastAPI 설정
###############################################

app = FastAPI()

# CORS (Vite dev 서버 도메인 허용)
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 입력: skeleton (C,T,V,M)
class SkeletonInput(BaseModel):
    skeleton: List[List[List[List[float]]]]  # 4D 배열


class ActionPredictResponse(BaseModel):
    action_id: int
    action_label: str
    probs: List[float]


class FocusPredictResponse(BaseModel):
    focus_id: int
    focus_label: str
    probs: List[float]


# 👉 프론트에서 /predict_action 또는 /predict 둘 다 사용할 수 있게
@app.post("/predict_action", response_model=ActionPredictResponse)
@app.post("/predict", response_model=ActionPredictResponse)
def predict_action(input_data: SkeletonInput):
    """
    행동 예측 엔드포인트 (행동 모델만 사용)
    """
    skel_np = np.array(input_data.skeleton, dtype=np.float32)  # (C,T,V,M)
    pred_idx, probs = predict_action_single(skel_np)
    action_label = ACTION_ID_TO_LABEL.get(pred_idx, f"class_{pred_idx}")

    return ActionPredictResponse(
        action_id=pred_idx,
        action_label=action_label,
        probs=probs.tolist(),
    )


@app.post("/predict_focus", response_model=FocusPredictResponse)
def predict_focus(input_data: SkeletonInput):
    """
    집중도 예측 엔드포인트 (집중도 모델이 있을 때만 정상 동작)
    """
    if focus_model is None:
        # 프론트에서 이 코드 보고 "집중도 모델 아직 없음" 처리하면 됨
        raise HTTPException(
            status_code=503,
            detail="Focus model not loaded (no checkpoint found).",
        )

    skel_np = np.array(input_data.skeleton, dtype=np.float32)  # (C,T,V,M)
    pred_idx, probs = predict_focus_single(skel_np)
    focus_label = FOCUS_ID_TO_LABEL.get(pred_idx, f"class_{pred_idx}")

    return FocusPredictResponse(
        focus_id=pred_idx,
        focus_label=focus_label,
        probs=probs.tolist(),
    )
