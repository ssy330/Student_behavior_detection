import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

###############################################
# 0. 기본 설정 (여기만 네 환경에 맞게 수정)
###############################################

# ✅ ① stgcn_onecyclelr.pth 경로 (너가 로컬에 저장한 위치로 바꿔줘)
STATE_PATH = r"C:\dev\cv\model\stgcn_onecyclelr.pth"  # <-- 이 줄만 꼭 수정!

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


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


# A 행렬 생성 (학습 때랑 동일)
layout = 'ntu-rgb+d'
strategy = 'spatial'
max_hop = 1
dilation = 1
num_node = 25
edge, center = get_edge()
hop_dis = get_hop_distance(num_node, edge, max_hop=max_hop)
A_np = get_adjacency(hop_dis, center, num_node, max_hop, dilation)
A = torch.tensor(A_np, dtype=torch.float32, requires_grad=False)


###############################################
# 2. ST-GCN 기본 블록들 (학습 코드 그대로)
###############################################

class ConvTemporalGraphical(nn.Module):
    """
    Graph convolution + temporal conv
    """

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
        x = torch.einsum('nkctv,kvw->nctw', (x, A))

        return x.contiguous(), A


class st_gcn(nn.Module):
    """
    Spatial Temporal GCN block
    """

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
                    stride=(stride, 1)
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
    """
    ST-GCN 전체 모델
    Input shape: (N, C, T, V, M)
    Output: (N, num_class)
    """

    def __init__(self, in_channels, num_class, A, edge_importance_weighting, dropout):
        super().__init__()

        self.register_buffer('A', A)

        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))

        channels = [64, 64, 64, 128, 128, 256]

        self.st_gcn_networks = nn.ModuleList((
            st_gcn(in_channels, channels[0], kernel_size, 1, dropout=0.1, residual=False),
            st_gcn(channels[0], channels[1], kernel_size, 1, dropout=0.2),
            st_gcn(channels[1], channels[2], kernel_size, 1, dropout=0.3),
            st_gcn(channels[2], channels[3], kernel_size, 2, dropout=0.3),
            st_gcn(channels[3], channels[4], kernel_size, 2, dropout=0.3),
            st_gcn(channels[4], channels[5], kernel_size, 2, dropout=0.3),
        ))

        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for _ in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        last_channels = channels[-1]
        self.fcn = nn.Conv2d(last_channels, num_class, kernel_size=1)

    def forward(self, x):
        # x: (N, C, T, V, M)
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()      # N, M, V, C, T
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()      # N, M, C, T, V
        x = x.view(N * M, C, T, V)

        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, M, -1, 1, 1).mean(dim=1)

        x = self.fcn(x)
        x = x.view(x.size(0), -1)
        return x


###############################################
# 3. 체크포인트 로드해서 "진짜 모델" 만들기
###############################################

def load_stgcn_model(state_path: str, in_channels: int = 3, dropout: float = 0.2):
    """
    stgcn_onecyclelr.pth 를 로드해서 완성된 Model 객체를 반환.
    num_class는 체크포인트의 fcn.weight 크기에서 자동으로 추론.
    """
    # 먼저 checkpoint 로드
    checkpoint = torch.load(state_path, map_location=device)
    state_dict = checkpoint["model_state_dict"]

    # num_class 자동 추론: fcn.weight.shape = (num_class, 256, 1, 1)
    fcn_weight = state_dict["fcn.weight"]
    num_class = fcn_weight.shape[0]
    print(f"Detected num_class from checkpoint: {num_class}")

    # 모델 생성
    model = Model(
        in_channels=in_channels,
        num_class=num_class,
        A=A,
        edge_importance_weighting=True,
        dropout=dropout
    ).to(device)

    # 가중치 로드
    model.load_state_dict(state_dict)
    model.eval()

    print("✅ Model loaded from:", state_path)
    return model, num_class


###############################################
# 4. 입력 전처리 & 예측 함수 예시
###############################################

def prepare_input(skel_array: np.ndarray) -> torch.Tensor:
    """
    skel_array: shape (C, T, V, M) 의 numpy 배열
    반환: shape (1, C, T, V, M) 의 torch.Tensor (batch 차원 추가)
    """
    assert skel_array.ndim == 4, "Expected input shape (C, T, V, M)"
    x = torch.from_numpy(skel_array).float()
    x = x.unsqueeze(0)  # batch dimension: N=1
    return x.to(device)


def predict_single(model: nn.Module, skel_array: np.ndarray):
    """
    단일 샘플 (C, T, V, M) 에 대해:
    - logits
    - predicted class index
    - softmax 확률
    반환
    """
    x = prepare_input(skel_array)

    with torch.no_grad():
        logits = model(x)          # shape: (1, num_class)
        probs = torch.softmax(logits, dim=1)
        pred_idx = int(torch.argmax(probs, dim=1).item())
        probs_np = probs.cpu().numpy()[0]

    return logits.cpu().numpy()[0], pred_idx, probs_np


###############################################
# 5. 실행 예시 (직접 skeleton 넣어서 테스트)
###############################################

if __name__ == "__main__":
    # 1) 모델 로드
    model, num_class = load_stgcn_model(STATE_PATH)

    # 2) 예시 입력 만들기 (실제 사용할 땐 이 부분을 네 데이터로 바꿔)
    #    NTU-RGB+D 기준: C=3, V=25, M=2, T=프레임 수
    C = 3
    T = 60    # 예시로 60 frame
    V = 25
    M = 2

    # 랜덤 skeleton (실제론 여기에 너의 skeleton 데이터를 넣어야 함)
    dummy_skeleton = np.zeros((C, T, V, M), dtype=np.float32)

    # TODO: 여기서 dummy_skeleton에 실제 joint 좌표 채우기
    # ex) dummy_skeleton[0, t, v, m] = x좌표
    #     dummy_skeleton[1, t, v, m] = y좌표
    #     dummy_skeleton[2, t, v, m] = score or z

    # 3) 예측
    logits, pred_idx, probs = predict_single(model, dummy_skeleton)

    print("\n=== Inference Result ===")
    print("Logits shape:", logits.shape)       # (num_class,)
    print("Predicted class index:", pred_idx)
    print("Probabilities (first 10):", probs[:10])
