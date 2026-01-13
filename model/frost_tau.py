"""
Temporal Attention Unit (TAU) for Frost Prediction

Paper: "Temporal Attention Unit: Towards Efficient Spatiotemporal Predictive Learning" (CVPR 2023)
Authors: Cheng Tan et al.

논문 구조 그대로 구현:
1. Spatial Encoder: 단순 2D CNN (ConvSC 블록)
2. Temporal Module: TAU 블록 스택
   - Intra-frame Statical Attention (SA): DW Conv -> Dilated DW Conv -> 1x1 Conv
   - Inter-frame Dynamical Attention (DA): AvgPool -> FC (Squeeze-and-Excitation)
   - Output: H' = (SA ⊗ DA) ⊙ H
3. Spatial Decoder: 단순 2D CNN with skip connection
4. Loss: MSE + α * DDR (Differential Divergence Regularization)

Sliding Window 지원:
- window_size: 각 윈도우에 포함되는 타임스텝 수
- window_stride: 윈도우 간 이동 거리
- 논문대로 구현하려면 window_size=1, window_stride=1 설정
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath, trunc_normal_


# ============================================================================
# Basic Building Blocks (논문의 SimVP 기반)
# ============================================================================

class BasicConv2d(nn.Module):
    """기본 2D Conv + Norm + Act"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, act_norm=True):
        super().__init__()
        self.act_norm = act_norm
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding, dilation, groups, bias=False)
        if act_norm:
            self.norm = nn.GroupNorm(2, out_channels)
            self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.act_norm:
            x = self.norm(x)
            x = self.act(x)
        return x


class ConvSC(nn.Module):
    """Spatial Conv with down/up sampling"""

    def __init__(self, C_in, C_out, kernel_size=3, downsampling=False, upsampling=False, act_norm=True):
        super().__init__()

        stride = 2 if downsampling else 1
        padding = (kernel_size - stride + 1) // 2

        if upsampling:
            self.conv = nn.Sequential(
                nn.Conv2d(C_in, C_out * 4, kernel_size, stride=1, padding=padding),
                nn.PixelShuffle(2)
            )
        else:
            self.conv = nn.Conv2d(C_in, C_out, kernel_size, stride, padding)

        self.norm = nn.GroupNorm(2, C_out) if act_norm else nn.Identity()
        self.act = nn.SiLU(inplace=True) if act_norm else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


# ============================================================================
# Spatial Encoder / Decoder (논문: 단순 2D CNN)
# ============================================================================

class Encoder(nn.Module):
    """
    Spatial Encoder

    논문: "In the spatial encoder and decoder, the sequential input data is
    reshaped to (B×T)×C×H×W so that only spatial correlations are taken into account."
    """

    def __init__(self, C_in, C_hid, N_S, spatio_kernel=3):
        super().__init__()

        layers = [ConvSC(C_in, C_hid, spatio_kernel, downsampling=True)]
        for _ in range(N_S - 1):
            layers.append(ConvSC(C_hid, C_hid, spatio_kernel, downsampling=True))

        self.enc = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B*T, C, H, W)
        latent = self.enc[0](x)
        skip = latent  # skip connection용

        for layer in self.enc[1:]:
            latent = layer(latent)

        return latent, skip


class Decoder(nn.Module):
    """
    Spatial Decoder with skip connection

    논문: "A residual connection is added from the first convolutional layer
    to the last transposed convolutional layer for preserving spatial-dependent features."
    """

    def __init__(self, C_hid, C_out, N_S, spatio_kernel=3):
        super().__init__()

        layers = []
        for _ in range(N_S - 1):
            layers.append(ConvSC(C_hid, C_hid, spatio_kernel, upsampling=True))

        self.dec = nn.Sequential(*layers) if layers else nn.Identity()
        self.dec_out = ConvSC(C_hid, C_hid, spatio_kernel, upsampling=True)
        self.readout = nn.Conv2d(C_hid, C_out, 1)

    def forward(self, latent, skip=None):
        x = self.dec(latent) if isinstance(self.dec, nn.Sequential) else latent

        if skip is not None:
            x = x + skip  # Residual connection

        x = self.dec_out(x)
        return self.readout(x)


# ============================================================================
# Temporal Attention Unit (TAU) - 논문 핵심
# ============================================================================

class IntraFrameStaticalAttention(nn.Module):
    """
    Intra-frame Statical Attention (SA)

    논문 수식: SA = Conv_1×1(DW-D Conv(DW Conv(H)))

    - DW Conv: Depthwise Convolution
    - DW-D Conv: Dilated Depthwise Convolution
    - "large receptive field on intra-frames"를 구현하여 프레임 내 장거리 의존성 포착
    """

    def __init__(self, dim, kernel_size=21, dilation=3):
        super().__init__()

        padding = kernel_size // 2
        dilated_padding = dilation * (kernel_size // 2)

        # DW Conv: Depthwise convolution
        self.dw_conv = nn.Conv2d(dim, dim, kernel_size, padding=padding, groups=dim)

        # DW-D Conv: Dilated depthwise convolution
        self.dw_d_conv = nn.Conv2d(dim, dim, kernel_size, padding=dilated_padding,
                                    groups=dim, dilation=dilation)

        # Conv 1x1
        self.conv_1x1 = nn.Conv2d(dim, dim, 1)

        self.norm = nn.BatchNorm2d(dim)

    def forward(self, x):
        # SA = Conv_1×1(DW-D Conv(DW Conv(H)))
        x = self.dw_conv(x)
        x = self.dw_d_conv(x)
        x = self.conv_1x1(x)
        x = self.norm(x)
        return x


class InterFrameDynamicalAttention(nn.Module):
    """
    Inter-frame Dynamical Attention (DA)

    논문 수식: DA = FC(AvgPool(H))

    Squeeze-and-Excitation 방식으로 채널 attention 가중치를 학습
    "timeline을 따라 temporal evolutions"를 학습
    """

    def __init__(self, dim, reduction=4):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction, dim, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # DA = FC(AvgPool(H))
        B, C, H, W = x.shape

        # Global Average Pooling
        y = self.avg_pool(x).view(B, C)

        # FC layers
        y = self.fc(y).view(B, C, 1, 1)

        return y


class TAUBlock(nn.Module):
    """
    Temporal Attention Unit Block

    논문 최종 수식: H' = (SA ⊗ DA) ⊙ H

    - SA: Intra-frame statical attention
    - DA: Inter-frame dynamical attention
    - ⊗: Kronecker product (여기서는 element-wise broadcast multiplication)
    - ⊙: Hadamard product (element-wise multiplication)
    """

    def __init__(self, dim, kernel_size=21, mlp_ratio=4.0, drop=0.0, drop_path=0.0):
        super().__init__()

        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)

        # Intra-frame Statical Attention
        self.statical_attn = IntraFrameStaticalAttention(dim, kernel_size)

        # Inter-frame Dynamical Attention
        self.dynamical_attn = InterFrameDynamicalAttention(dim)

        # MLP
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Conv2d(hidden_dim, dim, 1),
            nn.Dropout(drop)
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        # H' = (SA ⊗ DA) ⊙ H

        # Attention
        x_norm = self.norm1(x)
        sa = self.statical_attn(x_norm)  # (B, T*C, H, W)
        da = self.dynamical_attn(x_norm)  # (B, T*C, 1, 1)

        # (SA ⊗ DA) ⊙ H: SA와 DA를 곱한 후 입력과 곱함
        attn = sa * da  # broadcast multiplication
        x = x + self.drop_path(attn)

        # MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class TemporalModule(nn.Module):
    """
    Temporal Module (Stacked TAU Blocks)

    논문: "Stacks of TAU modules are placed in the middle of the spatial encoder
    and decoder to extract temporal-dependent features."

    "In the temporal module, the feature is reshaped to B×(T×C)×H×W so that
    frames are arranged in order on the channel dimension."
    """

    def __init__(self, T, C_hid, N_T=8, kernel_size=21, mlp_ratio=4.0,
                 drop=0.0, drop_path=0.1):
        super().__init__()

        self.T = T
        self.C_hid = C_hid

        # 입력 채널: T * C_hid (시간축을 채널로 펼침)
        dim = T * C_hid

        # Stochastic depth decay
        dpr = [x.item() for x in torch.linspace(0, drop_path, N_T)]

        self.blocks = nn.ModuleList([
            TAUBlock(dim, kernel_size, mlp_ratio, drop, dpr[i])
            for i in range(N_T)
        ])

    def forward(self, x):
        # x: (B, T, C_hid, H, W)
        B, T, C, H, W = x.shape

        # Reshape to (B, T*C, H, W) - 프레임들을 채널 차원에 정렬
        x = x.reshape(B, T * C, H, W)

        # TAU blocks
        for block in self.blocks:
            x = block(x)

        # Reshape back to (B, T, C, H, W)
        x = x.reshape(B, T, C, H, W)

        return x


# ============================================================================
# Differential Divergence Regularization (DDR) Loss
# ============================================================================

class DifferentialDivergenceLoss(nn.Module):
    """
    Differential Divergence Regularization

    논문: "The mean square error loss only focuses on intra-frame differences"
    DDR은 프레임 간 변화를 추가로 학습하도록 함

    수식:
    1. Forward difference: Δŷᵢ = ŷᵢ₊₁ - ŷᵢ, Δyᵢ = yᵢ₊₁ - yᵢ
    2. Softmax로 확률 분포 변환
    3. KL-divergence 계산

    논문: "차이가 큰 프레임들이 softmax 경쟁 메커니즘을 통해 페널티를 받는다"
    """

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        """
        Args:
            pred: (B, T, C, H, W) - 예측 프레임
            target: (B, T, C, H, W) - 타겟 프레임
        """
        # Forward difference
        pred_diff = pred[:, 1:] - pred[:, :-1]  # (B, T-1, C, H, W)
        target_diff = target[:, 1:] - target[:, :-1]

        B, T_1, C, H, W = pred_diff.shape

        # Flatten
        pred_diff = pred_diff.reshape(B, T_1, -1)
        target_diff = target_diff.reshape(B, T_1, -1)

        # Softmax로 확률 분포 변환
        pred_prob = F.softmax(pred_diff, dim=-1) + self.eps
        target_prob = F.softmax(target_diff, dim=-1) + self.eps

        # KL divergence
        kl = target_prob * (torch.log(target_prob) - torch.log(pred_prob))
        kl = kl.sum(dim=-1).mean()

        return kl


# ============================================================================
# FrostTAU: Frost Prediction을 위한 TAU 모델
# ============================================================================

class FrostTAU(nn.Module):
    """
    TAU 기반 Frost Prediction 모델

    논문 구조:
    [Input] -> [Spatial Encoder] -> [Temporal Module (TAU)] -> [Spatial Decoder] -> [Output]
                      ↓                                              ↑
                      └──────────── Skip Connection ─────────────────┘

    Loss: Focal Loss + Non-frost BCE + CSI Loss (baseline과 동일)

    Sliding Window 지원:
    - window_size=1, window_stride=1: 논문 방식 (각 프레임을 독립적으로 처리)
    - window_size=seq_len, window_stride=1: 기존 방식 (전체 시퀀스를 한번에 처리)
    """

    def __init__(
        self,
        seq_len: int = 3,
        in_channels: int = 19,
        img_size: int = 192,
        out_channels: int = 1,
        hid_S: int = 64,
        N_S: int = 4,
        N_T: int = 8,
        kernel_size: int = 21,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        drop_path: float = 0.1,
        # Window parameters
        window_size: int = None,  # None이면 seq_len 사용 (기존 방식)
        window_stride: int = 1,
    ):
        """
        Args:
            seq_len: 입력 시퀀스 길이 (T)
            in_channels: 채널 수 (C)
            img_size: 이미지 크기 (H=W)
            out_channels: 출력 채널 수 (classification용)
            hid_S: Encoder/Decoder hidden channels
            N_S: Encoder/Decoder layer 수
            N_T: TAU block 수
            kernel_size: TAU attention kernel size
            mlp_ratio: MLP expansion ratio
            drop: Dropout rate
            drop_path: Stochastic depth rate
            window_size: 각 윈도우의 타임스텝 수 (None이면 seq_len, 논문대로면 1)
            window_stride: 윈도우 간 이동 거리 (논문대로면 1)
        """
        super().__init__()

        self.seq_len = seq_len
        self.in_channels = in_channels
        self.hid_S = hid_S
        self.N_S = N_S

        # Window parameters
        self.window_size = window_size if window_size is not None else seq_len
        self.window_stride = window_stride

        # 윈도우 개수 계산
        self.n_windows = (seq_len - self.window_size) // self.window_stride + 1

        # Spatial Encoder
        self.encoder = Encoder(in_channels, hid_S, N_S)

        # Temporal Module (TAU) - window_size 기준으로 생성
        self.temporal = TemporalModule(self.window_size, hid_S, N_T, kernel_size, mlp_ratio, drop, drop_path)

        # Spatial Decoder
        self.decoder = Decoder(hid_S, in_channels, N_S)

        # Classification Head
        # 모든 윈도우의 출력을 채널로 concat: n_windows * window_size * in_channels
        head_in_channels = self.n_windows * self.window_size * in_channels
        self.head = nn.Sequential(
            nn.Conv2d(head_in_channels, in_channels * 2, 3, padding=1),
            nn.BatchNorm2d(in_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels * 2, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _get_windows(self, x):
        """
        입력 시퀀스를 윈도우로 분할.

        Args:
            x: (B, T, C, H, W)
        Returns:
            List of (B, window_size, C, H, W) tensors
        """
        windows = []
        for i in range(self.n_windows):
            start_idx = i * self.window_stride
            end_idx = start_idx + self.window_size
            window = x[:, start_idx:end_idx].contiguous()  # (B, window_size, C, H, W)
            windows.append(window)
        return windows

    def _process_window(self, x_win):
        """
        단일 윈도우 처리: Encoder -> TAU -> Decoder

        Args:
            x_win: (B, window_size, C, H, W)
        Returns:
            y_pred: (B, window_size, C, H, W)
        """
        B, T_win, C, H, W = x_win.shape

        # ====== Spatial Encoder ======
        x_flat = x_win.reshape(B * T_win, C, H, W)
        latent, skip = self.encoder(x_flat)  # (B*T_win, hid_S, H', W')

        _, C_hid, H_lat, W_lat = latent.shape

        # ====== Temporal Module (TAU) ======
        latent = latent.reshape(B, T_win, C_hid, H_lat, W_lat)
        latent = self.temporal(latent)

        # ====== Spatial Decoder ======
        latent = latent.reshape(B * T_win, C_hid, H_lat, W_lat)
        y_pred = self.decoder(latent, skip)  # (B*T_win, C, H, W)
        y_pred = y_pred.reshape(B, T_win, C, H, W)

        return y_pred

    def _to_head_input(self, window_outputs):
        """
        모든 윈도우 출력을 Classification Head 입력으로 변환.
        집계 없이 모든 윈도우의 모든 타임스텝을 채널로 concat.

        Args:
            window_outputs: List of (B, window_size, C, H, W) tensors
        Returns:
            feat: (B, n_windows * window_size * C, H, W)
        """
        flattened = []
        for y in window_outputs:
            B, T_win, C, H, W = y.shape
            flattened.append(y.reshape(B, T_win * C, H, W))

        # 모든 윈도우를 채널 방향으로 연결
        feat = torch.cat(flattened, dim=1)
        return feat

    def forward(self, x, y_target=None):
        """
        Forward pass with sliding window support.

        논문 데이터 형태 변환:
        - Encoder/Decoder: (B×T)×C×H×W (공간 상관관계만)
        - Temporal Module: B×(T×C)×H×W (시간 순서대로 채널에 배치)

        Args:
            x: (B, T, C, H, W) 입력 시퀀스
            y_target: (B, T, C, H, W) 타겟 시퀀스 (학습 시 MSE/DDR loss 계산용)

        Returns:
            If y_target is None: pred_map (B, out_channels, H, W)
            If y_target is not None: (pred_map, loss_dict)
        """
        B, T, C, H, W = x.shape

        # 윈도우로 분할
        x_windows = self._get_windows(x)

        # 각 윈도우 처리
        y_pred_windows = []
        for x_win in x_windows:
            y_pred = self._process_window(x_win)
            y_pred_windows.append(y_pred)

        # Classification Head 입력 생성
        feat = self._to_head_input(y_pred_windows)
        pred_map = self.head(feat)

        # 학습 모드: MSE + DDR loss 계산
        if y_target is not None:
            y_windows = self._get_windows(y_target)

            loss_mse_total = 0.0
            loss_ddr_total = 0.0
            ddr_loss_fn = DifferentialDivergenceLoss()

            for y_pred, y_win in zip(y_pred_windows, y_windows):
                loss_mse_total += F.mse_loss(y_pred, y_win)

                # DDR은 window_size > 1일 때만 계산 가능
                if self.window_size > 1:
                    loss_ddr_total += ddr_loss_fn(y_pred, y_win)

            loss_mse = loss_mse_total / self.n_windows
            loss_ddr = loss_ddr_total / self.n_windows if self.window_size > 1 else torch.tensor(0.0, device=x.device)

            loss_dict = {
                'loss_mse': loss_mse,
                'loss_ddr': loss_ddr,
            }
            return pred_map, loss_dict

        return pred_map

    def forward_by_patch(self, x, **kwargs):
        """호환성 API"""
        return self.forward(x)


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B, T, C, H, W = 2, 3, 19, 192, 192

    print(f"Testing FrostTAU (Sliding Window) on {device}")
    print(f"Input shape: ({B}, {T}, {C}, {H}, {W})")

    # 다양한 윈도우 설정 테스트
    test_configs = [
        {"window_size": None, "window_stride": 1, "desc": "기존 방식 (전체 시퀀스)"},
        {"window_size": 1, "window_stride": 1, "desc": "논문 방식 (각 프레임 독립)"},
        {"window_size": 2, "window_stride": 1, "desc": "2-frame sliding window"},
        {"window_size": 3, "window_stride": 1, "desc": "전체 시퀀스 (window_size=seq_len)"},
    ]

    for config in test_configs:
        ws = config["window_size"]
        stride = config["window_stride"]
        desc = config["desc"]

        print(f"\n{'='*60}")
        print(f"Testing: {desc}")
        print(f"  window_size={ws}, window_stride={stride}")

        model = FrostTAU(
            seq_len=T,
            in_channels=C,
            img_size=H,
            out_channels=1,
            hid_S=64,
            N_S=4,
            N_T=8,
            kernel_size=21,
            mlp_ratio=4.0,
            drop=0.0,
            drop_path=0.1,
            window_size=ws,
            window_stride=stride,
        ).to(device)

        actual_ws = model.window_size
        n_windows = model.n_windows
        head_channels = n_windows * actual_ws * C

        print(f"  -> actual window_size={actual_ws}, n_windows={n_windows}")
        print(f"  -> head_in_channels={head_channels}")

        # Parameter count
        n_params = sum(p.numel() for p in model.parameters())
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Total parameters: {n_params:,}")
        print(f"  Trainable parameters: {n_trainable:,}")

        # Test inference
        x = torch.randn(B, T, C, H, W).to(device)
        model.eval()
        with torch.no_grad():
            pred = model(x)
        print(f"  Inference output shape: {pred.shape}")

        # Test training with loss
        model.train()
        y_target = x.clone()
        pred, loss_dict = model(x, y_target)
        print(f"  Training output shape: {pred.shape}")
        print(f"  Losses: MSE={loss_dict['loss_mse']:.4f}, DDR={loss_dict['loss_ddr']:.4f}")

        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    print("\n" + "="*60)
    print("All tests passed!")
