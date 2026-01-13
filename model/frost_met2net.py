"""
FrostMet2Net: Met2Net-style Two-Stage Training for Frost Prediction

Based on:
- Met2Net: A Decoupled Two-Stage Spatio-Temporal Forecasting Model (arXiv:2507.17189)
- Adapted for binary frost prediction task

Key features:
1. Shared ResNet18 backbone for efficient feature extraction
2. Variable-specific projection heads
3. Two-stage training with momentum updates
4. No aggregation mode: preserves all temporal information via channel concatenation
"""

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath, trunc_normal_

# ResNet backbone import
try:
    from model.backbone.resnet import resnet18, BasicBlock
except:
    from backbone.resnet import resnet18, BasicBlock


# ============================================================================
# Shared ResNet Backbone
# ============================================================================

class ResNetBackbone(nn.Module):
    """
    Shared ResNet18 backbone for all channels.
    Uses only early layers to preserve spatial resolution.
    """
    
    def __init__(self, in_channels, out_channels=128, pretrained=False):
        super().__init__()
        
        # Load ResNet18 structure
        resnet = resnet18(pretrained=False)
        
        # Modify first conv for custom input channels
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool  # stride=2, 총 4x downsampling
        
        # Use layer1, layer2 (output: 128 channels)
        self.layer1 = resnet.layer1  # 64 channels, no downsampling
        self.layer2 = resnet.layer2  # 128 channels, stride=2 → 총 8x downsampling
        
        # Projection to desired output channels
        self.proj = nn.Conv2d(128, out_channels, 1) if out_channels != 128 else nn.Identity()
        
        # 192 / 8 = 24 (spatial size after backbone)
        # 또는 stride 조절해서 192 / 4 = 48로 맞출 수 있음
        
    def forward(self, x):
        # x: (B*T, C, H, W)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # /4
        
        x = self.layer1(x)   # 64ch
        x = self.layer2(x)   # 128ch, /2 → 총 /8
        
        x = self.proj(x)
        return x


class ResNetBackboneLight(nn.Module):
    """
    Lighter ResNet backbone with configurable downsampling.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        downsample_ratio: Spatial downsampling ratio (4, 8, or 16)
            - 4: 192 → 48
            - 8: 192 → 24
            - 16: 192 → 12
    """
    
    def __init__(self, in_channels, out_channels=64, downsample_ratio=8):
        super().__init__()
        
        assert downsample_ratio in [4, 8, 16], \
            f"downsample_ratio must be 4, 8, or 16, got {downsample_ratio}"
        
        self.downsample_ratio = downsample_ratio
        
        # Initial conv: stride=2 → 192 → 96
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # ResNet BasicBlocks (no additional downsampling)
        self.layer1 = self._make_layer(64, 64, 2)
        
        # Layer2: stride=2 → 96 → 48 (총 /4)
        self.layer2 = self._make_layer(64, out_channels, 2, stride=2)
        
        # 추가 downsampling layers (optional)
        if downsample_ratio >= 8:
            # Layer3: stride=2 → 48 → 24 (총 /8)
            self.layer3 = self._make_layer(out_channels, out_channels, 2, stride=2)
        
        if downsample_ratio >= 16:
            # Layer4: stride=2 → 24 → 12 (총 /16)
            self.layer4 = self._make_layer(out_channels, out_channels, 2, stride=2)
        
    def _make_layer(self, in_planes, out_planes, num_blocks, stride=1):
        downsample = None
        if stride != 1 or in_planes != out_planes:
            downsample = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes),
            )
        
        layers = [BasicBlock(in_planes, out_planes, stride, downsample)]
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_planes, out_planes))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)    # /2 → 96
        
        x = self.layer1(x)  # 96
        x = self.layer2(x)  # /2 → 48 (총 /4)
        
        if self.downsample_ratio >= 8:
            x = self.layer3(x)  # /2 → 24 (총 /8)
        
        if self.downsample_ratio >= 16:
            x = self.layer4(x)  # /2 → 12 (총 /16)
        
        return x


# ============================================================================
# Variable-specific Projection
# ============================================================================

class VariableProjection(nn.Module):
    """
    Project shared backbone features to variable-specific representations.
    """
    
    def __init__(self, in_channels, hid_S, var_groups):
        super().__init__()
        
        self.var_groups = var_groups
        self.n_vars = len(var_groups)
        self.hid_S = hid_S
        
        # Variable-specific projection heads
        self.proj_list = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, hid_S, 1),
                nn.BatchNorm2d(hid_S),
                nn.ReLU(inplace=True),
                nn.Conv2d(hid_S, hid_S, 3, padding=1),
                nn.BatchNorm2d(hid_S),
                nn.ReLU(inplace=True),
            )
            for _ in var_groups
        ])
    
    def forward(self, x):
        # x: (B*T, C_backbone, H', W')
        # Output: (B*T, N_var, hid_S, H', W')
        
        outputs = []
        for proj in self.proj_list:
            outputs.append(proj(x))
        
        return torch.stack(outputs, dim=1)  # (B*T, N_var, hid_S, H', W')


# ============================================================================
# Basic Building Blocks (from original)
# ============================================================================

class BasicConv2d(nn.Module):
    """Basic 2D convolution with normalization and activation."""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, upsampling=False, act_norm=True):
        super().__init__()
        self.act_norm = act_norm
        
        if upsampling:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * 4, kernel_size=kernel_size,
                          stride=1, padding=padding),
                nn.PixelShuffle(2)
            )
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                  stride=stride, padding=padding)
        
        self.norm = nn.GroupNorm(2, out_channels)
        self.act = nn.SiLU(inplace=True)
    
    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.act(self.norm(y))
        return y


class ConvSC(nn.Module):
    """Convolution with optional downsampling/upsampling."""
    
    def __init__(self, C_in, C_out, kernel_size=3, downsampling=False,
                 upsampling=False, act_norm=True):
        super().__init__()
        
        stride = 2 if downsampling else 1
        padding = (kernel_size - stride + 1) // 2
        
        self.conv = BasicConv2d(C_in, C_out, kernel_size=kernel_size, stride=stride,
                                upsampling=upsampling, padding=padding, act_norm=act_norm)
    
    def forward(self, x):
        return self.conv(x)


# ============================================================================
# Decoder
# ============================================================================

class Decoder(nn.Module):
    """
    Decoder with configurable upsampling.
    
    Args:
        C_hid: Hidden channels
        C_out: Output channels
        N_S: Number of spatial blocks (unused, kept for compatibility)
        spatio_kernel: Kernel size for convolutions
        upsample_ratio: Spatial upsampling ratio (4, 8, or 16)
            - 4: 48 → 192 (2x upsample twice)
            - 8: 24 → 192 (2x upsample three times)
            - 16: 12 → 192 (2x upsample four times)
    """
    def __init__(self, C_hid, C_out, N_S, spatio_kernel=3, upsample_ratio=4):
        super().__init__()
        
        assert upsample_ratio in [4, 8, 16], \
            f"upsample_ratio must be 4, 8, or 16, got {upsample_ratio}"
        
        # upsample 횟수: log2(ratio)
        n_upsample = {4: 2, 8: 3, 16: 4}[upsample_ratio]
        
        layers = []
        for _ in range(n_upsample):
            layers.append(ConvSC(C_hid, C_hid, spatio_kernel, upsampling=True))
        
        self.dec = nn.Sequential(*layers)
        self.readout = nn.Conv2d(C_hid, C_out, 1)
    
    def forward(self, hid):
        for layer in self.dec:
            hid = layer(hid)
        return self.readout(hid)


# ============================================================================
# Variable Attention (Cross-variable Interaction)
# ============================================================================

def attention_s(q, k, v):
    """Variable-wise attention."""
    q_flat = q.view(q.size(0), q.size(1), -1)
    k_flat = k.view(k.size(0), k.size(1), -1)
    v_flat = v.view(v.size(0), v.size(1), -1)
    
    scale = math.sqrt(q.size(2) * q.size(3) * q.size(4))
    scores = torch.matmul(q_flat, k_flat.transpose(-2, -1)) / scale
    scores = F.softmax(scores, dim=-1)
    
    out = torch.matmul(scores, v_flat)
    return out.view(q.size())


class MultiHeadVariableAttention(nn.Module):
    """Multi-head attention across variable dimension."""
    
    def __init__(self, d_model, heads=8):
        super().__init__()
        self.d_model = d_model
        self.h = heads
        
        self.q_conv = nn.Sequential(
            nn.Conv2d(d_model, d_model, 1, bias=False),
            nn.GroupNorm(1, d_model)
        )
        self.k_conv = nn.Sequential(
            nn.Conv2d(d_model, d_model, 1, bias=False),
            nn.GroupNorm(1, d_model)
        )
        self.v_conv = nn.Sequential(
            nn.Conv2d(d_model, d_model, 1, bias=False),
            nn.GroupNorm(1, d_model)
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(d_model, d_model, 1, bias=False),
            nn.GroupNorm(1, d_model),
            nn.SiLU(inplace=True)
        )
    
    def forward(self, x):
        B, N, C, H, W = x.shape
        
        x_flat = x.view(B * N, C, H, W)
        
        q = self.q_conv(x_flat).view(B, N, self.h, C // self.h, H, W)
        k = self.k_conv(x_flat).view(B, N, self.h, C // self.h, H, W)
        v = self.v_conv(x_flat).view(B, N, self.h, C // self.h, H, W)
        
        q = q.permute(0, 2, 1, 3, 4, 5).reshape(B * self.h, N, C // self.h, H, W)
        k = k.permute(0, 2, 1, 3, 4, 5).reshape(B * self.h, N, C // self.h, H, W)
        v = v.permute(0, 2, 1, 3, 4, 5).reshape(B * self.h, N, C // self.h, H, W)
        
        out = attention_s(q, k, v)
        
        out = out.view(B, self.h, N, C // self.h, H, W)
        out = out.permute(0, 2, 1, 3, 4, 5).reshape(B * N, C, H, W)
        out = self.out_conv(out).view(B, N, C, H, W)
        
        return out


# ============================================================================
# TAU Block
# ============================================================================

class TAUBlock(nn.Module):
    """Temporal Attention Unit Block."""
    
    def __init__(self, dim, kernel_size=21, mlp_ratio=4.0, drop=0.0, drop_path=0.0):
        super().__init__()
        
        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)
        
        padding = kernel_size // 2
        self.attn = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size, padding=padding, groups=dim),
            nn.BatchNorm2d(dim),
            nn.Conv2d(dim, dim, 1),
            nn.GELU()
        )
        
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
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ============================================================================
# CIAttBlock & CIMidNet (Translator)
# ============================================================================

class CIAttBlock(nn.Module):
    """Cross-variable Interaction Attention Block."""
    
    def __init__(self, d_model, heads=8, mlp_ratio=4.0, drop=0.0, drop_path=0.0):
        super().__init__()
        
        self.norm1 = nn.GroupNorm(1, d_model)
        self.norm2 = nn.GroupNorm(1, d_model)
        
        self.attn = MultiHeadVariableAttention(d_model, heads)
        self.tau = TAUBlock(d_model, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)
    
    def forward(self, x):
        B, N, C, H, W = x.shape
        
        x = x + self.attn(x)
        x = self.norm1(x.view(-1, C, H, W)).view(B, N, C, H, W)
        
        x_flat = x.view(B * N, C, H, W)
        x_flat = x_flat + self.tau(x_flat)
        x = self.norm2(x_flat).view(B, N, C, H, W)
        
        return x


class CIMidNet(nn.Module):
    """Cross-variable Interaction Mid-level Network (Translator)."""
    
    def __init__(self, in_channels, d_model, n_layers, heads=8,
                 mlp_ratio=4.0, drop=0.0, drop_path=0.1):
        super().__init__()
        
        self.conv_in = nn.Conv2d(in_channels, d_model, 1)
        
        dpr = [x.item() for x in torch.linspace(0, drop_path, n_layers)]
        self.layers = nn.ModuleList([
            CIAttBlock(d_model, heads, mlp_ratio, drop, dpr[i])
            for i in range(n_layers)
        ])
        
        self.conv_out = nn.Conv2d(d_model, in_channels, 1)
    
    def forward(self, x):
        B, N, TC, H, W = x.shape
        
        x = x.view(B * N, TC, H, W)
        x = self.conv_in(x).view(B, N, -1, H, W)
        
        for layer in self.layers:
            x = layer(x)
        
        x = x.view(B * N, -1, H, W)
        x = self.conv_out(x).view(B, N, TC, H, W)
        
        return x


# ============================================================================
# FrostMet2Net with ResNet Backbone
# ============================================================================

class FrostMet2Net(nn.Module):
    """
    Met2Net-style model for Frost Prediction with Shared ResNet Backbone.

    Architecture:
    1. Shared ResNet backbone extracts features from all channels
    2. Variable-specific projection heads create group-wise representations
    3. CIMidNet (Translator) for cross-variable interaction
    4. Variable-specific decoders reconstruct each group
    5. Classification head for frost prediction

    No Aggregation Mode:
    - Sliding window를 사용하여 시퀀스를 분할
    - 각 윈도우의 결과를 집계(mean, last 등)하지 않고 모두 채널로 연결
    - 모든 temporal 정보를 보존
    """

    def __init__(
        self,
        seq_len: int = 3,
        in_channels: int = 19,
        img_size: int = 192,
        out_channels: int = 1,
        hid_S: int = 32,
        hid_T: int = 256,
        N_S: int = 2,
        N_T: int = 4,
        n_heads: int = 8,
        momentum: float = 0.999,
        var_groups: list = None,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        drop_path: float = 0.1,
        # Window parameters (집계 없음 - concat으로 처리)
        window_size: int = 1,
        window_stride: int = 1,
        # Backbone parameters
        use_backbone: bool = True,
        backbone_out_channels: int = 64,
        downsample_ratio: int = 8,  # 4, 8, or 16
    ):
        """
        No aggregation version - 윈도우는 사용하지만 집계 대신 채널 연결.

        Args:
            seq_len: Number of input timesteps (e.g., 3 for [-18h, -15h, -12h])
            in_channels: Number of input channels per timestep
            img_size: Spatial size of input (assumed square)
            out_channels: Number of output channels (1 for binary classification)
            hid_S: Hidden spatial dimension
            hid_T: Hidden temporal dimension for translator
            N_S: Number of spatial blocks (unused, for compatibility)
            N_T: Number of translator layers
            n_heads: Number of attention heads
            momentum: Momentum for EMA update
            var_groups: Channel grouping for variable-specific processing
            mlp_ratio: MLP expansion ratio
            drop: Dropout rate
            drop_path: Drop path rate
            window_size: Number of timesteps per window
            window_stride: Step size between windows
            use_backbone: Whether to use ResNet backbone
            backbone_out_channels: Output channels of backbone
            downsample_ratio: Spatial downsampling ratio (4, 8, or 16)
        """
        super().__init__()

        self.seq_len = seq_len
        self.in_channels = in_channels
        self.hid_S = hid_S
        self.momentum = momentum
        self.use_backbone = use_backbone
        self.downsample_ratio = downsample_ratio

        # Window parameters
        self.window_size = window_size
        self.window_stride = window_stride

        # 윈도우 개수 계산
        self.n_windows = (seq_len - window_size) // window_stride + 1

        # Variable groups
        if var_groups is None:
            self.var_groups = [4, 2, 1, 3, 6, 3]
        else:
            self.var_groups = var_groups

        assert sum(self.var_groups) == in_channels, \
            f"var_groups sum ({sum(self.var_groups)}) != in_channels ({in_channels})"

        self.n_vars = len(self.var_groups)

        # Spatial dimensions after backbone
        self.H_enc = img_size // downsample_ratio
        self.W_enc = self.H_enc

        # ====== Shared Backbone (q and k) ======
        if use_backbone:
            self.backbone_q = ResNetBackboneLight(in_channels, backbone_out_channels, downsample_ratio)
            self.backbone_k = ResNetBackboneLight(in_channels, backbone_out_channels, downsample_ratio)

            # Variable-specific projection
            self.var_proj_q = VariableProjection(backbone_out_channels, hid_S, self.var_groups)
            self.var_proj_k = VariableProjection(backbone_out_channels, hid_S, self.var_groups)
        else:
            # Fallback to simple conv encoder (original style)
            self.backbone_q = nn.Sequential(
                ConvSC(in_channels, hid_S * self.n_vars, downsampling=True),
                ConvSC(hid_S * self.n_vars, hid_S * self.n_vars, downsampling=True),
            )
            self.backbone_k = nn.Sequential(
                ConvSC(in_channels, hid_S * self.n_vars, downsampling=True),
                ConvSC(hid_S * self.n_vars, hid_S * self.n_vars, downsampling=True),
            )
            self.var_proj_q = None
            self.var_proj_k = None

        # ====== Variable-specific Decoders ======
        self.dec_q_list = nn.ModuleList([
            Decoder(hid_S, g, N_S, upsample_ratio=downsample_ratio) for g in self.var_groups
        ])
        self.dec_k_list = nn.ModuleList([
            Decoder(hid_S, g, N_S, upsample_ratio=downsample_ratio) for g in self.var_groups
        ])

        # ====== Translator ======
        # 각 윈도우의 timestep들을 처리
        translator_in_channels = window_size * hid_S
        self.hid_q = CIMidNet(translator_in_channels, hid_T, N_T, n_heads,
                              mlp_ratio, drop, drop_path)
        self.hid_k = CIMidNet(translator_in_channels, hid_T, N_T, n_heads,
                              mlp_ratio, drop, drop_path)

        # ====== Classification Head (No Aggregation) ======
        # Input: 모든 윈도우 * 윈도우 내 타임스텝 * 채널을 concat
        # n_windows * window_size * in_channels
        head_in_channels = self.n_windows * window_size * in_channels
        self.head = nn.Sequential(
            nn.Conv2d(head_in_channels, in_channels * 2, 3, padding=1),
            nn.BatchNorm2d(in_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels * 2, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1)
        )

        # Initialize
        self._init_weights()
    
    def _init_weights(self):
        """Initialize momentum networks."""
        # Backbone
        if self.use_backbone:
            for param_q, param_k in zip(self.backbone_q.parameters(), self.backbone_k.parameters()):
                param_k.requires_grad = False
                param_k.data.copy_(param_q.data)
            
            for param_q, param_k in zip(self.var_proj_q.parameters(), self.var_proj_k.parameters()):
                param_k.requires_grad = False
                param_k.data.copy_(param_q.data)
        else:
            for param_q, param_k in zip(self.backbone_q.parameters(), self.backbone_k.parameters()):
                param_k.requires_grad = False
                param_k.data.copy_(param_q.data)
        
        # Decoders
        for dec_q, dec_k in zip(self.dec_q_list, self.dec_k_list):
            for param_q, param_k in zip(dec_q.parameters(), dec_k.parameters()):
                param_k.requires_grad = False
                param_k.data.copy_(param_q.data)
        
        # Translator
        for param_q, param_k in zip(self.hid_q.parameters(), self.hid_k.parameters()):
            param_k.requires_grad = False
            param_k.data.copy_(param_q.data)
    
    @torch.no_grad()
    def _momentum_update_encoder_decoder(self):
        """Copy q weights to k."""
        if self.use_backbone:
            for param_q, param_k in zip(self.backbone_q.parameters(), self.backbone_k.parameters()):
                param_k.data.copy_(param_q.data)
            for param_q, param_k in zip(self.var_proj_q.parameters(), self.var_proj_k.parameters()):
                param_k.data.copy_(param_q.data)
        else:
            for param_q, param_k in zip(self.backbone_q.parameters(), self.backbone_k.parameters()):
                param_k.data.copy_(param_q.data)
        
        for dec_q, dec_k in zip(self.dec_q_list, self.dec_k_list):
            for param_q, param_k in zip(dec_q.parameters(), dec_k.parameters()):
                param_k.data.copy_(param_q.data)
    
    @torch.no_grad()
    def _momentum_update_translator(self):
        """EMA update for translator."""
        for param_q, param_k in zip(self.hid_q.parameters(), self.hid_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)
    
    def _get_windows(self, x):
        """
        Split input sequence into windows.

        Args:
            x: (B, T, C, H, W)
        Returns:
            List of (B, window_size, C, H, W) tensors
        """
        windows = []
        for i in range(self.n_windows):
            start_idx = i * self.window_stride
            end_idx = start_idx + self.window_size
            window = x[:, start_idx:end_idx]  # (B, window_size, C, H, W)
            windows.append(window)
        return windows

    def _encode(self, x, use_q=True):
        """
        Encode with shared backbone + variable projection.

        Args:
            x: (B, T, C, H, W) - 단일 윈도우의 입력
        Returns:
            z: (B, N_var, T*hid_S, H', W')
        """
        B, T, C, H, W = x.shape
        # 슬라이싱으로 생성된 텐서가 non-contiguous할 수 있으므로 contiguous() 호출
        x = x.contiguous().view(B * T, C, H, W)

        # Shared backbone
        backbone = self.backbone_q if use_q else self.backbone_k
        feat = backbone(x)  # (B*T, backbone_out, H', W')

        # Variable-specific projection
        if self.use_backbone:
            var_proj = self.var_proj_q if use_q else self.var_proj_k
            z = var_proj(feat)  # (B*T, N_var, hid_S, H', W')
        else:
            # Reshape for non-backbone case
            z = feat.view(B * T, self.n_vars, self.hid_S, feat.shape[-2], feat.shape[-1])

        H_, W_ = z.shape[-2], z.shape[-1]

        # Reshape: (B, N_var, T*hid_S, H', W')
        z = z.view(B, T, self.n_vars, self.hid_S, H_, W_)
        z = z.permute(0, 2, 1, 3, 4, 5)  # (B, N_var, T, hid_S, H', W')
        z = z.reshape(B, self.n_vars, T * self.hid_S, H_, W_)

        return z
    
    def _decode(self, z, use_q=True):
        """
        Decode from latent space.
        
        Args:
            z: (B, N_var, T*hid_S, H', W')
        Returns:
            y: (B, T, C, H, W)
        """
        B, N_var, TC, H_, W_ = z.shape
        T = TC // self.hid_S
        
        dec_list = self.dec_q_list if use_q else self.dec_k_list
        
        # Reshape: (B*T, N_var, hid_S, H', W')
        z = z.view(B, N_var, T, self.hid_S, H_, W_)
        z = z.permute(0, 2, 1, 3, 4, 5)
        z = z.reshape(B * T, N_var, self.hid_S, H_, W_)
        
        y_list = []
        for i, dec in enumerate(dec_list):
            y_i = dec(z[:, i])
            y_list.append(y_i)
        
        y = torch.cat(y_list, dim=1)
        _, C, H, W = y.shape
        y = y.view(B, T, C, H, W)
        
        return y
    
    def _to_head_input(self, window_outputs):
        """
        Convert all window outputs to classification head input.
        No aggregation - concatenate all windows' all timesteps as channels.

        Args:
            window_outputs: List of (B, window_size, C, H, W) tensors (length = n_windows)
        Returns:
            feat: (B, n_windows * window_size * C, H, W)
        """
        # 각 윈도우 출력을 (B, window_size * C, H, W)로 flatten
        flattened = []
        for y in window_outputs:
            B, T, C, H, W = y.shape
            flattened.append(y.view(B, T * C, H, W))

        # 모든 윈도우를 채널 방향으로 연결
        # (B, n_windows * window_size * C, H, W)
        feat = torch.cat(flattened, dim=1)
        return feat

    def forward(self, x, y_target=None):
        """
        Forward pass (No aggregation version with sliding windows).

        Args:
            x: (B, T, C, H, W) - 전체 시퀀스
            y_target: (B, T, C, H, W) or None

        Returns:
            If training: (pred_map, loss_dict)
            If inference: pred_map
        """
        B, T, C, H, W = x.shape

        if y_target is None:
            return self._inference(x)

        # 윈도우로 분할
        x_windows = self._get_windows(x)
        y_windows = self._get_windows(y_target)

        loss_rec_total = 0.0
        loss_latent_total = 0.0
        y_pred_windows = []

        for x_win, y_win in zip(x_windows, y_windows):
            # ====== Stage 1: Train Encoder/Decoder ======
            z_x = self._encode(x_win, use_q=True)

            self.hid_k.eval()
            with torch.no_grad():
                z_y_pred_s1 = self.hid_k(z_x)
            self.hid_k.train()

            y_rec = self._decode(z_y_pred_s1, use_q=True)
            loss_rec_total += F.mse_loss(y_rec, y_win)

            # ====== Stage 2: Train Translator ======
            self._momentum_update_encoder_decoder()

            with torch.no_grad():
                z_x_k = self._encode(x_win, use_q=False)
                z_y_target = self._encode(y_win, use_q=False)

            z_y_pred_s2 = self.hid_q(z_x_k)
            loss_latent_total += F.mse_loss(z_y_pred_s2, z_y_target)

            self._momentum_update_translator()

            # ====== Classification input 수집 ======
            with torch.no_grad():
                y_pred = self._decode(z_y_pred_s2, use_q=False)
            y_pred_windows.append(y_pred)

        # 평균 loss
        loss_rec = loss_rec_total / self.n_windows
        loss_latent = loss_latent_total / self.n_windows

        # No aggregation: 모든 윈도우의 모든 타임스텝을 채널로 연결
        feat = self._to_head_input(y_pred_windows)
        pred_map = self.head(feat)

        loss_dict = {
            'loss_rec': loss_rec,
            'loss_latent': loss_latent,
            'loss_total': loss_rec + loss_latent
        }

        return pred_map, loss_dict

    def _inference(self, x):
        """Inference mode (No aggregation with sliding windows)."""
        # 윈도우로 분할
        x_windows = self._get_windows(x)

        y_pred_windows = []
        for x_win in x_windows:
            z_x = self._encode(x_win, use_q=True)
            z_y_pred = self.hid_q(z_x)
            y_pred = self._decode(z_y_pred, use_q=True)
            y_pred_windows.append(y_pred)

        # No aggregation: 모든 윈도우의 모든 타임스텝을 채널로 연결
        feat = self._to_head_input(y_pred_windows)
        return self.head(feat)
    
    def forward_by_patch(self, x, **kwargs):
        """Compatibility API."""
        return self._inference(x)


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B, T, C, H, W = 2, 3, 19, 192, 192

    print(f"Testing FrostMet2Net (No Aggregation + Sliding Window) on {device}")
    print(f"Input shape: ({B}, {T}, {C}, {H}, {W})")

    # Test different window configurations
    test_configs = [
        {"window_size": 1, "window_stride": 1},  # 3 windows, each 1 timestep
        {"window_size": 2, "window_stride": 1},  # 2 windows, each 2 timesteps
        {"window_size": 3, "window_stride": 1},  # 1 window, all 3 timesteps
    ]

    for config in test_configs:
        ws = config["window_size"]
        stride = config["window_stride"]
        n_windows = (T - ws) // stride + 1
        head_channels = n_windows * ws * C

        print(f"\n{'='*60}")
        print(f"Testing window_size={ws}, stride={stride}")
        print(f"  -> n_windows={n_windows}, head_in_channels={head_channels}")
        print(f"{'='*60}")

        model = FrostMet2Net(
            seq_len=T,
            in_channels=C,
            img_size=H,
            out_channels=1,
            hid_S=32,
            hid_T=128,
            N_S=2,
            N_T=4,
            n_heads=4,
            var_groups=[4, 2, 1, 3, 6, 3],
            window_size=ws,
            window_stride=stride,
            use_backbone=True,
            backbone_out_channels=64,
            downsample_ratio=8,
        ).to(device)

        # Parameter count
        n_params = sum(p.numel() for p in model.parameters())
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {n_params:,}")
        print(f"Trainable parameters: {n_trainable:,}")

        # Test inference
        x = torch.randn(B, T, C, H, W).to(device)
        model.eval()
        with torch.no_grad():
            pred = model(x)
        print(f"Inference output shape: {pred.shape}")

        # Test training
        model.train()
        y_target = torch.randn(B, T, C, H, W).to(device)
        pred, loss_dict = model(x, y_target)
        print(f"Training output shape: {pred.shape}")
        print(f"Losses: rec={loss_dict['loss_rec']:.4f}, latent={loss_dict['loss_latent']:.4f}")

        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    print("\n" + "="*60)
    print("All tests passed!")