"""
TAU (Temporal Attention Unit) + DeepLabV3+ 모델
기존 baseline과 호환되도록 수정
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List

try:
    from model._deeplab import DeepLabHeadV3Plus
    from model.modeling import _load_model
except:
    DeepLabHeadV3Plus = None
    _load_model = None


# ============================================================
# Temporal Attention Components
# ============================================================

class TemporalPositionalEncoding(nn.Module):
    """시간적 위치 인코딩"""
    
    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pe = self.pe[:x.size(1)]
        pe = pe.unsqueeze(0)
        return self.dropout(x + pe)


class TemporalAttention(nn.Module):
    """Temporal Multi-Head Self-Attention"""
    
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1, causal: bool = True):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.causal = causal
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, C = x.shape
        
        Q = self.W_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if self.causal:
            causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(B, T, C)
        output = self.W_o(context)
        
        return output, attn_weights


class TAUBlock(nn.Module):
    """TAU Block - Temporal Attention + FFN"""
    
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        self.temporal_attn = TemporalAttention(d_model, n_heads, dropout, causal=True)
        self.norm1 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_out, attn_weights = self.temporal_attn(x)
        x = self.norm1(x + attn_out)
        
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x, attn_weights


# ============================================================
# TAU Early Fusion Module
# ============================================================

class TAUFusionEarly(nn.Module):
    """
    Early Fusion: 시간축 attention 후 single time step으로 융합
    
    입력: (B, T*C + C_misc, H, W) - 기존 baseline과 동일한 형태
    출력: (B, C + C_misc, H, W) - backbone 입력 형태
    """
    
    def __init__(
        self, 
        n_times: int, 
        n_channels: int,
        n_misc: int = 0,
        d_model: int = 64, 
        n_heads: int = 8, 
        n_layers: int = 2, 
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.n_times = n_times
        self.n_channels = n_channels
        self.n_misc = n_misc
        self.d_model = d_model
        
        # 각 시점의 채널을 d_model로 projection
        self.input_proj = nn.Conv2d(n_channels, d_model, kernel_size=1)
        
        # Positional encoding
        self.pos_encoding = TemporalPositionalEncoding(d_model, max_len=50, dropout=dropout)
        
        # TAU blocks
        self.tau_blocks = nn.ModuleList([
            TAUBlock(d_model, n_heads, dropout) for _ in range(n_layers)
        ])
        
        # Output projection: d_model -> n_channels
        self.output_proj = nn.Conv2d(d_model, n_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T*C + C_misc, H, W) - baseline과 동일한 입력
        Returns:
            (B, C + C_misc, H, W) - 시간축 융합된 출력
        """
        B, total_C, H, W = x.shape
        
        temporal_channels = self.n_times * self.n_channels
        
        # Temporal과 Misc 분리
        temporal_x = x[:, :temporal_channels]  # (B, T*C, H, W)
        misc_x = x[:, temporal_channels:] if self.n_misc > 0 else None  # (B, C_misc, H, W)
        
        # Temporal을 (B, T, C, H, W)로 reshape
        temporal_x = temporal_x.view(B, self.n_times, self.n_channels, H, W)
        
        # 1. 각 시점을 d_model로 projection
        x_proj = []
        for t in range(self.n_times):
            x_t = self.input_proj(temporal_x[:, t])  # (B, d_model, H, W)
            x_proj.append(x_t)
        x_proj = torch.stack(x_proj, dim=1)  # (B, T, d_model, H, W)
        
        # 2. Spatial을 flatten하여 temporal attention 적용
        x_flat = x_proj.permute(0, 3, 4, 1, 2).reshape(B * H * W, self.n_times, self.d_model)
        
        # 3. Positional encoding
        x_flat = self.pos_encoding(x_flat)
        
        # 4. TAU blocks
        for block in self.tau_blocks:
            x_flat, _ = block(x_flat)
        
        # 5. 마지막 시점의 feature 사용 (forecasting)
        x_out = x_flat[:, -1, :]  # (B*H*W, d_model)
        x_out = x_out.view(B, H, W, self.d_model).permute(0, 3, 1, 2)  # (B, d_model, H, W)
        
        # 6. Output projection
        x_fused = self.output_proj(x_out)  # (B, C, H, W)
        
        # 7. Misc channels 추가
        if misc_x is not None:
            x_fused = torch.cat([x_fused, misc_x], dim=1)  # (B, C + C_misc, H, W)
        
        return x_fused


# ============================================================
# TAU + DeepLabV3+ Model
# ============================================================

class TAUDeepLab(nn.Module):
    """
    TAU + DeepLabV3+ 결합 모델
    
    기존 baseline의 forward/forward_by_patch 인터페이스 유지
    """
    
    def __init__(
        self,
        n_channels: int = 16,
        n_misc: int = 3,
        n_times: int = 4,
        num_classes: int = 2,
        patch_size: int = 256,
        tau_n_heads: int = 8,
        tau_n_layers: int = 2,
        tau_dropout: float = 0.1,
        pretrained_backbone: bool = True
    ):
        super().__init__()
        
        self.n_channels = n_channels
        self.n_misc = n_misc
        self.n_times = n_times
        self.num_classes = num_classes
        self.patch_size = patch_size
        
        # 총 입력 채널: satellite * times + misc
        self.total_input_channels = n_channels * n_times + n_misc
        
        # TAU Early Fusion
        self.tau_fusion = TAUFusionEarly(
            n_times=n_times,
            n_channels=n_channels,
            n_misc=n_misc,
            d_model=64,
            n_heads=tau_n_heads,
            n_layers=tau_n_layers,
            dropout=tau_dropout
        )
        
        # Backbone 입력 채널: fusion 후
        backbone_in_channels = n_channels + n_misc
        
        # DeepLabV3+ Backbone
        if _load_model is not None:
            self.backbone = _load_model(
                'deeplabv3plus', 'resnet18', 1, 
                output_stride=None, 
                pretrained_backbone=pretrained_backbone
            )
            self.backbone.backbone.conv1 = nn.Conv2d(
                backbone_in_channels, 64, 
                kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
            )
            self.backbone.classifier = DeepLabHeadV3Plus(
                in_channels=512, 
                low_level_channels=64, 
                num_classes=1, 
                aspp_dilate=[6, 12, 18]
            )
            self.backbone.classifier.classifier = nn.Conv2d(112, num_classes, 3, 1, 1)
        else:
            self.backbone = self._build_simple_backbone(backbone_in_channels, num_classes)
    
    def _build_simple_backbone(self, in_channels: int, num_classes: int) -> nn.Module:
        """DeepLab을 사용할 수 없을 때 간단한 backbone"""
        return nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(256, num_classes, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T*C + C_misc, H, W) - baseline과 동일
        Returns:
            (B, num_classes, H, W)
        """
        # TAU fusion
        x_fused = self.tau_fusion(x)  # (B, C + C_misc, H, W)
        
        # Backbone
        output = self.backbone(x_fused)
        
        return output
    
    def forward_by_patch(
        self, 
        x: torch.Tensor, 
        patch_size: int = None,
        overlap: int = None, 
        mode: str = 'cosine', 
        eps: float = 1e-8
    ) -> torch.Tensor:
        """
        큰 이미지를 패치 단위로 처리 - 기존 baseline과 동일한 인터페이스
        """
        patch_size = patch_size or self.patch_size
        B, C, H, W = x.shape
        
        if H == patch_size and W == patch_size:
            return self.forward(x)
        
        overlap = patch_size // 2 if overlap is None else overlap
        stride = patch_size - overlap
        
        pred_map = torch.zeros([B, self.num_classes, H, W], device=x.device)
        weight_mask_total = torch.zeros([B, self.num_classes, H, W], device=x.device)
        
        # Weight mask
        if mode == 'linear':
            weight_1d = torch.linspace(0, 1, overlap, device=x.device)
        elif mode == 'cosine':
            weight_1d = 0.5 * (1 - torch.cos(torch.linspace(0, math.pi, overlap, device=x.device)))
        
        patch_weight = torch.ones([1, 1, patch_size, patch_size], device=x.device)
        patch_weight[:, :, :overlap, :] *= weight_1d.view(1, 1, -1, 1)
        patch_weight[:, :, -overlap:, :] *= weight_1d.flip(0).view(1, 1, -1, 1)
        patch_weight[:, :, :, :overlap] *= weight_1d.view(1, 1, 1, -1)
        patch_weight[:, :, :, -overlap:] *= weight_1d.flip(0).view(1, 1, 1, -1)
        
        # Patch centers
        patch_centers = set()
        center = H // 2
        patch_centers.add(center)
        
        for i in range(center, patch_size // 2, -stride):
            patch_centers.add(i)
        patch_centers.add(patch_size // 2)
        
        for i in range(center, H - patch_size // 2, stride):
            patch_centers.add(i)
        patch_centers.add(H - patch_size // 2)
        
        patch_idx = sorted(patch_centers)
        
        for i in patch_idx:
            for j in patch_idx:
                y1, y2 = i - patch_size // 2, i + patch_size // 2
                x1, x2 = j - patch_size // 2, j + patch_size // 2
                
                patch = x[:, :, y1:y2, x1:x2]
                pred = self.forward(patch)
                
                pred_map[:, :, y1:y2, x1:x2] += pred * patch_weight
                weight_mask_total[:, :, y1:y2, x1:x2] += patch_weight
        
        pred_map = pred_map / (weight_mask_total + eps)
        
        return pred_map


# ============================================================
# Convenience Function - 기존 get_model과 호환
# ============================================================

def get_tau_model(
    total_channels: int,
    patch_size: int,
    n_channels: int,
    n_misc: int,
    n_times: int,
    num_classes: int = 2,
    tau_n_heads: int = 8,
    tau_n_layers: int = 2
) -> TAUDeepLab:
    """
    TAU 모델 생성 함수
    
    Args:
        total_channels: 총 입력 채널 수 (검증용)
        patch_size: 패치 크기
        n_channels: 위성 채널 수
        n_misc: misc 채널 수
        n_times: 시간 스텝 수
        num_classes: 출력 클래스 수
        tau_n_heads: attention head 수
        tau_n_layers: TAU block 수
    """
    expected_channels = n_channels * n_times + n_misc
    assert total_channels == expected_channels, \
        f"total_channels ({total_channels}) != n_channels*n_times + n_misc ({expected_channels})"
    
    model = TAUDeepLab(
        n_channels=n_channels,
        n_misc=n_misc,
        n_times=n_times,
        num_classes=num_classes,
        patch_size=patch_size,
        tau_n_heads=tau_n_heads,
        tau_n_layers=tau_n_layers
    )
    
    return model