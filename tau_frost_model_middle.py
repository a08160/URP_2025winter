"""
TAU (Temporal Attention Unit) + DeepLabV3+ 모델 - Middle Fusion 버전

원 논문 (CVPR 2023) 구조 기반:
- Spatial Encoder → TAU Blocks → Spatial Decoder

기존 Early Fusion과의 차이:
- Early: Input → TAU → Backbone
- Middle: Input → Encoder → TAU → Decoder (원 논문 방식)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List

try:
    from model._deeplab import DeepLabHeadV3Plus, ASPP
    from model.modeling import _load_model
    from model.backbone.resnet import resnet18
except:
    DeepLabHeadV3Plus = None
    ASPP = None
    _load_model = None
    resnet18 = None


# ============================================================
# Temporal Attention Components (동일)
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
# TAU Middle Fusion Module (원 논문 방식)
# ============================================================

class TAUFusionMiddle(nn.Module):
    """
    Middle Fusion: Encoder → TAU → 시간축 융합
    
    원 논문 구조 기반:
    - 각 시점을 독립적으로 spatial encoding
    - Encoded feature에서 temporal attention
    - 마지막 시점 feature 출력
    """
    
    def __init__(
        self,
        n_times: int,
        encoder_channels: int = 256,  # Encoder output channel
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.n_times = n_times
        self.encoder_channels = encoder_channels
        self.d_model = d_model
        
        # Encoder output을 d_model로 projection (필요시)
        self.input_proj = nn.Conv2d(encoder_channels, d_model, kernel_size=1) \
            if encoder_channels != d_model else nn.Identity()
        
        # Positional encoding
        self.pos_encoding = TemporalPositionalEncoding(d_model, max_len=50, dropout=dropout)
        
        # TAU blocks
        self.tau_blocks = nn.ModuleList([
            TAUBlock(d_model, n_heads, dropout) for _ in range(n_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Conv2d(d_model, encoder_channels, kernel_size=1) \
            if encoder_channels != d_model else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C, H, W) - T개 시점의 encoded features
        Returns:
            (B, C, H, W) - 시간축 융합된 출력 (마지막 시점 기준)
        """
        B, T, C, H, W = x.shape
        
        # 1. 각 시점을 d_model로 projection
        x_proj = []
        for t in range(T):
            x_t = self.input_proj(x[:, t])  # (B, d_model, H, W)
            x_proj.append(x_t)
        x_proj = torch.stack(x_proj, dim=1)  # (B, T, d_model, H, W)
        
        # 2. Spatial을 flatten하여 temporal attention 적용
        # (B, T, d_model, H, W) → (B*H*W, T, d_model)
        x_flat = x_proj.permute(0, 3, 4, 1, 2).reshape(B * H * W, T, self.d_model)
        
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
        
        return x_fused


# ============================================================
# Spatial Encoder (ResNet-based)
# ============================================================

class SpatialEncoder(nn.Module):
    """
    ResNet 기반 Spatial Encoder
    각 시점을 독립적으로 인코딩
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 256,
        pretrained: bool = True
    ):
        super().__init__()
        
        self.out_channels = out_channels
        
        # ResNet-18 기반 encoder
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet blocks (layer1, layer2만 사용 - 경량화)
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, out_channels, 2, stride=2)
        
    def _make_layer(self, in_ch: int, out_ch: int, blocks: int, stride: int = 1):
        layers = []
        
        # 첫 번째 block (downsample 가능)
        downsample = None
        if stride != 1 or in_ch != out_ch:
            downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        layers.append(BasicBlock(in_ch, out_ch, stride, downsample))
        
        # 나머지 blocks
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_ch, out_ch))
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            high_level: (B, out_channels, H/16, W/16)
            low_level: (B, 64, H/4, W/4) for skip connection
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # H/4
        
        low_level = self.layer1(x)  # H/4, 64ch
        x = self.layer2(low_level)  # H/8
        high_level = self.layer3(x)  # H/16
        
        return high_level, low_level


class BasicBlock(nn.Module):
    """ResNet BasicBlock"""
    
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.downsample = downsample
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


# ============================================================
# Spatial Decoder (DeepLabV3+ style)
# ============================================================

class SpatialDecoder(nn.Module):
    """
    DeepLabV3+ 스타일 Decoder
    ASPP + Low-level feature fusion
    """
    
    def __init__(
        self,
        high_level_channels: int = 256,
        low_level_channels: int = 64,
        num_classes: int = 2,
        aspp_dilate: List[int] = [6, 12, 18]
    ):
        super().__init__()
        
        # Low-level feature projection
        self.project_low = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        # ASPP
        self.aspp = ASPPModule(high_level_channels, 256, aspp_dilate)
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Conv2d(256 + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, 1)
        )
    
    def forward(self, high_level: torch.Tensor, low_level: torch.Tensor) -> torch.Tensor:
        """
        Args:
            high_level: (B, C_high, H/16, W/16)
            low_level: (B, C_low, H/4, W/4)
        Returns:
            (B, num_classes, H, W)
        """
        # ASPP
        x = self.aspp(high_level)  # (B, 256, H/16, W/16)
        
        # Upsample to low-level resolution
        x = F.interpolate(x, size=low_level.shape[2:], mode='bilinear', align_corners=False)
        
        # Low-level projection and concat
        low_level = self.project_low(low_level)
        x = torch.cat([x, low_level], dim=1)  # (B, 256+48, H/4, W/4)
        
        # Classify
        x = self.classifier(x)  # (B, num_classes, H/4, W/4)
        
        # Upsample to original resolution
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        
        return x


class ASPPModule(nn.Module):
    """Atrous Spatial Pyramid Pooling"""
    
    def __init__(self, in_channels: int, out_channels: int, dilations: List[int]):
        super().__init__()
        
        self.convs = nn.ModuleList()
        
        # 1x1 conv
        self.convs.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))
        
        # Dilated convs
        for d in dilations:
            self.convs.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=d, dilation=d, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
        
        # Global pooling
        self.convs.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))
        
        # Project concatenated features
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * (len(dilations) + 2), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = []
        for conv in self.convs[:-1]:
            res.append(conv(x))
        
        # Global pooling branch
        global_feat = self.convs[-1](x)
        global_feat = F.interpolate(global_feat, size=x.shape[2:], mode='bilinear', align_corners=False)
        res.append(global_feat)
        
        x = torch.cat(res, dim=1)
        x = self.project(x)
        
        return x


# ============================================================
# TAU + DeepLabV3+ Model (Middle Fusion)
# ============================================================

class TAUDeepLabMiddle(nn.Module):
    """
    TAU + DeepLabV3+ 결합 모델 - Middle Fusion 버전
    
    구조:
    Input (B, T*C + misc, H, W)
        ↓
    각 시점별 Spatial Encoder
        ↓
    TAU Fusion (encoded features에서)
        ↓
    Spatial Decoder
        ↓
    Output (B, num_classes, H, W)
    """
    
    def __init__(
        self,
        n_channels: int = 16,
        n_misc: int = 3,
        n_times: int = 4,
        num_classes: int = 2,
        patch_size: int = 256,
        encoder_channels: int = 256,
        tau_d_model: int = 256,
        tau_n_heads: int = 8,
        tau_n_layers: int = 2,
        tau_dropout: float = 0.1
    ):
        super().__init__()
        
        self.n_channels = n_channels
        self.n_misc = n_misc
        self.n_times = n_times
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.encoder_channels = encoder_channels
        
        # 총 입력 채널
        self.total_input_channels = n_channels * n_times + n_misc
        
        # 각 시점의 입력 채널 (satellite + misc)
        self.per_time_channels = n_channels + n_misc
        
        # Spatial Encoder (모든 시점에서 weight sharing)
        self.encoder = SpatialEncoder(
            in_channels=self.per_time_channels,
            out_channels=encoder_channels
        )
        
        # TAU Middle Fusion
        self.tau_fusion = TAUFusionMiddle(
            n_times=n_times,
            encoder_channels=encoder_channels,
            d_model=tau_d_model,
            n_heads=tau_n_heads,
            n_layers=tau_n_layers,
            dropout=tau_dropout
        )
        
        # Spatial Decoder
        self.decoder = SpatialDecoder(
            high_level_channels=encoder_channels,
            low_level_channels=64,
            num_classes=num_classes
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T*C + C_misc, H, W)
        Returns:
            (B, num_classes, H, W)
        """
        B, _, H, W = x.shape
        
        temporal_channels = self.n_times * self.n_channels
        
        # Temporal과 Misc 분리
        temporal_x = x[:, :temporal_channels]  # (B, T*C, H, W)
        misc_x = x[:, temporal_channels:] if self.n_misc > 0 else None
        
        # Temporal을 (B, T, C, H, W)로 reshape
        temporal_x = temporal_x.view(B, self.n_times, self.n_channels, H, W)
        
        # 1. 각 시점별로 Spatial Encoding
        high_level_features = []
        low_level_features = []
        
        for t in range(self.n_times):
            # 각 시점에 misc 추가
            x_t = temporal_x[:, t]  # (B, C, H, W)
            if misc_x is not None:
                x_t = torch.cat([x_t, misc_x], dim=1)  # (B, C+misc, H, W)
            
            high_feat, low_feat = self.encoder(x_t)
            high_level_features.append(high_feat)
            low_level_features.append(low_feat)
        
        # Stack for TAU: (B, T, C, H', W')
        high_level_features = torch.stack(high_level_features, dim=1)
        
        # 2. TAU Fusion (on encoded features)
        fused_features = self.tau_fusion(high_level_features)  # (B, C, H', W')
        
        # 3. Spatial Decoding (마지막 시점의 low-level feature 사용)
        output = self.decoder(fused_features, low_level_features[-1])
        
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
# Convenience Function
# ============================================================

def get_tau_model_middle(
    total_channels: int,
    patch_size: int,
    n_channels: int,
    n_misc: int,
    n_times: int,
    num_classes: int = 2,
    encoder_channels: int = 256,
    tau_d_model: int = 256,
    tau_n_heads: int = 8,
    tau_n_layers: int = 2
) -> TAUDeepLabMiddle:
    """
    TAU Middle Fusion 모델 생성 함수
    """
    expected_channels = n_channels * n_times + n_misc
    assert total_channels == expected_channels, \
        f"total_channels ({total_channels}) != n_channels*n_times + n_misc ({expected_channels})"
    
    model = TAUDeepLabMiddle(
        n_channels=n_channels,
        n_misc=n_misc,
        n_times=n_times,
        num_classes=num_classes,
        patch_size=patch_size,
        encoder_channels=encoder_channels,
        tau_d_model=tau_d_model,
        tau_n_heads=tau_n_heads,
        tau_n_layers=tau_n_layers
    )
    
    return model


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":
    # 테스트
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 설정 (노트북과 동일)
    n_channels = 16
    n_misc = 3
    n_times = 3
    total_channels = n_channels * n_times + n_misc  # 51
    
    print(f"Total channels: {total_channels}")
    print(f"n_channels: {n_channels}, n_misc: {n_misc}, n_times: {n_times}")
    
    # 모델 생성
    model = get_tau_model_middle(
        total_channels=total_channels,
        patch_size=384,
        n_channels=n_channels,
        n_misc=n_misc,
        n_times=n_times,
        num_classes=2,
        encoder_channels=256,
        tau_d_model=256,
        tau_n_heads=8,
        tau_n_layers=2
    ).to(device)
    
    # 파라미터 수 계산
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Forward 테스트
    x = torch.randn(2, total_channels, 384, 384).to(device)
    
    with torch.no_grad():
        output = model(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # 메모리 사용량 (CUDA)
    if torch.cuda.is_available():
        print(f"\nGPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**2:.1f} MB")
    
    print("\n모델 구조:")
    print(model)
