"""
FrostResNetgSTA v2: ResNet18 (Pretrained Encoder) + Temporal-Aware gSTA + DeepLabV3+ Decoder

개선사항 (based on feedback):
1. Temporal Attention Aggregation: T개 frame을 attention 기반 weighted sum
2. Low-level feature도 temporal-aware하게 처리
3. gSTA를 frame 단위로 적용 후 temporal aggregation

아키텍처:
    Input (B, T, C, H, W)
           ↓
    ResNet18 Encoder (pretrained, output_stride=8)
           ↓ high-level: (B, T, 512, H/8, W/8)
           ↓ low-level:  (B, T, 64, H/4, W/4)
           ↓
    Channel Projection (512 → hid_T)
           ↓
    Temporal Conv3D (frame 간 interaction)
           ↓
    gSTA Blocks (각 frame별 spatial processing)
           ↓
    Temporal Attention Aggregation (T → 1)
           ↓
    ASPP + DeepLabV3+ Decoder (low-level fusion)
           ↓
    Output (B, out_channels, H, W)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torchvision.models.utils import load_state_dict_from_url


# ============================================================================
# ResNet18 Components
# ============================================================================

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        self.conv1 = conv3x3(inplanes, planes, stride, dilation=1)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, dilation=dilation)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

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


class ResNet18Encoder(nn.Module):
    """
    ResNet18 Encoder with configurable output_stride for dense prediction.
    """

    def __init__(self, in_channels=3, output_stride=8, pretrained=True):
        super(ResNet18Encoder, self).__init__()

        if output_stride == 8:
            replace_stride_with_dilation = [False, True, True]
        elif output_stride == 16:
            replace_stride_with_dilation = [False, False, True]
        else:
            replace_stride_with_dilation = [False, False, False]

        norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        self.groups = 1
        self.base_width = 64

        self.conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(BasicBlock, 64, 2)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2,
                                       dilate=replace_stride_with_dilation[2])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if pretrained and in_channels == 3:
            self._load_pretrained_weights()
        elif pretrained and in_channels != 3:
            self._load_pretrained_weights_partial(in_channels)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation

        if dilate:
            self.dilation *= stride
            stride = 1

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _load_pretrained_weights(self):
        state_dict = load_state_dict_from_url(model_urls['resnet18'], progress=True)
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith('fc')}
        self.load_state_dict(state_dict, strict=False)
        print("Loaded ImageNet pretrained weights for ResNet18")

    def _load_pretrained_weights_partial(self, in_channels):
        state_dict = load_state_dict_from_url(model_urls['resnet18'], progress=True)
        pretrained_conv1 = state_dict['conv1.weight']

        if in_channels > 3:
            repeat_times = (in_channels + 2) // 3
            new_conv1 = pretrained_conv1.repeat(1, repeat_times, 1, 1)[:, :in_channels, :, :]
            new_conv1 = new_conv1 * (3.0 / in_channels)
        else:
            new_conv1 = pretrained_conv1[:, :in_channels, :, :]

        state_dict['conv1.weight'] = new_conv1
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith('fc')}
        self.load_state_dict(state_dict, strict=False)
        print(f"Loaded partial pretrained weights for ResNet18 (in_channels={in_channels})")

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        low_level = self.layer1(x)
        x = self.layer2(low_level)
        x = self.layer3(x)
        out = self.layer4(x)

        return {'low_level': low_level, 'out': out}


# ============================================================================
# Temporal Attention Aggregator (핵심 개선)
# ============================================================================

class TemporalAttentionAggregator(nn.Module):
    """
    T개 frame feature를 attention 기반으로 aggregation.
    각 시점의 중요도를 학습해서 weighted sum.
    """
    def __init__(self, channels, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.channels = channels

        # Temporal attention score 계산
        self.temporal_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),
            nn.Linear(channels, channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 4, 1),
        )

    def forward(self, x):
        """
        Args:
            x: (B, T, C, H, W)
        Returns:
            out: (B, C, H, W)
            attn_weights: (B, T) - for visualization/debugging
        """
        B, T, C, H, W = x.shape

        # 각 frame별 attention score 계산
        attn_scores = []
        for t in range(T):
            score = self.temporal_attn(x[:, t])  # (B, 1)
            attn_scores.append(score)

        attn_scores = torch.cat(attn_scores, dim=1)  # (B, T)
        attn_weights = F.softmax(attn_scores, dim=1)  # (B, T)

        # Weighted sum
        attn_weights_expanded = attn_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (B, T, 1, 1, 1)
        out = (x * attn_weights_expanded).sum(dim=1)  # (B, C, H, W)

        return out, attn_weights


# ============================================================================
# gSTA Module (from SimVPv2) - 수정 없음
# ============================================================================

class gSTA(nn.Module):
    """
    Gated Spatiotemporal Attention Module from SimVPv2.
    Large kernel convolution decomposed into depth-wise + dilation + 1x1.
    """

    def __init__(self, channels, kernel_size=21, dilation=3, mlp_ratio=4.0, drop=0.0):
        super(gSTA, self).__init__()

        self.channels = channels
        dw_kernel = (kernel_size + dilation - 1) // dilation

        self.dw_conv = nn.Conv2d(
            channels, channels,
            kernel_size=2*dilation-1,
            padding=dilation-1,
            groups=channels
        )

        self.dw_dilation_conv = nn.Conv2d(
            channels, channels,
            kernel_size=dw_kernel,
            padding=(dw_kernel // 2) * dilation,
            dilation=dilation,
            groups=channels
        )

        self.channel_conv = nn.Conv2d(channels, channels * 2, kernel_size=1)

        hidden_dim = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, 1),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Conv2d(hidden_dim, channels, 1),
            nn.Dropout(drop),
        )

        self.norm1 = nn.GroupNorm(1, channels)
        self.norm2 = nn.GroupNorm(1, channels)

    def forward(self, x):
        residual = x

        x = self.norm1(x)
        x = self.dw_conv(x)
        x = self.dw_dilation_conv(x)
        x = self.channel_conv(x)

        gate, value = x.chunk(2, dim=1)
        x = torch.sigmoid(gate) * value

        x = residual + x
        x = x + self.mlp(self.norm2(x))

        return x


# ============================================================================
# Temporal gSTA (핵심 개선: frame 단위 처리)
# ============================================================================

class TemporalGSTA(nn.Module):
    """
    Frame 단위로 gSTA 적용 + temporal interaction.

    기존 문제: T*C를 channel로 flatten하면 temporal sequence로 인식 안됨
    개선:
      1. Temporal conv로 frame 간 interaction
      2. 각 frame에 gSTA 적용
      3. Attention으로 aggregation
    """

    def __init__(self, channels, seq_len, num_blocks=4, kernel_size=21, dilation=3,
                 mlp_ratio=4.0, drop=0.0):
        super().__init__()

        self.seq_len = seq_len
        self.num_blocks = num_blocks

        # Temporal interaction (frame 간 정보 교환)
        self.temporal_conv = nn.Conv3d(
            channels, channels,
            kernel_size=(3, 1, 1),
            padding=(1, 0, 0),
            groups=channels
        )
        self.temporal_bn = nn.BatchNorm3d(channels)

        # Spatial processing with gSTA (각 frame별)
        self.gsta_blocks = nn.ModuleList([
            gSTA(channels, kernel_size, dilation, mlp_ratio, drop)
            for _ in range(num_blocks)
        ])

        # Final temporal aggregation
        self.temporal_agg = TemporalAttentionAggregator(channels, seq_len)

    def forward(self, x):
        """
        Args:
            x: (B, T, C, H, W)
        Returns:
            out: (B, C, H, W)
            attn_weights: (B, T)
        """
        B, T, C, H, W = x.shape

        # 1. Temporal interaction
        x = x.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
        x = self.temporal_conv(x)
        x = self.temporal_bn(x)
        x = F.relu(x, inplace=True)
        x = x.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)

        # 2. Apply gSTA to each frame
        out_frames = []
        for t in range(T):
            frame = x[:, t]  # (B, C, H, W)
            for block in self.gsta_blocks:
                frame = block(frame)
            out_frames.append(frame)

        x = torch.stack(out_frames, dim=1)  # (B, T, C, H, W)

        # 3. Temporal aggregation
        out, attn_weights = self.temporal_agg(x)

        return out, attn_weights


# ============================================================================
# Decoder (DeepLabV3+ style)
# ============================================================================

class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling module."""

    def __init__(self, in_channels, out_channels=256, atrous_rates=[6, 12, 18]):
        super(ASPP, self).__init__()

        modules = []

        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))

        for rate in atrous_rates:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=rate,
                          dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))

        modules.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        res = []
        for conv in self.convs[:-1]:
            res.append(conv(x))

        global_feat = self.convs[-1](x)
        global_feat = F.interpolate(global_feat, size=x.shape[2:],
                                    mode='bilinear', align_corners=False)
        res.append(global_feat)

        res = torch.cat(res, dim=1)
        return self.project(res)


class DeepLabDecoder(nn.Module):
    """DeepLabV3+ style decoder with low-level feature fusion."""

    def __init__(self, low_level_channels=64, high_level_channels=256,
                 out_channels=256, num_classes=1):
        super(DeepLabDecoder, self).__init__()

        self.low_level_proj = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(high_level_channels + 48, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Conv2d(out_channels, num_classes, 1)

    def forward(self, high_level_feat, low_level_feat, target_size):
        low_level_feat = self.low_level_proj(low_level_feat)

        high_level_feat = F.interpolate(
            high_level_feat,
            size=low_level_feat.shape[2:],
            mode='bilinear',
            align_corners=False
        )

        x = torch.cat([high_level_feat, low_level_feat], dim=1)
        x = self.fusion(x)

        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)

        return self.classifier(x)


# ============================================================================
# Main Model: FrostResNetgSTA (v2 - Temporal-Aware)
# ============================================================================

class FrostResNetgSTA(nn.Module):
    """
    Frost prediction model with proper temporal modeling:
    - ResNet18 (pretrained) for spatial feature extraction per frame
    - Temporal Conv3D for inter-frame interaction
    - gSTA for spatial processing per frame
    - Attention-based temporal aggregation
    - DeepLabV3+ style decoder for dense prediction

    Args:
        seq_len: Number of input frames (T)
        in_channels: Channels per frame (C)
        img_size: Input image size (H, W assumed square)
        out_channels: Output channels (1 for frost probability)
        hid_T: Hidden channels for temporal modeling
        N_T: Number of gSTA blocks
        output_stride: ResNet output stride (8 recommended)
        pretrained: Use ImageNet pretrained ResNet18
        kernel_size: gSTA kernel size (default 21)
        dilation: gSTA dilation (default 3)
        aspp_dilate: ASPP dilation rates

    Input: (B, T, C, H, W)
    Output: (B, out_channels, H, W)
    """

    def __init__(
        self,
        seq_len: int,
        in_channels: int,
        img_size: int,
        out_channels: int = 1,
        hid_T: int = 64,
        N_T: int = 4,
        output_stride: int = 8,
        pretrained: bool = True,
        kernel_size: int = 21,
        dilation: int = 3,
        aspp_dilate: list = [6, 12, 18],
        drop: float = 0.0,
    ):
        super(FrostResNetgSTA, self).__init__()

        self.seq_len = seq_len
        self.in_channels = in_channels
        self.hid_T = hid_T

        # 1. ResNet18 Encoder (frame별 독립 처리)
        self.encoder = ResNet18Encoder(
            in_channels=in_channels,
            output_stride=output_stride,
            pretrained=pretrained
        )

        # 2. Channel projection: 512 -> hid_T
        self.projection = nn.Sequential(
            nn.Conv2d(512, hid_T, 1, bias=False),
            nn.BatchNorm2d(hid_T),
            nn.ReLU(inplace=True)
        )

        # 3. Temporal-aware gSTA (핵심 개선)
        self.temporal_gsta = TemporalGSTA(
            channels=hid_T,
            seq_len=seq_len,
            num_blocks=N_T,
            kernel_size=kernel_size,
            dilation=dilation,
            mlp_ratio=4.0,
            drop=drop
        )

        # 4. Low-level temporal aggregator (동일한 attention 사용)
        self.low_level_agg = TemporalAttentionAggregator(64, seq_len)

        # 5. ASPP for multi-scale context
        self.aspp = ASPP(hid_T, 256, aspp_dilate)

        # 6. Decoder with low-level fusion
        self.decoder = DeepLabDecoder(
            low_level_channels=64,
            high_level_channels=256,
            out_channels=256,
            num_classes=out_channels
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for non-pretrained modules"""
        for m in [self.projection, self.temporal_gsta, self.low_level_agg,
                  self.aspp, self.decoder]:
            for module in m.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight, mode='fan_out',
                                           nonlinearity='relu')
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
                elif isinstance(module, nn.Conv3d):
                    nn.init.kaiming_normal_(module.weight, mode='fan_out',
                                           nonlinearity='relu')
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
                elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm)):
                    nn.init.constant_(module.weight, 1)
                    nn.init.constant_(module.bias, 0)
                elif isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, 0, 0.01)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, T, C, H, W)

        Returns:
            Frost probability map of shape (B, out_channels, H, W)
        """
        B, T, C, H, W = x.shape

        # 1. Encode each frame with ResNet18
        x_flat = x.view(B * T, C, H, W)
        features = self.encoder(x_flat)

        high_level = features['out']       # (B*T, 512, H', W')
        low_level = features['low_level']  # (B*T, 64, H/4, W/4)

        _, _, H_feat, W_feat = high_level.shape
        _, _, H_low, W_low = low_level.shape

        # 2. Project high-level features
        high_level = self.projection(high_level)  # (B*T, hid_T, H', W')

        # 3. Reshape to (B, T, C, H, W) for temporal processing
        high_level = high_level.view(B, T, self.hid_T, H_feat, W_feat)
        low_level = low_level.view(B, T, 64, H_low, W_low)

        # 4. Temporal-aware gSTA processing
        high_level, high_attn = self.temporal_gsta(high_level)  # (B, hid_T, H', W')

        # 5. Low-level temporal aggregation (같은 attention 패턴 활용 가능)
        low_level, low_attn = self.low_level_agg(low_level)  # (B, 64, H/4, W/4)

        # 6. ASPP multi-scale context
        high_level = self.aspp(high_level)  # (B, 256, H', W')

        # 7. Decode with low-level fusion
        out = self.decoder(high_level, low_level, target_size=(H, W))

        return out

    def forward_with_attention(self, x):
        """
        Forward pass that also returns attention weights for visualization.

        Returns:
            out: (B, out_channels, H, W)
            high_attn: (B, T) - high-level temporal attention
            low_attn: (B, T) - low-level temporal attention
        """
        B, T, C, H, W = x.shape

        x_flat = x.view(B * T, C, H, W)
        features = self.encoder(x_flat)

        high_level = features['out']
        low_level = features['low_level']

        _, _, H_feat, W_feat = high_level.shape
        _, _, H_low, W_low = low_level.shape

        high_level = self.projection(high_level)

        high_level = high_level.view(B, T, self.hid_T, H_feat, W_feat)
        low_level = low_level.view(B, T, 64, H_low, W_low)

        high_level, high_attn = self.temporal_gsta(high_level)
        low_level, low_attn = self.low_level_agg(low_level)

        high_level = self.aspp(high_level)
        out = self.decoder(high_level, low_level, target_size=(H, W))

        return out, high_attn, low_attn

    def forward_by_patch(self, x, patch_size=None, stride=None, **kwargs):
        """
        Patch-wise inference for large images.
        If input size matches patch_size, just do normal forward.
        """
        B, T, C, H, W = x.shape

        if patch_size is None or (H == patch_size and W == patch_size):
            return self.forward(x)

        if stride is None:
            stride = patch_size // 2

        pad_h = (patch_size - H % patch_size) % patch_size
        pad_w = (patch_size - W % patch_size) % patch_size

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        _, _, _, H_pad, W_pad = x.shape

        out_channels = 1
        output = torch.zeros(B, out_channels, H_pad, W_pad, device=x.device)
        count = torch.zeros(B, out_channels, H_pad, W_pad, device=x.device)

        for i in range(0, H_pad - patch_size + 1, stride):
            for j in range(0, W_pad - patch_size + 1, stride):
                patch = x[:, :, :, i:i+patch_size, j:j+patch_size]
                pred = self.forward(patch)
                output[:, :, i:i+patch_size, j:j+patch_size] += pred
                count[:, :, i:i+patch_size, j:j+patch_size] += 1

        output = output / count.clamp(min=1)
        output = output[:, :, :H, :W]

        return output


# ============================================================================
# Utility function for model creation
# ============================================================================

def get_frost_resnet_gsta(
    seq_len: int = 4,
    in_channels: int = 19,
    img_size: int = 384,
    out_channels: int = 1,
    hid_T: int = 64,
    N_T: int = 4,
    output_stride: int = 8,
    pretrained: bool = True,
    device: str = None
):
    """Create FrostResNetgSTA model."""
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'

    model = FrostResNetgSTA(
        seq_len=seq_len,
        in_channels=in_channels,
        img_size=img_size,
        out_channels=out_channels,
        hid_T=hid_T,
        N_T=N_T,
        output_stride=output_stride,
        pretrained=pretrained,
    )

    model = model.to(device)

    return model


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing FrostResNetgSTA v2 (Temporal-Aware)")
    print("=" * 60)

    config = {
        'seq_len': 3,        # 3 time steps (matching time_range=[-18, -15, -12])
        'in_channels': 19,   # 16 satellite channels + 3 misc
        'img_size': 192,
        'out_channels': 1,
        'hid_T': 64,
        'N_T': 4,
        'output_stride': 8,
        'pretrained': False,
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    model = FrostResNetgSTA(**config).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    batch_size = 2
    x = torch.randn(batch_size, config['seq_len'], config['in_channels'],
                    config['img_size'], config['img_size'], device=device)
    print(f"\nInput shape: {x.shape}")

    with torch.no_grad():
        out = model(x)
        out_attn, high_attn, low_attn = model.forward_with_attention(x)

    print(f"Output shape: {out.shape}")
    print(f"High-level temporal attention: {high_attn}")
    print(f"Low-level temporal attention: {low_attn}")

    if device == 'cuda':
        print(f"\nGPU Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"GPU Memory cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

    print("\n" + "=" * 60)
    print("Test passed!")
    print("=" * 60)


# ============================================================================
# Alias for compatibility
# ============================================================================
FrostSimVPv2 = FrostResNetgSTA
