"""
Met2Net Frost Stage: 서리 예측을 위한 래퍼 모델
Met2Net backbone + Frost Classification Head
"""

import torch
from torch import nn
import torch.nn.functional as F
import math


class Met2NetFrostStage(nn.Module):
    """
    Met2Net을 backbone으로 사용하는 서리 예측 모델
    
    구조:
    1. Met2Net backbone: 시공간 예측 (x -> pre_y)
    2. Frost Head: pre_y의 첫 프레임 -> 서리 확률 맵
    
    학습:
    - backbone의 loss (reconstruction, latent, prediction)
    - frost classification loss (focal, CSI 등)
    """
    
    def __init__(self, in_shape, out_heads=2, **met_kwargs):
        """
        Args:
            in_shape: [T, C, H, W] - 입력 텐서 shape
            out_heads: 출력 클래스 수 (기본 2: frost/no-frost)
            **met_kwargs: Met2Net_Model에 전달할 인자들
                - hid_S: Encoder hidden channels
                - hid_T: Translator hidden dimension
                - N_S: Encoder/Decoder layers
                - N_T: Translator layers
                - momentum_ema: EMA momentum
        """
        super().__init__()
        
        # Lazy import to avoid circular dependency
        from met2.simvp_met2net import Met2Net_Model
        
        self.backbone = Met2Net_Model(in_shape=in_shape, **met_kwargs)
        
        T, C, H, W = in_shape
        self.T = T
        self.C = C
        
        # Frost Classification Head
        # pre_y[:, 0]의 채널(C)을 받아서 2채널(frost/no-frost) 맵 생성
        self.head = nn.Sequential(
            nn.Conv2d(C, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, out_heads, kernel_size=1)
        )
        
        self._init_head()
    
    def _init_head(self):
        """Head 가중치 초기화"""
        for m in self.head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x_raw, y_raw):
        """
        학습 모드 forward
        
        Args:
            x_raw: (B, T, C, H, W) - 입력 시퀀스
            y_raw: (B, T_out, C, H, W) - 타겟 시퀀스
            
        Returns:
            logits_map: (B, out_heads, H, W) - 서리 예측 맵
            met_losses: (loss_total, loss_rec, loss_latent, loss_pre) - backbone losses
        """
        # Backbone forward
        pre_y, loss_met, loss_rec, loss_latent, loss_pre = self.backbone(x_raw, y_raw)
        
        # Frost classification: t=0 프레임 사용
        # pre_y: (B, T_out, C, H, W)
        feat = pre_y[:, 0]  # (B, C, H, W)
        logits_map = self.head(feat)  # (B, out_heads, H, W)
        
        return logits_map, (loss_met, loss_rec, loss_latent, loss_pre)

    @torch.no_grad()
    def sample(self, x_raw):
        """
        추론 모드: x만으로 frost map 생성
        
        Args:
            x_raw: (B, T, C, H, W)
        Returns:
            logits_map: (B, out_heads, H, W)
        """
        pre_y = self.backbone.sample(x_raw)  # (B, T, C, H, W)
        feat = pre_y[:, 0]  # (B, C, H, W)
        logits_map = self.head(feat)  # (B, out_heads, H, W)
        return logits_map

    @torch.no_grad()
    def forward_by_patch(self, x_raw, patch_size, overlap=None, mode='cosine', eps=1e-8):
        """
        패치 기반 추론: 큰 이미지를 패치 단위로 처리 후 병합
        
        Args:
            x_raw: (B, T, C, H, W)
            patch_size: 패치 크기
            overlap: 패치 간 겹침 (기본: patch_size // 2)
            mode: 가중치 모드 ('cosine' or 'linear')
            eps: 수치 안정성을 위한 작은 값
            
        Returns:
            pred_map: (B, out_heads, H, W) - 전체 서리 예측 맵
        """
        B, T, C, H, W = x_raw.shape
        out_heads = self.head[-1].out_channels
        
        # 패치 크기와 이미지 크기가 같으면 그냥 처리
        if H == patch_size and W == patch_size:
            return self.sample(x_raw)
        
        # 겹침 설정
        overlap = patch_size // 2 if overlap is None else overlap
        stride = patch_size - overlap
        
        # 출력 버퍼
        pred_map = torch.zeros((B, out_heads, H, W), device=x_raw.device)
        weight_map = torch.zeros((B, out_heads, H, W), device=x_raw.device)
        
        # 패치 중심점 계산
        patch_centers_y = self._compute_patch_centers(H, patch_size, stride)
        patch_centers_x = self._compute_patch_centers(W, patch_size, stride)
        
        # 가중치 마스크 생성 (cosine/linear blending)
        wmask = self._create_weight_mask(patch_size, overlap, mode, x_raw.device)
        wmask = wmask.unsqueeze(0).unsqueeze(0)  # (1, 1, patch_size, patch_size)
        wmask = wmask.expand(B, out_heads, -1, -1)  # (B, out_heads, patch_size, patch_size)
        
        # 패치별 처리
        for cy in patch_centers_y:
            for cx in patch_centers_x:
                y0 = cy - patch_size // 2
                x0 = cx - patch_size // 2
                
                # 패치 추출
                patch = x_raw[:, :, :, y0:y0+patch_size, x0:x0+patch_size]
                
                # 추론
                logits = self.sample(patch)  # (B, out_heads, patch_size, patch_size)
                
                # 가중치 적용하여 누적
                pred_map[:, :, y0:y0+patch_size, x0:x0+patch_size] += logits * wmask
                weight_map[:, :, y0:y0+patch_size, x0:x0+patch_size] += wmask
        
        # 가중치 정규화
        return pred_map / (weight_map + eps)
    
    def _compute_patch_centers(self, size, patch_size, stride):
        """패치 중심점 계산"""
        centers = set()
        center = size // 2
        centers.add(center)
        
        # 중심에서 왼쪽/위로
        pos = center
        while pos > patch_size // 2:
            pos -= stride
            centers.add(max(pos, patch_size // 2))
        centers.add(patch_size // 2)
        
        # 중심에서 오른쪽/아래로
        pos = center
        while pos < size - patch_size // 2:
            pos += stride
            centers.add(min(pos, size - patch_size // 2))
        centers.add(size - patch_size // 2)
        
        return sorted(centers)
    
    def _create_weight_mask(self, patch_size, overlap, mode, device):
        """블렌딩 가중치 마스크 생성"""
        if mode == 'linear':
            w1d = torch.linspace(0, 1, overlap, device=device)
        else:  # cosine
            w1d = 0.5 * (1 - torch.cos(torch.linspace(0, math.pi, overlap, device=device)))
        
        # 2D 마스크 생성
        wmask = torch.ones((patch_size, patch_size), device=device)
        
        # 상하좌우 경계에 가중치 적용
        wmask[:overlap, :] *= w1d.view(-1, 1)
        wmask[-overlap:, :] *= w1d.flip(0).view(-1, 1)
        wmask[:, :overlap] *= w1d.view(1, -1)
        wmask[:, -overlap:] *= w1d.flip(0).view(1, -1)
        
        return wmask


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Test configuration
    B, T, C, H, W = 2, 4, 19, 256, 256
    T_out = 4
    patch_size = 128
    
    x = torch.randn((B, T, C, H, W)).to(device)
    y = torch.randn((B, T_out, C, H, W)).to(device)
    
    model = Met2NetFrostStage(
        in_shape=[T, C, H, W],
        out_heads=2,
        hid_S=8,
        hid_T=64,
        N_S=4,
        N_T=2,
        momentum_ema=0.9
    ).to(device)
    
    print("=" * 50)
    print("Testing Met2NetFrostStage")
    print("=" * 50)
    
    print("\n1. Training forward pass...")
    logits_map, met_losses = model(x, y)
    loss_met, loss_rec, loss_latent, loss_pre = met_losses
    print(f"   Logits shape: {logits_map.shape}")
    print(f"   Losses - Met: {loss_met.item():.4f}, Rec: {loss_rec.item():.4f}, "
          f"Latent: {loss_latent.item():.4f}, Pre: {loss_pre.item():.4f}")
    
    print("\n2. Sample (inference) mode...")
    with torch.no_grad():
        logits = model.sample(x)
    print(f"   Sample output shape: {logits.shape}")
    
    print("\n3. Patch-based inference...")
    with torch.no_grad():
        logits_patch = model.forward_by_patch(x, patch_size=patch_size)
    print(f"   Patch inference output shape: {logits_patch.shape}")
    
    print("\n✓ All tests passed!")
