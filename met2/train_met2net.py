"""
Met2Net Frost Training Script
서리 예측을 위한 Met2Net 학습 스크립트

주요 수정사항:
1. Met2Net 2단계 학습 구조 적용
2. Loss 구성: backbone loss + classification loss
3. AMP (Automatic Mixed Precision) 지원
4. 체크포인트 저장/복원
"""

import os
import math
import random
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.ops import sigmoid_focal_loss
from torch.nn.functional import binary_cross_entropy_with_logits

from sklearn.metrics import roc_auc_score, confusion_matrix

# Local imports
from utils.gk2a_dataset_stage import GK2ADataset_stage
from met2.met2net_frost_stage import Met2NetFrostStage


def set_seed(seed):
    """재현성을 위한 시드 설정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calc_measure_valid(y_true, y_pred, cutoff=0.5):
    """성능 지표 계산"""
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    
    # NaN 제거
    valid_mask = ~np.isnan(y_true)
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]
    
    if len(y_true) == 0:
        return 0.0, 0.0, 0.0
    
    # Confusion matrix
    cfmat = confusion_matrix(y_true, y_pred > cutoff, labels=[0, 1])
    acc = np.trace(cfmat) / np.sum(cfmat)
    csi = cfmat[1, 1] / (np.sum(cfmat) - cfmat[0, 0] + 1e-8)
    
    try:
        auroc = roc_auc_score(y_true, y_pred)
    except:
        auroc = 0.0
    
    return csi, acc, auroc


def train_step(model, x_raw, y_raw, labels, coords, station_xy, patch_size,
               cls_num=0, lambda_met=1.0, device=None):
    """
    단일 학습 스텝
    
    Args:
        model: Met2NetFrostStage 모델
        x_raw: (B, T, C, H, W) 입력
        y_raw: (B, T, C, H, W) 타겟
        labels: (B, N) 관측소별 서리 라벨
        coords: (B, 2) 패치 좌상단 좌표
        station_xy: (N, 2) 관측소 좌표
        patch_size: 패치 크기
        cls_num: 분류 클래스 인덱스
        lambda_met: backbone loss 가중치
        device: 디바이스
        
    Returns:
        loss_total: 총 손실
        loss_seg: classification 손실
        loss_met: backbone 손실
    """
    if device is None:
        device = x_raw.device
    
    x_raw = x_raw.to(device, non_blocking=True)
    y_raw = y_raw.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    coords = coords.to(device, non_blocking=True).long()
    
    # Forward
    logits_map, met_losses = model(x_raw, y_raw)
    loss_met = met_losses[0]  # total met loss
    
    # logits_map: (B, out_heads, H, W)
    if logits_map.dim() == 5:
        logits_map = logits_map[:, 0]
    
    pred_map = logits_map[:, cls_num]  # (B, H, W)
    pred_map_fp32 = pred_map.float()
    
    # 관측소 위치에서 예측값 추출
    B = pred_map.shape[0]
    N = station_xy.shape[0]
    
    px = coords[:, 0].view(B, 1)
    py = coords[:, 1].view(B, 1)
    sx = station_xy[:, 0].view(1, N)
    sy = station_xy[:, 1].view(1, N)
    
    rel_x = sx - px
    rel_y = sy - py
    
    # 패치 내부에 있는 관측소만
    in_patch = (rel_x >= 0) & (rel_x < patch_size) & (rel_y >= 0) & (rel_y < patch_size)
    
    b_idx = torch.arange(B, device=device).view(B, 1).expand(B, N)
    
    pred_vec = torch.zeros_like(labels)
    pred_vec[in_patch] = pred_map_fp32[b_idx[in_patch], rel_y[in_patch], rel_x[in_patch]]
    
    # Valid mask
    labels_valid = (~torch.isnan(labels)).float() * in_patch.float()
    labels_clean = torch.nan_to_num(labels, 0.0)
    denom = labels_valid.sum().clamp(min=1.0)
    
    # Focal loss
    loss_focal_raw = sigmoid_focal_loss(pred_vec, labels_clean, alpha=-1, gamma=2, reduction='none')
    loss_focal = (loss_focal_raw * labels_valid).sum() / denom
    
    # Non-frost loss (all-zero batch)
    valid_any = labels_valid.sum() > 0
    all_zero = (labels_clean * labels_valid).sum() == 0
    
    if valid_any and all_zero:
        loss_non_frost = binary_cross_entropy_with_logits(
            pred_map, torch.zeros_like(pred_map), reduction='mean'
        )
    else:
        loss_non_frost = torch.tensor(0.0, device=device)
    
    # CSI loss
    sig = torch.sigmoid(pred_vec)
    tp = torch.sum(sig * labels_clean * labels_valid, dim=0)
    fn = torch.sum((1 - sig) * labels_clean * labels_valid, dim=0)
    fp = torch.sum(sig * (1 - labels_clean) * labels_valid, dim=0)
    
    loss_csi = torch.mean(-torch.log(tp + 1e-10) + torch.log(tp + fn + fp + 1e-10))
    
    # Total loss
    loss_seg = loss_focal + loss_non_frost + loss_csi
    loss_total = loss_seg + lambda_met * loss_met
    
    return loss_total, loss_seg, loss_met


@torch.no_grad()
def evaluate(model, dataloader, station_xy, patch_size, device, cutoff=0.5):
    """모델 평가"""
    model.eval()
    
    all_preds = []
    all_labels = []
    
    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        x_raw, y_raw, labels, coords = batch[0]
        x_raw = x_raw.to(device)
        
        # 패치 기반 추론
        pred_map = model.forward_by_patch(x_raw, patch_size=patch_size)[:, 0]
        pred_map = torch.sigmoid(pred_map)
        
        # 관측소 위치에서 예측값 추출
        pred_vec = []
        for x, y in station_xy.cpu().numpy():
            pred_vec.append(pred_map[:, y, x])
        pred_vec = torch.stack(pred_vec, dim=1)
        
        all_preds.append(pred_vec.cpu().numpy())
        all_labels.append(labels.numpy())
    
    preds = np.concatenate(all_preds, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    csi, acc, auroc = calc_measure_valid(labels, preds, cutoff)
    
    return csi, acc, auroc, preds, labels


def main():
    # ===============================
    # 설정
    # ===============================
    seeds = [0]
    
    # 데이터 설정
    ASOS = True
    AAFOS = False
    
    channels = '16ch'
    time_range = [-21, -18, -15, -12]
    resolution = '2km'
    postfix = 'met2net_stage'
    
    # 경로 설정
    output_path = "results/"
    output_path += 'asos_' if ASOS else ''
    output_path += 'aafos_' if AAFOS else ''
    output_path += f'{channels}_time{time_range}_{resolution}'
    output_path += f'_{postfix}' if postfix else ''
    
    print(f"Output path: {output_path}")
    os.makedirs(output_path, exist_ok=True)
    
    # 하이퍼파라미터
    batch_size = 16
    num_workers = 4
    epochs = 25
    lr = 1e-3
    decay = [10, 20]
    lr_decay = 0.1
    weight_decay = 1e-5
    threshold = [0.25]
    
    # 이미지 크기
    origin_size = 900 if resolution == '2km' else 1800
    image_size = 512 if resolution == '2km' else 1024
    patch_size = 256
    
    # 채널 설정
    channels_name = [
        'vi004', 'vi005', 'vi006', 'vi008', 'nr013', 'nr016',
        'sw038', 'wv063', 'wv069', 'wv073', 'ir087', 'ir096',
        'ir105', 'ir112', 'ir123', 'ir133'
    ]
    channels_calib = channels_name.copy()
    
    channels_mean = [
        1.1912e-01, 1.1464e-01, 1.0734e-01, 1.2504e-01, 5.4983e-02, 9.0381e-02,
        2.7813e+02, 2.3720e+02, 2.4464e+02, 2.5130e+02, 2.6948e+02, 2.4890e+02,
        2.7121e+02, 2.7071e+02, 2.6886e+02, 2.5737e+02
    ]
    channels_std = [
        0.1306, 0.1303, 0.1306, 0.1501, 0.0268, 0.0838,
        15.8211, 6.1468, 7.8054, 9.3251, 16.4265, 9.6150,
        17.2518, 17.6064, 17.0090, 12.5026
    ]
    
    # MISC 채널
    misc_channels = {
        'elevation': 'elevation_1km_3600.npy',
        'vegetation': 'vegetation_1km_3600.npy',
        'watermap': 'watermap_1km_avg_3600.npy'
    }
    
    # ===============================
    # 데이터 준비
    # ===============================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # ASOS 좌표 (예시 - 실제 좌표로 교체 필요)
    # station_xy_asos = torch.tensor([[x1, y1], [x2, y2], ...], device=device)
    # 임시로 랜덤 좌표 사용
    N_stations = 23
    station_xy_asos = torch.randint(0, image_size, (N_stations, 2), device=device)
    
    # MISC 이미지 로드
    asos_image_size = image_size  # crop 후 크기
    misc_images = []
    for misc_name, misc_path in misc_channels.items():
        misc_image = np.load(f'assets/misc_channels/{misc_path}', allow_pickle=True)
        misc_image = cv2.resize(misc_image, (asos_image_size, asos_image_size), 
                               interpolation=cv2.INTER_CUBIC)
        misc_images.append(misc_image)
    
    misc_images = np.stack(misc_images, axis=0)
    misc_images = torch.tensor(misc_images, dtype=torch.float32)
    
    # Normalization stats
    misc_mean = misc_images.mean(dim=(1, 2)).tolist()
    misc_std = misc_images.std(dim=(1, 2)).clamp(min=1e-6).tolist()
    
    total_mean = channels_mean + misc_mean
    total_std = channels_std + misc_std
    
    # Transform
    transform = transforms.Compose([
        transforms.Normalize(mean=total_mean, std=total_std)
    ])
    
    # 데이터셋 설정
    C_frame = len(channels_name) + len(misc_channels)
    T_in = len(time_range)
    
    train_data_info_list = []
    if ASOS:
        train_data_info_list.append({
            'label_type': 'asos',
            'start_date_str': '20200101',
            'end_date_str': '20230630',
            'hour_col_pairs': [(6, 'AM')],
            'label_keys': [
                '93', '108', '112', '119', '131', '133', '136', '143',
                '146', '156', '177', '102', '104', '115', '138', '152',
                '155', '159', '165', '168', '169', '184', '189'
            ]
        })
    
    test_data_info_list = []
    if ASOS:
        test_data_info_list.append({
            'label_type': 'asos',
            'start_date_str': '20230701',
            'end_date_str': '20240630',
            'hour_col_pairs': [(6, 'AM')],
            'label_keys': [
                '93', '108', '112', '119', '131', '133', '136', '143',
                '146', '156', '177', '102', '104', '115', '138', '152',
                '155', '159', '165', '168', '169', '184', '189'
            ]
        })
    
    # 패치 후보 (실제 좌표로 교체 필요)
    patch_candidates = {
        'asos': np.random.randint(0, image_size - patch_size, (100, 2))
    }
    
    # ===============================
    # 모델 설정
    # ===============================
    in_shape = [T_in, C_frame, patch_size, patch_size]
    print(f"Input shape: {in_shape}")
    
    # ===============================
    # 학습 루프
    # ===============================
    for seed in seeds:
        set_seed(seed)
        print(f"\n{'='*50}")
        print(f"Training with seed {seed}")
        print(f"{'='*50}")
        
        save_dir = f'{output_path}/{seed}'
        os.makedirs(save_dir, exist_ok=True)
        
        # 체크포인트 확인
        ckpt_path = f'{save_dir}/ckpt.pt'
        if os.path.exists(ckpt_path):
            print(f'Seed {seed} already done. Skipping...')
            continue
        
        # 모델 생성
        model = Met2NetFrostStage(
            in_shape=in_shape,
            out_heads=2,
            hid_S=8,
            hid_T=64,
            N_S=4,
            N_T=2,
            momentum_ema=0.9
        ).to(device)
        
        # Optimizer, Scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decay, gamma=lr_decay)
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        
        # 데이터로더 (실제 환경에서는 주석 해제)
        # train_dataset = GK2ADataset_stage(...)
        # train_dataloader = DataLoader(train_dataset, ...)
        
        # 더미 학습 루프 (실제 데이터 없이 구조 테스트)
        print("\n[Note] Using dummy data for structure testing")
        
        best_csi = 0.0
        best_epoch = 0
        
        for epoch in range(epochs):
            model.train()
            
            # 더미 배치
            x_raw = torch.randn(batch_size, T_in, C_frame, patch_size, patch_size, device=device)
            y_raw = x_raw.clone()
            labels = torch.randint(0, 2, (batch_size, N_stations), device=device).float()
            coords = torch.randint(0, image_size - patch_size, (batch_size, 2), device=device)
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.cuda.amp.autocast(enabled=True):
                loss, loss_seg, loss_met = train_step(
                    model, x_raw, y_raw, labels, coords,
                    station_xy=station_xy_asos,
                    patch_size=patch_size,
                    cls_num=0,
                    lambda_met=1.0,
                    device=device
                )
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            print(f"[Epoch {epoch:03d}] loss={loss.item():.4f} "
                  f"seg={loss_seg.item():.4f} met={loss_met.item():.4f}")
            
            scheduler.step()
        
        # 체크포인트 저장
        ckpt = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epochs - 1,
            'best_csi': best_csi,
        }
        torch.save(ckpt, ckpt_path)
        print(f"\nCheckpoint saved to {ckpt_path}")
    
    print("\n" + "="*50)
    print("Training completed!")
    print("="*50)


if __name__ == '__main__':
    import cv2  # main에서 필요
    main()
