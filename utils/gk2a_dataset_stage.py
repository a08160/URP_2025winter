"""
GK2A Dataset for Met2Net Stage Training
기존 데이터셋을 Stage 학습에 맞게 수정
- x_raw: 입력 시퀀스 (B, T_in, C, H, W)
- y_raw: 타겟 시퀀스 (B, T_out, C, H, W) - 입력과 동일하거나 미래 시점
"""

import os
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm
from datetime import datetime, timedelta
import random
import cv2

import torch
from torch.utils.data import Dataset

TOTAL_START_DATE = '20200101'
TOTAL_END_DATE = '20250731'

LABEL_DIR = {
    'aafos': 'assets/labels/aafos_2201_2507.csv',
    'asos': 'assets/labels/asos_2001_2507.csv'
}


class GK2ADataset_stage(Dataset):
    """
    GK2A 위성 데이터셋 - Met2Net Stage 학습용
    
    출력 형태:
    - x_raw: (T_in, C, H, W) 또는 패치 적용 시 (T_in, C, patch_size, patch_size)
    - y_raw: x_raw와 동일 (reconstruction 학습용)
    - labels: (N,) 관측소별 서리 라벨
    - coords: (2,) 패치 좌상단 좌표 [x0, y0]
    """
    
    def __init__(self, data_path: str, output_path: str, data_info_list: list, 
                 channels: str, time_range: tuple, channels_calib: list, 
                 image_size: int, misc_images: torch.Tensor, 
                 patch_size: int = None, patch_candidates: dict = None, 
                 transform=None, train: bool = True):
        """
        Args:
            data_path: 데이터 디렉토리
            output_path: 출력 디렉토리
            data_info_list: 라벨 종류별 데이터 정보 리스트
            channels: 채널 키 ('16ch' 등)
            time_range: 입력 시간 범위 (예: [-21, -18, -15, -12])
            channels_calib: calibration 적용할 채널 리스트
            image_size: 출력 이미지 크기
            misc_images: 추가 채널 (elevation, vegetation 등), (C_misc, H, W)
            patch_size: 패치 크기 (None이면 전체 이미지)
            patch_candidates: 라벨 타입별 패치 후보 좌표
            transform: 전처리 변환
            train: 학습 모드 여부
        """
        super(GK2ADataset_stage, self).__init__()
        
        self.data_path = data_path
        self.output_path = output_path
        self.data_info_list = [d for d in data_info_list if d is not None]
        
        self.channels = channels
        self.time_range = time_range
        self.calib = channels_calib
        self.image_size = image_size
        self.misc_images = misc_images
        
        self.patch_size = patch_size
        self.patch_candidates = patch_candidates
        self.patch = False
        
        self.transform = transform
        self.train = train
        
        # Calibration 테이블 로드
        self.calib_table = pd.read_excel(
            'assets/20191115_gk-2a ami calibration table_v3.1_ir133_srf_shift.xlsx',
            sheet_name='Calibration Table_WN', header=0
        )
        self.calib_col = {
            'vi004': 'VIS 0.4', 'vi005': 'VIS 0.5', 'vi006': 'VIS 0.6', 
            'vi008': 'VIS 0.8', 'nr013': 'NIR 1.3', 'nr016': 'NIR 1.6', 
            'sw038': 'IR 3.8', 'wv063': 'IR 6.3', 'wv069': 'IR 6.9', 
            'wv073': 'IR 7.3', 'ir087': 'IR 8.7', 'ir096': 'IR 9.6', 
            'ir105': 'IR 10.5', 'ir112': 'IR 11.2', 'ir123': 'IR 12.3', 
            'ir133': 'IR 13.3'
        }
        
        self.prepare_data_info_list()
        self.sync_dataset_length()
        print("GK2A Dataset (Stage) initialized\n")
        
    def patchfy(self, enable: bool):
        """패치 모드 활성화/비활성화"""
        if enable:
            assert self.patch_size is not None, "patch_size should be specified"
            assert self.patch_candidates is not None, "patch_candidates should be provided"
        self.patch = enable
        print(f'Dataset patchfy set to {self.patch}.\n')

    def calibrate(self, image, channel):
        """채널별 calibration 적용"""
        H, W = image.shape
        calib_table = self.calib_table
        origin_col = self.calib_col[channel]
        
        albedo_col = calib_table.columns[calib_table.columns.get_loc(origin_col) + 1]
        calib_col = np.array(calib_table[albedo_col][1:].dropna(), dtype=np.float32)
        
        image = image.flatten()
        indices = np.clip(image.astype(np.int32), 0, len(calib_col) - 1)
        image = calib_col[indices].reshape(H, W)
        
        return torch.from_numpy(image)

    def prepare_data_info_list(self):
        """데이터 정보 리스트 준비"""
        frost_conditions = {'서리', '서릿발', '동로', '무빙', '수상', '수빙'}
        
        for data_info in self.data_info_list:
            label_type = data_info['label_type'].lower()
            print(f'== Preparing {label_type}...\n')
            
            label_table = pd.read_csv(LABEL_DIR[label_type], index_col=0, parse_dates=True)
            
            start_date_str = data_info['start_date_str']
            end_date_str = data_info['end_date_str']
            hour_col_pairs = data_info['hour_col_pairs']
            label_keys = data_info['label_keys']
            
            data_dict_list = []
            for date in tqdm(pd.date_range(start_date_str, end_date_str, freq='D'), 
                           desc=f"Processing {label_type}", ncols=100):
                date_str = date.strftime('%Y%m%d')
                
                for hour, label_col in hour_col_pairs:
                    date_hour = date + timedelta(hours=hour)
                    
                    # 이미지 파일 존재 여부 확인
                    date_hour_delta_list = [
                        date_hour + timedelta(hours=delta) for delta in self.time_range
                    ]
                    
                    valid = True
                    for date_hour_delta in date_hour_delta_list:
                        date_hour_delta_str = date_hour_delta.strftime('%Y%m%d%H%M')
                        file_path = os.path.join(
                            self.data_path, 
                            date_hour_delta_str[:8], 
                            f'{self.channels}_{date_hour_delta_str}.npy'
                        )
                        if not os.path.exists(file_path):
                            valid = False
                            break
                    
                    if not valid:
                        continue
                    
                    data_dict = {
                        'date_hour': date_hour.strftime('%Y%m%d%H%M'),
                        'image_list': [d.strftime('%Y%m%d%H%M') for d in date_hour_delta_list],
                        'label_col': label_col,
                        'label_dict': {},
                    }
                    
                    # 여름철(5-9월)은 서리 없음
                    if date.month in [5, 6, 7, 8, 9]:
                        data_dict['label_dict'] = {k: 0 for k in label_keys}
                        data_dict_list.append(data_dict)
                        continue
                    
                    # 라벨 처리
                    for label_key in label_keys:
                        row = label_table[
                            (label_table.index == date) & 
                            (label_table['key'] == int(label_key))
                        ]
                        
                        if row.empty:
                            data_dict['label_dict'][label_key] = (
                                np.nan if label_type == 'aafos' else 0
                            )
                            continue
                        
                        frost_record = row.iloc[0][label_col]
                        if any(cond in str(frost_record) for cond in frost_conditions):
                            data_dict['label_dict'][label_key] = 1
                        else:
                            data_dict['label_dict'][label_key] = 0
                    
                    data_dict_list.append(data_dict)
            
            data_info['data_dict_list'] = data_dict_list
            print(f'\n  - Total {len(data_dict_list)} image-label pairs prepared')
            
            # 저장
            os.makedirs(self.output_path, exist_ok=True)
            mode = "train" if self.train else "test"
            yaml_path = f'{self.output_path}/{mode}_{label_type}_image_label_list.yaml'
            yaml.dump(data_dict_list, open(yaml_path, 'w'))
            print(f'  - {yaml_path} saved\n')

    def sync_dataset_length(self):
        """데이터셋 길이 동기화"""
        lengths = [len(d['data_dict_list']) for d in self.data_info_list]
        self.length = max(lengths)
        
        for data_info in self.data_info_list:
            length = len(data_info['data_dict_list'])
            share = self.length // length
            remainder = self.length % length
            
            indices = list(range(length)) * share + random.sample(range(length), remainder)
            data_info['data_indices'] = np.array(indices)
            
            print(f'== {data_info["label_type"]} dataset length synced to {len(indices)}')

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        Returns:
            list of tuples: [(x_raw, y_raw, labels, coords), ...] for each label type
        """
        data = []
        
        for data_info in self.data_info_list:
            index = data_info['data_indices'][idx]
            data_dict = data_info['data_dict_list'][index]
            
            label_type = data_info['label_type'].lower()
            image_date_list = data_dict['image_list']
            label_dict = data_dict['label_dict']
            
            # 이미지 로드 및 전처리
            images = []
            for image_date in image_date_list:
                image_path = os.path.join(
                    self.data_path, 
                    image_date[:8], 
                    f'{self.channels}_{image_date}.npy'
                )
                image = np.load(image_path).astype(np.float32)
                
                # Shape 처리: (H, W) -> (C, H, W) or (C, H, W) -> (C, H, W)
                if len(image.shape) == 2:
                    image = np.expand_dims(image, axis=0)
                
                # Resize
                image = np.transpose(image, (1, 2, 0))  # (H, W, C)
                image = cv2.resize(image, (self.image_size, self.image_size), 
                                 interpolation=cv2.INTER_CUBIC)
                image = np.transpose(image, (2, 0, 1))  # (C, H, W)
                
                # Calibration
                if self.calib is not None:
                    for i, channel in enumerate(self.calib):
                        if channel in ['ir087', 'ir096', 'ir105', 'ir112', 
                                      'ir123', 'ir133', 'sw038', 'wv063', 
                                      'wv069', 'wv073']:
                            clip_range = (190, 310)
                        elif channel in ['vi004', 'vi005', 'vi006', 'vi008', 
                                        'nr013', 'nr016']:
                            clip_range = (0.05, 1.0)
                        else:
                            clip_range = None
                        
                        calibrated = self.calibrate(image[i], channel)
                        if clip_range:
                            calibrated = torch.clamp(calibrated, clip_range[0], clip_range[1])
                        image[i] = calibrated.numpy()
                
                image = torch.from_numpy(image)  # (C, H, W)
                images.append(image)
            
            # Stack: (T, C, H, W)
            images = torch.stack(images, dim=0)
            
            # Misc channels 추가
            if self.misc_images is not None:
                # misc_images: (C_misc, H, W) -> (T, C_misc, H, W)
                misc_expanded = self.misc_images.unsqueeze(0).expand(len(images), -1, -1, -1)
                images = torch.cat([images, misc_expanded], dim=1)
            
            # Labels
            labels = torch.tensor([v for v in label_dict.values()], dtype=torch.float32)
            
            # Patch 처리
            if self.patch:
                candidates = self.patch_candidates[label_type]
                coords = candidates[np.random.randint(0, len(candidates))]
                
                x0, y0 = coords[0], coords[1]
                images = images[:, :, y0:y0+self.patch_size, x0:x0+self.patch_size]
                coords = torch.tensor(coords, dtype=torch.long)
            else:
                coords = torch.tensor([0, 0], dtype=torch.long)
            
            # Transform 적용
            if self.transform:
                # transform은 (C, H, W) 기대 -> (T, C, H, W)에 대해 각 프레임 적용
                T_len = images.shape[0]
                transformed = []
                for t in range(T_len):
                    transformed.append(self.transform(images[t]))
                images = torch.stack(transformed, dim=0)
            
            # x_raw와 y_raw는 동일 (reconstruction 학습)
            x_raw = images
            y_raw = images.clone()
            
            data.append((x_raw, y_raw, labels, coords))
        
        return data


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    
    # 테스트 설정
    data_path = '/home/work/js/repo/data/kma_data/date_kst'
    output_path = 'results_test'
    channels = '16ch'
    calib = ['vi004', 'vi005', 'vi006', 'vi008', 'nr013', 'nr016', 
             'sw038', 'wv063', 'wv069', 'wv073', 'ir087', 'ir096', 
             'ir105', 'ir112', 'ir123', 'ir133']
    time_range = list(range(0, -12, -3))
    image_size = 512
    patch_size = 256
    
    misc_images = torch.randn(3, image_size, image_size)
    
    train_data_info_list = [
        {
            'label_type': 'asos',
            'start_date_str': '20240101',
            'end_date_str': '20240131',
            'hour_col_pairs': [(6, 'AM')],
            'label_keys': ['93', '108', '112']
        }
    ]
    
    patch_candidates = {
        'asos': np.array([[50, 100], [100, 150], [200, 250]])
    }
    
    print("Creating dataset...")
    dataset = GK2ADataset_stage(
        data_path=data_path,
        output_path=output_path,
        data_info_list=train_data_info_list,
        channels=channels,
        time_range=time_range,
        channels_calib=calib,
        image_size=image_size,
        misc_images=misc_images,
        patch_size=patch_size,
        patch_candidates=patch_candidates,
        train=True
    )
    
    dataset.patchfy(True)
    
    print(f"\nDataset length: {len(dataset)}")
    
    # 샘플 확인
    batch = dataset[0]
    x_raw, y_raw, labels, coords = batch[0]
    print(f"x_raw shape: {x_raw.shape}")
    print(f"y_raw shape: {y_raw.shape}")
    print(f"labels shape: {labels.shape}")
    print(f"coords: {coords}")
