import os
import netCDF4 as nc
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm
from datetime import datetime, timedelta
import random
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    from .projection import *
except:
    from utils.projection import *

TOTAL_START_DATE = '20200101'
TOTAL_END_DATE = '20250731'

LABEL_DIR = {
    'aafos': 'assets/labels/aafos_2201_2507.csv',
    'asos': 'assets/labels/asos_2001_2507.csv'
}

class GK2ADataset(Dataset):
    def __init__(
        self,
        data_path: str,
        output_path: str,
        data_info_list: list,
        channels: str,
        time_range: tuple,
        channels_calib: list,
        image_size: int,
        misc_images: torch.Tensor,
        patch_size: int = None,
        patch_candidates: dict = None,
        transform = None,
        train: bool = True,
        return_sequence: bool = False,  # SimVPv2용: (T, C, H, W) 반환
        verbose: bool = False,
    ):
        '''
        SimVPv2를 위한 GK2A 데이터셋 (gk2a_dataset.py와 호환)

        Parameters:
        -----------
        - data_path: 데이터 디렉토리 (date_kst_URP_384 등)
        - output_path: 결과 저장 경로
        - data_info_list: 라벨 정보 리스트
        - channels: 채널 타입 ('16ch' 등)
        - time_range: 시간 오프셋 리스트 (예: [-21, -18, -15, -12])
        - channels_calib: calibration 적용할 채널 리스트
        - image_size: 이미지 크기
        - misc_images: 추가 채널 (elevation, vegetation 등)
        - patch_size: 패치 크기
        - patch_candidates: 패치 좌표 후보
        - transform: 데이터 변환
        - train: 학습용 여부
        - return_sequence: True면 (T, C, H, W), False면 (T*C, H, W)
        - verbose: 디버깅 메시지 출력
        '''
        super(GK2ADataset, self).__init__()
        self.data_path = data_path
        self.output_path = output_path
        self.data_info_list = data_info_list if data_info_list is not None else []

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
        self.return_sequence = return_sequence
        self.verbose = verbose

        # Calibration table
        self.calib_table = pd.read_excel(
            'assets/20191115_gk-2a ami calibration table_v3.1_ir133_srf_shift.xlsx',
            sheet_name='Calibration Table_WN', header=0
        )
        self.calib_col = {
            'vi004':'VIS 0.4','vi005':'VIS 0.5','vi006':'VIS 0.6','vi008':'VIS 0.8','nr013':'NIR 1.3',
            'nr016':'NIR 1.6','sw038':'IR 3.8','wv063':'IR 6.3','wv069':'IR 6.9','wv073':'IR 7.3',
            'ir087':'IR 8.7','ir096':'IR 9.6','ir105':'IR 10.5','ir112':'IR 11.2','ir123':'IR 12.3','ir133':'IR 13.3'
        }

        # 데이터 준비
        self.prepare_data_info_list()
        self.sync_dataset_length()
        print("GK2A Dataset initialized\n\n")

    def patchfy(self, bool_val: bool):
        '''패치 모드 설정'''
        if bool_val:
            assert self.patch_size is not None, "patch_size should be specified for patchfy"
            assert self.patch_candidates is not None, "patch_candidates should be provided for patchfy"

        self.patch = bool_val
        print(f'Dataset patchfy set to {self.patch}.\n')

    def calibrate(self, image, channel):
        '''채널별 calibration 적용'''
        H, W = image.shape
        calib_table = self.calib_table
        origin_col = self.calib_col[channel]

        albedo_col = calib_table.columns[calib_table.columns.get_loc(origin_col) + 1]
        calib_col = np.array(calib_table[albedo_col][1:].dropna(), dtype=np.float32)

        image = image.flatten()
        indices = image.astype(np.int32)
        indices = np.clip(indices, 0, len(calib_col) - 1)
        image = calib_col[indices]
        image = image.reshape(H, W)
        image = torch.from_numpy(image)

        return image

    def prepare_data_info_list(self):
        '''데이터 정보 리스트 준비 (파일 존재 확인 포함)'''
        frost_conditions = set(['서리', '서릿발', '동로', '무빙', '수상', '수빙'])

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
                    date_hour = date + timedelta(hours=hour)  # KST
                    date_hour_delta_list = [date_hour + timedelta(hours=delta) for delta in self.time_range]

                    # 파일 존재 확인
                    valid = True
                    for date_hour_delta in date_hour_delta_list:
                        date_hour_delta_str = date_hour_delta.strftime('%Y%m%d%H%M')
                        file_path = os.path.join(
                            self.data_path,
                            date_hour_delta_str[0:8],
                            f'{self.channels}_{date_hour_delta_str}.npy'
                        )

                        if not os.path.exists(file_path):
                            valid = False
                            if self.verbose:
                                print(f'  - {date_str} {label_col} skipped, {date_hour_delta} not in date_table')
                            break

                    if not valid:
                        continue

                    # 이미지 날짜와 라벨 저장
                    data_dict = {
                        'date_hour': date_hour.strftime('%Y%m%d%H%M'),
                        'image_list': [d.strftime('%Y%m%d%H%M') for d in date_hour_delta_list],
                        'label_col': label_col,
                        'label_dict': {},
                    }

                    # 여름철은 모두 0으로 처리
                    if date.month in [5, 6, 7, 8, 9]:
                        data_dict['label_dict'] = {label_key: 0 for label_key in label_keys}
                        data_dict_list.append(data_dict)
                        continue

                    # 라벨 조회
                    for label_key in label_keys:
                        row = label_table[(label_table.index == date) & (label_table['key'] == int(label_key))]

                        if row.empty:
                            data_dict['label_dict'][label_key] = np.nan if label_type == 'aafos' else 0
                            continue

                        frost_record = row.iloc[0][label_col]
                        if any(cond in str(frost_record) for cond in frost_conditions):
                            data_dict['label_dict'][label_key] = 1
                        else:
                            data_dict['label_dict'][label_key] = 0

                    data_dict_list.append(data_dict)

            data_info['data_dict_list'] = data_dict_list
            print(f'\n  - Total {len(data_dict_list)} image-label pairs prepared')

            # YAML 저장
            os.makedirs(self.output_path, exist_ok=True)
            yaml_path = f'{self.output_path}/{"train" if self.train else "test"}_{label_type}_image_label_list.yaml'
            with open(yaml_path, 'w') as f:
                yaml.dump(data_dict_list, f)
            print(f'  - {os.path.basename(yaml_path)} saved\n')

    def sync_dataset_length(self):
        '''데이터셋 길이 동기화'''
        lengths = [len(data_info['data_dict_list']) for data_info in self.data_info_list]
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
        '''데이터 로딩'''
        data = []

        for i, data_info in enumerate(self.data_info_list):
            index = data_info['data_indices'][idx]
            data_dict = data_info['data_dict_list'][index]

            label_type = data_info['label_type'].lower()
            image_date_list = data_dict['image_list']
            label_dict = data_dict['label_dict']

            # 시퀀스로 이미지 로딩
            images = []
            for image_date in image_date_list:
                image_path = os.path.join(
                    self.data_path,
                    image_date[0:8],
                    f'{self.channels}_{image_date}.npy'
                )
                image = np.load(image_path).astype(np.float32)

                # Ensure (C, H, W)
                if len(image.shape) == 2:
                    image = np.expand_dims(image, axis=0)

                # 이미지 크기 처리 (gk2a_dataset.py와 동일)
                CROP_BASE = 64
                CROP_SIZE_384 = 384

                current_h, current_w = image.shape[1], image.shape[2]

                if current_h == 512 and current_w == 512:
                    # 512x512 원본 이미지 처리
                    if self.image_size == 512:
                        pass
                    elif self.image_size == 384:
                        image = image[:, CROP_BASE:CROP_BASE+CROP_SIZE_384, CROP_BASE:CROP_BASE+CROP_SIZE_384]
                    elif self.image_size == 192:
                        image = image[:, CROP_BASE:CROP_BASE+CROP_SIZE_384, CROP_BASE:CROP_BASE+CROP_SIZE_384]
                        image = np.transpose(image, (1,2,0))
                        image = cv2.resize(image, (192, 192), interpolation=cv2.INTER_AREA)
                        image = np.transpose(image, (2,0,1))
                    else:
                        image = np.transpose(image, (1,2,0))
                        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_CUBIC)
                        image = np.transpose(image, (2,0,1))
                elif current_h == 384 and current_w == 384:
                    # 384x384 (이미 crop된 데이터) 처리
                    if self.image_size == 384:
                        pass
                    elif self.image_size == 192:
                        image = np.transpose(image, (1,2,0))
                        image = cv2.resize(image, (192, 192), interpolation=cv2.INTER_AREA)
                        image = np.transpose(image, (2,0,1))
                    else:
                        image = np.transpose(image, (1,2,0))
                        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_CUBIC)
                        image = np.transpose(image, (2,0,1))
                elif current_h != self.image_size or current_w != self.image_size:
                    # 그 외 크기의 이미지는 직접 resize
                    image = np.transpose(image, (1,2,0))
                    image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_CUBIC)
                    image = np.transpose(image, (2,0,1))

                # Calibration
                if self.calib is not None:
                    for ch_idx, channel in enumerate(self.calib):
                        if channel in ['ir087','ir096','ir105','ir112','ir123','ir133','sw038','wv063','wv069','wv073']:
                            clip = (190, 310)
                        elif channel in ['vi004', 'vi005', 'vi006', 'vi008', 'nr013', 'nr016']:
                            clip = (0.05, 1.0)
                        image[ch_idx] = np.clip(self.calibrate(image[ch_idx], channel), clip[0], clip[1])

                image = torch.from_numpy(image)  # (C, H, W)
                images.append(image)

            # 시퀀스 형태로 스택: (T, C, H, W)
            images = torch.stack(images, dim=0)  # (T, C, H, W)

            # Misc channels 추가
            if self.misc_images is not None:
                if self.return_sequence:
                    # misc_images: (C_misc, H, W) -> (T, C_misc, H, W)
                    T = images.shape[0]
                    misc_repeated = self.misc_images.unsqueeze(0).repeat(T, 1, 1, 1)
                    images = torch.cat([images, misc_repeated], dim=1)  # (T, C_total, H, W)
                else:
                    # 기존 방식: flatten 후 concatenate
                    T, C, H, W = images.shape
                    images = images.view(T * C, H, W)  # (T*C, H, W)
                    images = torch.cat([images, self.misc_images], dim=0)  # (T*C + C_misc, H, W)

            labels = torch.tensor([v for k, v in label_dict.items()], dtype=torch.float)  # (N,)

            # Patch cropping
            if self.patch:
                assert self.patch_candidates is not None, "patch_candidates should be provided for patchfy"

                candidates = self.patch_candidates[label_type]
                coords = candidates[np.random.randint(0, candidates.shape[0])]

                if self.return_sequence:
                    # (T, C, H, W) -> (T, C, patch_size, patch_size)
                    images = images[:, :, coords[1]:coords[1]+self.patch_size, coords[0]:coords[0]+self.patch_size]
                else:
                    # (T*C + C_misc, H, W) -> (T*C + C_misc, patch_size, patch_size)
                    images = images[:, coords[1]:coords[1]+self.patch_size, coords[0]:coords[0]+self.patch_size]
            else:
                coords = np.array([0, 0])

            # Transform (각 시점별로 적용)
            if self.transform:
                if self.return_sequence:
                    # (T, C, H, W) -> 각 시점별로 transform 적용
                    T, C, H, W = images.shape
                    transformed = []
                    for t in range(T):
                        transformed.append(self.transform(images[t]))  # (C, H, W)
                    images = torch.stack(transformed, dim=0)  # (T, C, H, W)
                else:
                    # (T*C + C_misc, H, W)
                    images = self.transform(images)

            # Return format
            if self.return_sequence:
                # 이미 (T, C, H, W) 형태
                pass
            elif self.misc_images is None:
                # misc가 없고 return_sequence=False인 경우, 아직 flatten 안됨
                if images.dim() == 4:
                    T, C, H, W = images.shape
                    images = images.view(T * C, H, W)

            data.append((images, labels, coords))

        return data


if __name__ == "__main__":
    # Test code
    data_path = '/home/yulim/woncrab/data_kst_URP_384'
    output_path = 'results'
    channels = '16ch'
    calib = ['vi004','vi005','vi006','vi008','nr013','nr016','sw038','wv063','wv069','wv073','ir087','ir096','ir105','ir112','ir123','ir133']
    time_range = [-21, -18, -15, -12]
    image_size = 384
    patch_size = 384

    misc_images = torch.randn(3, image_size, image_size)

    train_data_info_list = [
        {
            'label_type': 'asos',
            'start_date_str': '20200102',  # 20200101은 전날 데이터 필요하므로 20200102부터
            'end_date_str': '20200105',
            'hour_col_pairs': [(6,'AM')],
            'label_keys': ['93','108','112','119','131','133','136','143','146','156','177','102','104','115','138','152','155','159','165','168','169','184','189']
        }
    ]

    patch_candidates = {
        'asos': np.array([[0, 0]]),
    }

    # Test return_sequence=True
    print("Testing return_sequence=True...")
    dataset = GK2ADataset(
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
        train=True,
        return_sequence=True,
        verbose=True
    )

    dataset.patchfy(True)

    print(f"\nDataset length: {len(dataset)}")

    if len(dataset) > 0:
        images, labels, coords = dataset[0][0]
        print(f"Images shape (return_sequence=True): {images.shape}")  # (T, C, H, W)
        print(f"Labels shape: {labels.shape}")
        print(f"Coords: {coords}")

    # Test return_sequence=False
    print("\n\nTesting return_sequence=False...")
    dataset2 = GK2ADataset(
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
        train=True,
        return_sequence=False,
        verbose=True
    )

    dataset2.patchfy(True)

    if len(dataset2) > 0:
        images2, labels2, coords2 = dataset2[0][0]
        print(f"Images shape (return_sequence=False): {images2.shape}")  # (T*C + C_misc, H, W)
        print(f"Labels shape: {labels2.shape}")
        print(f"Coords: {coords2}")
