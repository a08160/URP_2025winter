#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import netCDF4 as nc


import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import transforms

from utils import *

from sklearn.metrics import roc_auc_score, confusion_matrix
from torchvision.ops import sigmoid_focal_loss
from torch.nn.functional import binary_cross_entropy_with_logits

import torch.nn as nn

# ResNet18 + gSTA + DeepLabV3+ 모델 (openstl 대신 직접 구현)
from model.resnet18_simvpv2 import FrostResNetgSTA


# In[3]:


# ====== 실험 설정 ====== # 중요함
# 환경변수로 하이퍼파라미터를 받아서 터미널에서 다양한 설정으로 실행 가능
# 예: N_T=2 EPOCHS=30 python resnet18-simvpv2.py

import os
import json

def get_env(key, default, type_fn=str):
    """환경변수에서 값을 가져오고, 없으면 기본값 사용"""
    val = os.environ.get(key, None)
    if val is None:
        return default
    # 리스트 타입 처리 (JSON 형식)
    if type_fn == list:
        return json.loads(val)
    return type_fn(val)

seeds = [0, 1, 2]

# 사용할 데이터 종류 설정, 단일 학습인지 멀티테스크인지 설정 (URP 에서 AAFOS FALSE 설정)
ASOS = True
AAFOS = False
assert ASOS or AAFOS, "At least one of ASOS or AAFOS must be True."

channels = '16ch' # '16ch' or 'ae304', 16ch일때 [16,512,512]
resolution = '2km' # '1km':1024 or '2km':512

# ====== 환경변수로 받는 하이퍼파라미터들 ====== #
# 데이터 관련
time_range = get_env('TIME_RANGE', [-16,-15,-14,-13,-12], list)  # 예: TIME_RANGE='[-21,-18,-15,-12]'

# 학습 관련
batch_size = get_env('BATCH_SIZE', 32, int)
num_workers = get_env('NUM_WORKERS', 8, int)
epochs = get_env('EPOCHS', 25, int)
lr = get_env('LR', 1e-3, float)
decay = get_env('DECAY', [10, 20], list)  # 예: DECAY='[10,20]'
lr_decay = get_env('LR_DECAY', 0.1, float)
weight_decay = get_env('WEIGHT_DECAY', 1e-5, float)
threshold = get_env('THRESHOLD', [0.25], list)  # 예: THRESHOLD='[0.25,0.5]'

# 모델 하이퍼파라미터
hid_T = get_env('HID_T', 16, int)
N_T = get_env('N_T', 6, int)
output_stride = get_env('OUTPUT_STRIDE', 8, int)
drop = get_env('DROP', 0.0, float)

# postfix (실험 이름)
postfix = get_env('POSTFIX', f'resnet_simvpv2_192_{batch_size}_range{abs(time_range[0])}_os{output_stride}_hidT{hid_T}_NT{N_T}_drop{int(drop*100)}', str)

# ====== 모델 config 딕셔너리 ====== #
model_config = {
    'hid_T': hid_T,
    'N_T': N_T,
    'output_stride': output_stride,
    'drop': drop,
}

# ====== 출력 경로 설정 ====== #
output_path = "results/"
output_path += 'asos_' if ASOS else ''
output_path += 'aafos_' if AAFOS else ''
output_path += channels + '_'
output_path += 'time' + str(time_range) + '_'
output_path += resolution
output_path += ('_' + postfix) if postfix != '' else postfix

print("=" * 60)
print("실험 설정 (환경변수로 변경 가능)")
print("=" * 60)
print(f"Output path: {output_path}")
print(f"")
print(f"[데이터]")
print(f"  TIME_RANGE: {time_range}")
print(f"")
print(f"[학습]")
print(f"  BATCH_SIZE: {batch_size}")
print(f"  NUM_WORKERS: {num_workers}")
print(f"  EPOCHS: {epochs}")
print(f"  LR: {lr}")
print(f"  DECAY: {decay}")
print(f"  LR_DECAY: {lr_decay}")
print(f"  WEIGHT_DECAY: {weight_decay}")
print(f"  THRESHOLD: {threshold}")
print(f"")
print(f"[모델]")
print(f"  HID_T: {hid_T}")
print(f"  N_T: {N_T}")
print(f"  OUTPUT_STRIDE: {output_stride}")
print(f"  DROP: {drop}")
print(f"")
print(f"  model_config: {model_config}")
print("=" * 60)

# 기본 asos:aafos 비율 5:1
asos_aafos_ratio = 5.0
asos_weight = asos_aafos_ratio / (asos_aafos_ratio + 1.0 * AAFOS) if ASOS else 0.0
aafos_weight = 1.0 / (asos_aafos_ratio * ASOS + 1.0) if AAFOS else 0.0
print(f"ASOS weight: {asos_weight:.2f}, AAFOS weight: {aafos_weight:.2f}")

# 기타 ablation 설정
latlon = False

# 실험중인 설정들
central_patch = False
use_patch = False  # 192x192 전체 이미지 사용

#  ====== 채널 설정 ====== #
if channels == '16ch':
    channels_name = ['vi004','vi005','vi006','vi008','nr013','nr016','sw038','wv063','wv069','wv073','ir087','ir096','ir105','ir112','ir123','ir133'] # 시각화 용
    channels_calib = ['vi004','vi005','vi006','vi008','nr013','nr016','sw038','wv063','wv069','wv073','ir087','ir096','ir105','ir112','ir123','ir133']
    
    channels_mean = [1.1912e-01, 1.1464e-01, 1.0734e-01, 1.2504e-01, 5.4983e-02, 9.0381e-02,
                2.7813e+02, 2.3720e+02, 2.4464e+02, 2.5130e+02, 2.6948e+02, 2.4890e+02,
                2.7121e+02, 2.7071e+02, 2.6886e+02, 2.5737e+02]
    channels_std  = [0.1306,  0.1303,  0.1306,  0.1501,  0.0268,  0.0838, 15.8211,  6.1468,
                7.8054,  9.3251, 16.4265,  9.6150, 17.2518, 17.6064, 17.0090, 12.5026]

else:
    raise ValueError("Invalid channels.")


train_data_info_list = []
train_data_info_list.append({
    'label_type': 'asos', # 'asos' or 'aafos'
    'start_date_str': '20200101', #  라벨기준 일자. KST
    'end_date_str': '20230630',
    'hour_col_pairs': [(6,'AM')],
    'label_keys': ['93','108','112','119','131','133','136','143','146','156','177','102','104','115','138','152','155','159','165','168','169','184','189']
}) if ASOS else None

test_asos_data_info_list = [
    {
        'label_type': 'asos', # 'asos' or 'aafos'
        'start_date_str': '20230701', #  라벨기준 일자. KST
        'end_date_str': '20240630',
        'hour_col_pairs': [(6,'AM')],
        'label_keys': ['93','108','112','119','131','133','136','143','146','156','177','102','104','115','138','152','155','159','165','168','169','184','189']
    },
] if ASOS else None


origin_size = 900 if resolution == '2km' else 1800
image_size = 192  # 192x192 크기 (384 -> 192 resize)
patch_size = 192  # 패치 크기

# 384x384로 이미 crop된 데이터 경로 (GK2ADataset에서 384->192 resize 수행)
#data_path = '/home/yulim/woncrab/data_kst_URP_384'
data_path = '/home/yulim/woncrab/data_kst_1h/urp_processed'

misc_channels = {
    'elevation':'elevation_1km_3600.npy',
    'vegetation':'vegetation_1km_3600.npy',
    'watermap':'watermap_1km_avg_3600.npy'
}
lat_lon_path = 'assets/gk2a_ami_ko010lc_latlon.nc'

# 자동으로 설정되는 값 (get_station_map 함수 사용 권장)
asos_x_base, asos_y_base, asos_image_size = get_crop_base(image_size, label_type='asos')
aafos_x_base, aafos_y_base, aafos_image_size = get_crop_base(image_size, label_type='aafos')
aafos_x_base -= asos_x_base
aafos_y_base -= asos_y_base

seq_len = len(time_range)

total_channels = len(channels_name) + len(misc_channels.keys())
total_channels += 2 if latlon else 0
print(f"total_channels: {total_channels}")
print(f"image_size: {image_size}, patch_size: {patch_size}")


# In[4]:


# get_station_map 함수를 사용하여 관측소 좌표 계산
# input_size=384: 이미 384x384로 crop된 데이터 사용
asos_map_dict = get_station_map(image_size, label_type='asos', origin_size=origin_size)

# Land와 Coast 관측소 분리 (시각화용)
asos_land_keys = list(ASOS_LAND_COORD.keys())
asos_coast_keys = list(ASOS_COAST_COORD.keys())
asos_land_map = {k: v for k, v in asos_map_dict.items() if k in asos_land_keys}
asos_coast_map = {k: v for k, v in asos_map_dict.items() if k in asos_coast_keys}

print(f"Image size: {image_size}")
print(f"ASOS station keys: {list(asos_map_dict.keys())}")

# misc_images 처리: 3600 -> 512 resize -> 384 crop -> 192 resize
CROP_BASE = 64  # 512 -> 384 crop offset
CROP_SIZE_384 = 384

image = np.load('assets/misc_channels/watermap_1km_avg_3600.npy', allow_pickle=True)
image = -image + 1.0
image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_CUBIC)
if image_size == 384 or image_size == 192:
    image = image[CROP_BASE:CROP_BASE+CROP_SIZE_384, CROP_BASE:CROP_BASE+CROP_SIZE_384]  # 384x384로 crop
    if image_size == 192:
        image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)  # 384 -> 192로 resize

plt.figure(figsize=(8, 8))
plt.imshow(image, cmap='gray', vmin=-1, vmax=2)

for k, v in asos_land_map.items():
    plt.scatter(v[0], v[1], c='r', marker='o', s=10)
for k, v in asos_coast_map.items():
    plt.scatter(v[0], v[1], c='b', marker='o', s=10)

plt.scatter([], [], c='r', marker='o', label='ASOS Land')
plt.scatter([], [], c='b', marker='o', label='ASOS Coast')

plt.title(f'ASOS Frost Observation Sites ({image_size}x{image_size})')
plt.legend()
plt.show()


# In[5]:


if channels_mean is None:
    date_list = os.listdir(data_path)
    date_list = [date for date in date_list if '20200101' <= date < '20240101']

    npy_list = []
    for date in tqdm(date_list[::10]):
        for file in os.listdir(os.path.join(data_path, date)):
            if channels not in file:
                continue
            npy = np.load(os.path.join(data_path, date, file), allow_pickle=True).astype(np.float32) # (C, H, W)
            npy_list.append(npy)
    npy_array = np.stack(npy_list, axis=0)
    print(npy_array.shape) # (N, C, H, W)

    channels_mean = npy_array.mean(axis=(0,2,3)).tolist()
    channels_std = npy_array.std(axis=(0,2,3)).tolist()

print(channels_mean)
print(channels_std)


# In[6]:


lat_lon_data = nc.Dataset(lat_lon_path)
lat = lat_lon_data['lat'][:].data
lon = lat_lon_data['lon'][:].data

lat = cv2.resize(lat, (origin_size, origin_size), interpolation=cv2.INTER_CUBIC)
lon = cv2.resize(lon, (origin_size, origin_size), interpolation=cv2.INTER_CUBIC)
print(lat.shape, lon.shape)

# lat/lon 처리: origin_size에서 384 영역 crop 후 image_size로 resize
CROP_SIZE_384 = 384

# 384 영역 crop (asos_x_base, asos_y_base는 384 기준 좌표)
asos_lat_384 = lat[asos_y_base:asos_y_base+CROP_SIZE_384, asos_x_base:asos_x_base+CROP_SIZE_384]
asos_lon_384 = lon[asos_y_base:asos_y_base+CROP_SIZE_384, asos_x_base:asos_x_base+CROP_SIZE_384]

# image_size에 따라 resize
if image_size == 192:
    asos_lat = cv2.resize(asos_lat_384.astype(np.float32), (image_size, image_size), interpolation=cv2.INTER_AREA)
    asos_lon = cv2.resize(asos_lon_384.astype(np.float32), (image_size, image_size), interpolation=cv2.INTER_AREA)
else:
    asos_lat = asos_lat_384.astype(np.float32)
    asos_lon = asos_lon_384.astype(np.float32)
print(f'asos_lat shape: {asos_lat.shape}, asos_lon shape: {asos_lon.shape}')

lcc = ccrs.LambertConformal(central_longitude=126, central_latitude=38, standard_parallels=(30, 60))
proj = ccrs.PlateCarree()


# In[7]:


"""
MISC 0: Elevation (고도)
MISC 1: Water Bodies Index (수역 여부)
MISC 2: Vegetation Index (식생 여부)

처리 경로: 3600 -> 512 resize -> 384 crop -> (192 resize if image_size==192)
"""
CROP_BASE = 64  # 512 -> 384 crop offset
CROP_SIZE_384 = 384

misc_images = []
for misc_channel, misc_path in misc_channels.items():
    misc_image = np.load(f'assets/misc_channels/{misc_path}', allow_pickle=True)
    # 3600 -> 512로 resize
    misc_image = cv2.resize(misc_image, (512, 512), interpolation=cv2.INTER_CUBIC)
    # 512 -> 384 crop (위성 이미지와 동일한 영역)
    misc_image = misc_image[CROP_BASE:CROP_BASE+CROP_SIZE_384, CROP_BASE:CROP_BASE+CROP_SIZE_384]
    # image_size가 192인 경우 384 -> 192 resize
    if image_size == 192:
        misc_image = cv2.resize(misc_image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    misc_images.append(misc_image)
misc_images = np.stack(misc_images, axis=0)
misc_images = torch.tensor(misc_images, dtype=torch.float32)

if latlon:
    lat_image = cv2.resize(asos_lat, (asos_image_size, asos_image_size), interpolation=cv2.INTER_CUBIC)
    lon_image = cv2.resize(asos_lon, (asos_image_size, asos_image_size), interpolation=cv2.INTER_CUBIC)
    
    lat_image = torch.tensor(lat_image, dtype=torch.float32).unsqueeze(0)
    lon_image = torch.tensor(lon_image, dtype=torch.float32).unsqueeze(0)
    misc_images = torch.cat([misc_images, lat_image, lon_image], dim=0)

print(f'MISC images shape: {misc_images.shape}')

# normalize by channel
misc_mean = misc_images.mean(dim=[1,2], keepdim=True)
misc_std = misc_images.std(dim=[1,2], keepdim=True)

total_mean = channels_mean + misc_mean.squeeze().tolist()
total_std = channels_std + misc_std.squeeze().tolist()

print(f'Total mean: {total_mean}')
print(f'Total std: {total_std}')

fig, axs = plt.subplots(math.ceil(misc_images.shape[0]/3), 3, figsize=(9, 5))
for i in range(misc_images.shape[0]):
    ax = axs.flatten()[i]
    ax.imshow(misc_images[i].numpy(), cmap='viridis')
    ax.set_title(f'MISC {i}')
    colorbar = plt.colorbar(mappable=ax.images[0], ax=ax, fraction=0.046, pad=0.04)
    ax.axis('off')
for i in range(misc_images.shape[0], len(axs.flatten())):
    ax = axs.flatten()[i]
    ax.axis('off')
plt.tight_layout()
plt.show()


# In[8]:


# 192x192에서는 전체 이미지 사용 (패치 불필요)
# patch_candidates는 형식상 정의만 해둠
asos_patch_candidate = np.zeros([image_size, image_size], dtype=np.uint8)
for x, y in asos_map_dict.values():
    y_min = np.clip(y - 3*patch_size//4, 0, image_size - patch_size+1)
    y_max = np.clip(y - patch_size//4, 0, image_size - patch_size+1)
    x_min = np.clip(x - 3*patch_size//4, 0, image_size - patch_size+1)
    x_max = np.clip(x - patch_size//4, 0, image_size - patch_size+1)
    asos_patch_candidate[y_min:y_max, x_min:x_max] = 1
    plt.scatter(x, y, c='r', marker='o', s=5)
plt.imshow(misc_images[2], cmap='viridis', vmin=-1, vmax=2)
plt.imshow(asos_patch_candidate, cmap='gray', vmin=0, vmax=1, alpha=0.5)
asos_patch_candidate = np.argwhere(asos_patch_candidate == 1)[:, [1, 0]]
print(f'ASOS patch candidates: {asos_patch_candidate.shape}')
plt.title(f'ASOS Patch Candidates ({image_size}x{image_size})')
plt.show()

patch_candidates = {'asos': asos_patch_candidate}


# In[9]:


# === 이미지 + ASOS 좌표 시각적 검증 ===
import numpy as np
import matplotlib.pyplot as plt

# 1. 샘플 위성 이미지 로드 (이미 384x384로 crop된 데이터)
sample_date = '20230101'
sample_file = f'{data_path}/{sample_date}/16ch_{sample_date}0600.npy'
sample_image = np.load(sample_file).astype(np.float32)

# 384x384 데이터를 192로 resize (crop 불필요)
if image_size == 192:
    sample_image_resized = np.zeros((sample_image.shape[0], 192, 192), dtype=np.float32)
    for i in range(sample_image.shape[0]):
        sample_image_resized[i] = cv2.resize(sample_image[i], (192, 192), interpolation=cv2.INTER_AREA)
    sample_image = sample_image_resized

# 2. IR 채널로 배경 표시
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 좌: 위성 이미지 + ASOS 위치
ax1 = axes[0]
ax1.imshow(sample_image[10], cmap='gray')  # IR087 채널
for k, (x, y) in asos_land_map.items():
    ax1.scatter(x, y, c='red', s=50, edgecolors='white', linewidth=1)
    ax1.annotate(str(k), (x, y), xytext=(3, 3), textcoords='offset points', 
                fontsize=7, color='red')
for k, (x, y) in asos_coast_map.items():
    ax1.scatter(x, y, c='blue', s=50, edgecolors='white', linewidth=1)
    ax1.annotate(str(k), (x, y), xytext=(3, 3), textcoords='offset points', 
                fontsize=7, color='blue')
ax1.set_title(f'Satellite Image (IR087) + ASOS Stations ({image_size}x{image_size})')
ax1.legend(['Land', 'Coast'], loc='upper right')

# 우: 지형 맵(watermap) + ASOS 위치
ax2 = axes[1]
# misc_images[2]가 watermap
watermap = misc_images[2].numpy() if torch.is_tensor(misc_images[2]) else misc_images[2]
ax2.imshow(-watermap + 1, cmap='terrain')
for k, (x, y) in asos_land_map.items():
    ax2.scatter(x, y, c='red', s=50, edgecolors='white', linewidth=1)
for k, (x, y) in asos_coast_map.items():
    ax2.scatter(x, y, c='blue', s=50, edgecolors='white', linewidth=1)
ax2.set_title(f'Watermap + ASOS Stations ({image_size}x{image_size})')

plt.tight_layout()
plt.show()

# 3. 좌표 범위 검증 출력
print(f"\n=== ASOS 좌표 범위 검증 ({image_size}x{image_size}) ===")
all_coords = list(asos_land_map.values()) + list(asos_coast_map.values())
xs = [c[0] for c in all_coords]
ys = [c[1] for c in all_coords]
print(f"X 범위: {min(xs)} ~ {max(xs)} (이미지: 0~{image_size-1})")
print(f"Y 범위: {min(ys)} ~ {max(ys)} (이미지: 0~{image_size-1})")
print(f"범위 내 관측소: {sum(1 for x,y in all_coords if 0<=x<image_size and 0<=y<image_size)}/23")


# In[10]:


print(f"image_size: {image_size}, patch_size: {patch_size}")

transform = transforms.Compose([
    transforms.Normalize(mean=total_mean, std=total_std)
])

train_dataset = GK2ADataset(data_path=data_path, output_path=output_path, data_info_list=train_data_info_list,
                            channels=channels, time_range=time_range, channels_calib=channels_calib, image_size=image_size, misc_images=misc_images,
                            patch_size=patch_size, patch_candidates=patch_candidates, transform=transform, train=True, return_sequence=True)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

if ASOS:
    test_asos_dataset = GK2ADataset(data_path=data_path, output_path=output_path, data_info_list=test_asos_data_info_list,
                            channels=channels, time_range=time_range, channels_calib=channels_calib, image_size=image_size, misc_images=misc_images,
                            patch_size=patch_size, patch_candidates=None, transform=transform, train=False, return_sequence=True)
    test_asos_dataloader = DataLoader(test_asos_dataset, batch_size=batch_size//2, shuffle=False, num_workers=num_workers, drop_last=False)


# In[11]:


# 데이터 시각화 (SimVPv2: return_sequence=True -> (T, C, H, W))
train_dataset.patchfy(True)
batch = train_dataset[7]

for bi in range(len(batch)):
    images, label, coords = batch[bi]
    print(f"images: {tuple(images.shape)}, label: {tuple(label.shape)}, coords: {tuple(coords.shape)}")

    # (T, C, H, W) -> 각 시점별로 시각화
    T, C, H, W = images.shape
    print(f"  -> T={T}, C={C}, H={H}, W={W}")
    
    for t in range(T):
        frame = images[t]  # (C, H, W)
        print(f"\n=== Time step t={t} (time_range={time_range[t]}h) ===")
        
        fig, axs = plt.subplots(math.ceil(C/4), 4, figsize=(12, 2.5 * math.ceil(C/4)), subplot_kw={'projection': lcc})
        for c in range(len(axs.flatten())):
            ax = axs.flatten()[c]
            if c >= C:
                ax.axis('off')
                continue

            image = np.ones((asos_image_size, asos_image_size), dtype=np.float32)
            image *= -1.0
            image[coords[1]:coords[1]+patch_size, coords[0]:coords[0]+patch_size] = frame[c].numpy()

            ax.set_aspect('equal')
            ax.axis('off')
            
            im = ax.pcolormesh(asos_lon, asos_lat, image, cmap='viridis', transform=proj)
                
            ax.coastlines(resolution='10m', color='black', linewidth=0.5)
            ax.add_feature(cfeature.BORDERS.with_scale('10m'), edgecolor='black', linewidth=0.5)

            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
        plt.suptitle(f't={t} ({time_range[t]}h)')
        plt.tight_layout()
        plt.show()


# In[12]:


# 모델 생성 테스트 (FrostResNetgSTA: ResNet18 + gSTA + DeepLabV3+)
model = FrostResNetgSTA(
    seq_len=seq_len, 
    in_channels=total_channels, 
    img_size=patch_size,
    out_channels=1,
    pretrained=True,
    **model_config  # hid_T, N_T, output_stride, drop
)
print(f"Model input: (B, T={seq_len}, C={total_channels}, H={patch_size}, W={patch_size})")
print(f"Model config: {model_config}")

# 파라미터 수 출력
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")


# In[13]:


def calc_measure_valid(Y_test, Y_test_hat, cutoff=0.5):
    Y_test = Y_test.ravel()
    Y_test_hat = Y_test_hat.ravel()
    
    Y_valid = (~np.isnan(Y_test))
    Y_test = Y_test[Y_valid]
    Y_test_hat = Y_test_hat[Y_valid]

    cfmat = confusion_matrix(Y_test, Y_test_hat > cutoff, labels = [0,1])
    acc = np.trace(cfmat) / np.sum(cfmat)
    csi = cfmat[1,1] /(np.sum(cfmat) - cfmat[0,0] + 1e-8)
    
    try:
        auroc = roc_auc_score(Y_test, Y_test_hat)
    except Exception as e:
        auroc = 0.0

    return csi, acc, auroc


# In[ ]:


# --- 학습 함수 정의 ---

def train_step(model, images, labels, coords, map_dict, cls_idx=0):
    """FrostResNetgSTA 학습 스텝 (입력: (B, T, C, H, W))"""
    images = images.cuda(non_blocking=True)
    labels = labels.cuda(non_blocking=True)

    pred_map = model(images)[:, cls_idx]  # (B, H, W)
    pred_vec = torch.zeros_like(labels)
    
    for b, (px, py) in enumerate(coords):
        for i, (x, y) in enumerate(map_dict.values()):
            if px <= x < px + patch_size and py <= y < py + patch_size:
                pred_vec[b, i] = pred_map[b, y - py, x - px]
            else:
                pred_vec[b, i] = labels[b, i]

    labels_valid = (~torch.isnan(labels)).float()
    labels = torch.nan_to_num(labels, 0.0)

    loss_focal_raw = sigmoid_focal_loss(pred_vec, labels, alpha=-1, gamma=2, reduction='none')
    loss_focal = (loss_focal_raw * labels_valid).sum() / labels_valid.sum().clamp(min=1.0)

    valid_any_batch = (labels_valid.sum() > 0)
    all_zero_batch = ((labels * labels_valid).sum() == 0)

    if not (valid_any_batch and all_zero_batch):
        loss_non_frost = torch.tensor(0.0, device=labels.device)
    else:
        loss_non_frost = binary_cross_entropy_with_logits(
            pred_map, torch.zeros_like(pred_map), reduction='mean'
        )

    prob = torch.sigmoid(pred_vec)
    tp = torch.sum(prob * labels * labels_valid, dim=0)
    fn = torch.sum((1 - prob) * labels * labels_valid, dim=0)
    fp = torch.sum(prob * (1 - labels) * labels_valid, dim=0)

    loss_csi = torch.mean(-torch.log(tp + 1e-10) + torch.log(tp + fn + fp + 1e-10))

    return loss_focal + loss_non_frost + loss_csi


def get_cls_idx(task_name: str, out_channels: int):
    if out_channels == 1:
        return 0
    return 0 if task_name.lower() == "asos" else 1


# --- 학습 루프 ---
for seed in seeds:
    if os.path.exists(f'{output_path}/{seed}/ckpt.pt'):
        print(f'Seed {seed} already done. Skipping...')
        continue

    in_channels = total_channels
    out_channels = 2 if (ASOS and AAFOS) else 1

    # FrostResNetgSTA: model_config 참조 (설정 셀에서 정의)
    model = FrostResNetgSTA(
        seq_len=seq_len,
        in_channels=in_channels,
        img_size=patch_size,
        out_channels=out_channels,
        pretrained=True,
        **model_config  # hid_T, N_T, output_stride, drop
    ).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decay, gamma=lr_decay)

    start_epoch = 0

    results = dict(
        loss={'asos': [], 'aafos': [], 'total': []},
        csi={'asos': {}, 'aafos': {}},
        acc={'asos': {}, 'aafos': {}},
        auroc={'asos': [], 'aafos': []},
        best_asos={},
        best_aafos={},
        best_mean={},
    )

    for cutoff_str in threshold:
        cutoff_str = str(cutoff_str)
        results['csi']['asos'][cutoff_str] = []
        results['acc']['asos'][cutoff_str] = []
        results['best_asos'][cutoff_str] = {'csi': 0.0, 'epoch': -1, 'model': None}
        results['csi']['aafos'][cutoff_str] = []
        results['acc']['aafos'][cutoff_str] = []
        results['best_aafos'][cutoff_str] = {'csi': 0.0, 'epoch': -1, 'model': None}
        results['best_mean'][cutoff_str] = {'csi': 0.0, 'epoch': -1, 'model': None}

    if os.path.exists(f'{output_path}/{seed}/resume.pt'):
        resume = torch.load(f'{output_path}/{seed}/resume.pt', weights_only=False)
        model.load_state_dict(resume['model'])
        optimizer.load_state_dict(resume['optimizer'])
        scheduler.load_state_dict(resume['scheduler'])
        start_epoch = resume['epoch'] + 1
        results = resume['results']
        print(f'Resuming from epoch {start_epoch}...')

    for epoch in range(start_epoch, epochs):
        model.train()

        total_loss_asos = 0.0
        total_loss_aafos = 0.0
        total_loss = 0.0

        train_dataset.sync_dataset_length()

        for batch in tqdm(train_dataloader):
            if ASOS:
                images, label, coords = batch[0]
                cls_idx = get_cls_idx("asos", out_channels)
                loss_asos = train_step(model, images, label, coords, asos_map_dict, cls_idx=cls_idx)
            else:
                loss_asos = torch.tensor(0.0, device="cuda")

            if AAFOS:
                images, label, coords = batch[1] if ASOS else batch[0]
                cls_idx = get_cls_idx("aafos", out_channels)
                loss_aafos = train_step(model, images, label, coords, aafos_map_dict, cls_idx=cls_idx)
            else:
                loss_aafos = torch.tensor(0.0, device="cuda")

            loss = asos_weight * loss_asos + aafos_weight * loss_aafos

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss_asos += float(loss_asos.item())
            total_loss_aafos += float(loss_aafos.item())
            total_loss += float(loss.item())

        total_loss_asos /= len(train_dataloader)
        total_loss_aafos /= len(train_dataloader)
        total_loss /= len(train_dataloader)

        results['loss']['asos'].append(total_loss_asos)
        results['loss']['aafos'].append(total_loss_aafos)
        results['loss']['total'].append(total_loss)

        print(f'Epoch {epoch:2d} - Total Loss: {total_loss:.4f}, ASOS Loss: {total_loss_asos:.4f}, AAFOS Loss: {total_loss_aafos:.4f}')
        scheduler.step()

        # ------------------ 평가 ------------------
        model.eval()
        with torch.no_grad():
            asos_results_per_threshold = {}

            if ASOS:
                asos_pred_vec_list = []
                asos_labels_list = []

                for batch in test_asos_dataloader:
                    images, label, coords = batch[0]
                    images = images.cuda(non_blocking=True)

                    pred_map = model.forward_by_patch(images)[:, get_cls_idx("asos", out_channels)]
                    pred_map = torch.sigmoid(pred_map)

                    pred_vec = []
                    for x, y in asos_map_dict.values():
                        pred_vec.append(pred_map[:, y, x])
                    pred_vec = torch.stack(pred_vec, dim=1)

                    asos_pred_vec_list.append(pred_vec.cpu().numpy())
                    asos_labels_list.append(label.numpy())

                pred_vecs = np.concatenate(asos_pred_vec_list, axis=0)
                labels_np = np.concatenate(asos_labels_list, axis=0)

                for cutoff in threshold:
                    asos_result = calc_measure_valid(labels_np, pred_vecs, cutoff=cutoff)
                    csi, acc, auroc = asos_result[0], asos_result[1], asos_result[2]

                    cutoff_str = str(cutoff)
                    results['csi']['asos'][cutoff_str].append(csi)
                    results['acc']['asos'][cutoff_str].append(acc)

                    asos_results_per_threshold[cutoff_str] = asos_result

                    is_best = csi > results['best_asos'][cutoff_str]['csi']
                    print(f'\t - ASOS  (T={cutoff:.2f}): CSI {csi:.4f}, AUROC {auroc:.4f} {"*" if is_best else ""}')

                    if is_best:
                        results['best_asos'][cutoff_str]['csi'] = csi
                        results['best_asos'][cutoff_str]['epoch'] = epoch
                        results['best_asos'][cutoff_str]['model'] = model.state_dict()

                    asos_land_result = calc_measure_valid(labels_np[:, :11], pred_vecs[:, :11], cutoff=cutoff)
                    asos_coast_result = calc_measure_valid(labels_np[:, 11:], pred_vecs[:, 11:], cutoff=cutoff)
                    print(f'\t   - ASOS Land: CSI {asos_land_result[0]:.4f}, AUROC {asos_land_result[2]:.4f}')
                    print(f'\t   - ASOS Coast: CSI {asos_coast_result[0]:.4f}, AUROC {asos_coast_result[2]:.4f}')

                    # 로그 저장
                    log_dir = f'{output_path}/{seed}'
                    log_path = f'{log_dir}/train_log.csv'
                    os.makedirs(log_dir, exist_ok=True)
                    
                    is_first_write = not os.path.exists(log_path)
                    with open(log_path, 'a') as log_f:
                        if is_first_write:
                            log_f.write('Epoch,Threshold,ASOS_Loss,CSI,AUROC,Land_CSI,Coast_CSI\n')
                        log_line = (f'{epoch},{cutoff},{total_loss_asos:.6f},{csi:.6f},{auroc:.6f},'
                                    f'{asos_land_result[0]:.6f},{asos_coast_result[0]:.6f}\n')
                        log_f.write(log_line)

                results['auroc']['asos'].append(auroc)

            print()

        resume = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch,
            'results': results
        }
        os.makedirs(f'{output_path}/{seed}', exist_ok=True)
        torch.save(resume, f'{output_path}/{seed}/resume.pt')

    torch.save(results, f'{output_path}/{seed}/ckpt.pt')
    if os.path.exists(f'{output_path}/{seed}/resume.pt'):
        os.remove(f'{output_path}/{seed}/resume.pt')


# In[ ]:


if ASOS and not AAFOS:
    mode = 'best_asos'

results_df = pd.DataFrame(columns=['type','cutoff','label', 'csi', 'acc', 'auroc'])

results_dict = {}
for cutoff in threshold:
    results_dict[str(cutoff)] = {
        'asos': {},
        'aafos': {}
    }

for seed in seeds:
    ckpt_path = f'{output_path}/{seed}/ckpt.pt'
    if not os.path.exists(ckpt_path):
        print(f'Seed {seed} not found. Skipping...')
        continue
    
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    print(f'--- Processing Seed {seed} ({ckpt_path}) ---')

    # model_config 참조 (설정 셀에서 정의) - 학습과 동일한 구조 보장
    model = FrostResNetgSTA(
        seq_len=seq_len,
        in_channels=total_channels,
        img_size=patch_size,
        out_channels=1,
        pretrained=False,  # 체크포인트에서 로드하므로 pretrained 불필요
        **model_config  # hid_T, N_T, output_stride, drop
    )
    model.cuda()
    
    for cutoff in threshold:
        cutoff_str = str(cutoff)
        
        if cutoff_str not in ckpt[mode]:
            print(f"  Warning: T={cutoff_str} not found for mode '{mode}' in seed {seed}. Skipping.")
            continue
            
        try:
            model_data = ckpt[mode][cutoff_str]
            model.load_state_dict(model_data['model'])
            model.eval()
            print(f"  T={cutoff_str}: Loaded epoch {model_data['epoch']} with {mode} CSI {model_data['csi']:.4f}")
        except Exception as e:
            print(f"  Error loading model for T={cutoff_str}, Seed={seed}: {e}. Skipping.")
            continue
        
        
        with torch.no_grad():
            if ASOS:
                asos_pred_vec_list = []
                asos_labels_list = []
                
                for batch in tqdm(test_asos_dataloader, desc=f"ASOS Eval (Seed {seed}, T={cutoff_str})"):
                    images, label, coords = batch[0]
                    images = images.cuda()
                    
                    pred_map = model.forward_by_patch(images)[:,0]
                    pred_map = F.sigmoid(pred_map)
                    
                    pred_vec = []
                    for x, y in asos_map_dict.values():
                        pred_vec.append(pred_map[:, y, x])
                    pred_vec = torch.stack(pred_vec, dim=1)
                    
                    asos_pred_vec_list.append(pred_vec.cpu().numpy())
                    asos_labels_list.append(label.numpy())
                        
                pred_vecs = np.concatenate(asos_pred_vec_list, axis=0)
                labels = np.concatenate(asos_labels_list, axis=0)
                
                label_cols = [col for _, col in test_asos_data_info_list[0]['hour_col_pairs']]
                for col in label_cols:
                    indices = [i for i, data_dict in enumerate(test_asos_dataset.data_info_list[0]['data_dict_list']) if data_dict['label_col'] == col]
                    pred_vec_selected = pred_vecs[indices]
                    labels_selected = labels[indices]
                    
                    results = calc_measure_valid(labels_selected, pred_vec_selected, cutoff=cutoff)
                    
                    results_dict[cutoff_str]['asos'].setdefault(col, {})
                    results_dict[cutoff_str]['asos'][col].setdefault('csi', []).append(results[0])
                    results_dict[cutoff_str]['asos'][col].setdefault('acc', []).append(results[1])
                    results_dict[cutoff_str]['asos'][col].setdefault('auroc', []).append(results[2])


all_results_rows = []

for cutoff_str, type_dict_for_thr in results_dict.items():
    for data_type, type_dict in type_dict_for_thr.items():
        csi_mean_list = []
        acc_mean_list = []
        auroc_mean_list = []
        
        for label_col, metrics_dict in type_dict.items():
            
            if not metrics_dict.get('csi'):
                print(f"Warning: No data found for T={cutoff_str}, Type={data_type}, Label={label_col}. Skipping row.")
                continue

            csi_mean = np.mean(metrics_dict['csi'])
            csi_mean_list.append(csi_mean)
            csi_std = np.std(metrics_dict['csi'])
            
            acc_mean = np.mean(metrics_dict['acc']) 
            acc_mean_list.append(acc_mean)
            acc_std = np.std(metrics_dict['acc'])
            
            auroc_mean = np.mean(metrics_dict['auroc'])
            auroc_mean_list.append(auroc_mean)
            auroc_std = np.std(metrics_dict['auroc'])

            all_results_rows.append({
                'type': data_type,
                'threshold': cutoff_str,  
                'label': label_col,
                'csi': f'{csi_mean:.4f}',
                'acc': f'{acc_mean:.4f}',
                'auroc': f'{auroc_mean:.4f}',
            })
            
        all_results_rows.append({
            'type': data_type,
            'threshold': cutoff_str,
            'label': 'mean',
            'csi': f'{np.mean(csi_mean_list):.4f}',
            'acc': f'{np.mean(acc_mean_list):.4f}',
            'auroc': f'{np.mean(auroc_mean_list):.4f}',
        })


if all_results_rows:
    results_df = pd.DataFrame(all_results_rows)
else:
    print("Warning: No results were generated.")
print(results_df) 

output_csv_path = f'{output_path}/final_results_{mode}.csv'
results_df.to_csv(output_csv_path, index=False)
print(f"\nFinal results saved to {output_csv_path}")


# In[ ]:


# watermap 시각화 (image_size에 맞게 처리)
water_map = np.load('./assets/misc_channels/watermap_1km_avg_3600.npy')
water_map = cv2.resize(water_map, (512, 512), interpolation=cv2.INTER_CUBIC)

# 384 crop 후 image_size로 resize
CROP_BASE = 64
CROP_SIZE_384 = 384
water_map = water_map[CROP_BASE:CROP_BASE+CROP_SIZE_384, CROP_BASE:CROP_BASE+CROP_SIZE_384]
if image_size == 192:
    water_map = cv2.resize(water_map, (image_size, image_size), interpolation=cv2.INTER_AREA)

water_mask = -1.0 * water_map + 1.0
print(f"water_mask shape: {water_mask.shape}")

