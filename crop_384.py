"""
512x512 데이터를 384x384로 crop하여 저장하는 스크립트
원본: D:/data/date_kst_URP/
저장: D:/data/date_kst_URP_384/
"""

import os
import numpy as np
from tqdm import tqdm
from pathlib import Path

# 경로 설정
SRC_DIR = Path("D:/data/date_kst_URP")
DST_DIR = Path("D:/data/date_kst_URP_384")

# Crop 설정 (512x512 -> 384x384, 중앙 기준)
CROP_SIZE = 384
CROP_START_Y = 64
CROP_START_X = 64

def crop_and_save(src_path: Path, dst_path: Path):
    """
    단일 npy 파일을 crop하고 저장

    Parameters:
    -----------
    src_path : Path
        원본 파일 경로 (shape: 16, 512, 512)
    dst_path : Path
        저장할 파일 경로 (shape: 16, 384, 384)
    """
    # 원본 로드
    data = np.load(src_path)  # (16, 512, 512)

    # Crop
    data_cropped = data[:,
                        CROP_START_Y:CROP_START_Y+CROP_SIZE,
                        CROP_START_X:CROP_START_X+CROP_SIZE]

    # 저장
    np.save(dst_path, data_cropped)

def main():
    # 저장 디렉토리 생성
    DST_DIR.mkdir(parents=True, exist_ok=True)

    # 모든 날짜 폴더 가져오기
    date_folders = sorted([d for d in SRC_DIR.iterdir() if d.is_dir()])

    print(f"원본 경로: {SRC_DIR}")
    print(f"저장 경로: {DST_DIR}")
    print(f"총 날짜 폴더 수: {len(date_folders)}")
    print(f"Crop 설정: 512x512 -> 384x384 (y={CROP_START_Y}:{CROP_START_Y+CROP_SIZE}, x={CROP_START_X}:{CROP_START_X+CROP_SIZE})")
    print("-" * 50)

    total_files = 0

    for date_folder in tqdm(date_folders, desc="Processing dates"):
        # 대상 날짜 폴더 생성
        dst_date_folder = DST_DIR / date_folder.name
        dst_date_folder.mkdir(exist_ok=True)

        # 해당 날짜의 모든 npy 파일 처리
        npy_files = sorted(date_folder.glob("*.npy"))

        for npy_file in npy_files:
            dst_file = dst_date_folder / npy_file.name
            crop_and_save(npy_file, dst_file)
            total_files += 1

    print("-" * 50)
    print(f"완료! 총 {total_files}개 파일 처리됨")

    # 검증: 첫 번째 파일 확인
    sample_src = list(date_folders[0].glob("*.npy"))[0]
    sample_dst = DST_DIR / date_folders[0].name / sample_src.name

    src_data = np.load(sample_src)
    dst_data = np.load(sample_dst)

    print(f"\n검증 (첫 번째 파일):")
    print(f"  원본 shape: {src_data.shape}")
    print(f"  Crop 후 shape: {dst_data.shape}")

if __name__ == "__main__":
    main()
