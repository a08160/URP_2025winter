import math


ASOS_LAND_COORD = {
    93: (37.94738, 127.75443), 
    108: (37.57142, 126.9658), 
    112: (37.47772, 126.6249), 
    119: (37.25746, 126.983),  
    131: (36.63924, 127.44066),
    133: (36.37199, 127.3721), 
    136: (36.57293, 128.70734),
    143: (35.87797, 128.65295),
    146: (35.84092, 127.11718),
    156: (35.17294, 126.89156),
    177: (36.65759, 126.68772), 
}

ASOS_COAST_COORD = {
    102: (37.97396, 124.71237),
    104: (37.80456, 128.85535),
    115: (37.48129, 130.89864),
    138: (36.03201, 129.38002),
    152: (35.58237, 129.33469),
    155: (35.17019, 128.57281),
    159: (35.10468, 129.03203),
    165: (34.81732, 126.38151),
    168: (34.73929, 127.74063),
    169: (34.68719, 125.45105),
    184: (33.51411, 126.52969),
    189: (33.24616, 126.5653)  
}

AAFOS_COORD = {
    1: (37.85155, 128.81924),
    2: (37.67713, 128.71834),
    3: (37.64793, 128.56448),
    4: (37.61234, 128.37725),
    5: (37.56197, 128.37762), 
    6: (37.37748, 128.39469),
    7: (37.95461, 127.77626),
    8: (37.54548, 128.44108),
    9: (37.536505, 128.439104), 
    10: (37.80456, 128.85535), 
    11: (37.75147, 128.89099),
    12: (37.94738, 127.75443),
    # "13": (36.37207, 127.89492), 
    # "14": (36.27943, 128.89601),
}

def coord_to_map(lat, lon, origin_size):
    NX = origin_size    ## X축 격자점 수
    NY = origin_size    ## Y축 격자점 수

    Re = 6371.00877     ##  지구반경
    grid = 0.5 * (3600 / origin_size) # 격자 간격 (km)
    slat1 = 30.0        ##  표준위도 1
    slat2 = 60.0        ##  표준위도 2
    olon = 126.0        ##  기준점 경도
    olat = 38.0         ##  기준점 위도
    xo = (NX-1) / 2     ##  기준점 X좌표
    yo = (NY-1) / 2     ##  기준점 Y좌표

    PI = math.asin(1.0) * 2.0
    DEGRAD = PI / 180.0
    RADDEG = 180.0 / PI

    re = Re / grid
    slat1 = slat1 * DEGRAD
    slat2 = slat2 * DEGRAD
    olon = olon * DEGRAD
    olat = olat * DEGRAD

    sn = math.tan(PI * 0.25 + slat2 * 0.5) / math.tan(PI * 0.25 + slat1 * 0.5)
    sn = math.log(math.cos(slat1) / math.cos(slat2)) / math.log(sn)
    sf = math.tan(PI * 0.25 + slat1 * 0.5)
    sf = math.pow(sf, sn) * math.cos(slat1) / sn
    ro = math.tan(PI * 0.25 + olat * 0.5)
    ro = re * sf / math.pow(ro, sn)

    ra = math.tan(PI * 0.25 + lat * DEGRAD * 0.5)
    ra = re * sf / pow(ra, sn)
    theta = lon * DEGRAD - olon
    if theta > PI:
        theta -= 2.0 * PI
    if theta < -PI:
        theta += 2.0 * PI
    theta *= sn
    x = (ra * math.sin(theta)) + xo
    y = - (ro - ra * math.cos(theta)) + yo
    return int(x), int(y)

def get_crop_base(image_size, label_type=str):
    '''
    900x900 origin image에서 크롭할 x_base, y_base, crop_size 산출

    coord_to_map(lat, lon, 900)과 함께 사용하여 관측소 좌표를 계산합니다.

    데이터 경로:
        - 512x512: 900에서 직접 crop
        - 384x384: 512에서 (64,64) offset으로 crop (900 -> 512 -> 384)

    예시:
        x_base, y_base, crop_size = get_crop_base(image_size=384, label_type='asos')
        asos_map = {k: coord_to_map(*v, 900) for k, v in ASOS_COORD.items()}
        asos_map = {k: (v[0]-x_base, v[1]-y_base) for k, v in asos_map.items()}

    Parameters:
        image_size: target crop size (384 or 512)
        label_type: 'asos' or 'aafos'

    Returns:
        x_base, y_base: 900 기준 crop offset
        crop_size: crop size (aafos는 절반)
    '''
    # 512x512 crop에서 384x384로 추가 crop 시 offset
    CROP_OFFSET_384 = 64

    if label_type.lower() == 'asos':
        # 기본 offset (관측소 중심 맞춤): x=75, y=115
        if image_size == 384:
            # 900 -> 512 -> 384 경로
            # 900->512 base: (900-512)/2 + offset = 194 + 75 = 269, 194 + 115 = 309
            # 512->384 crop offset: 64
            # 최종: 269 + 64 = 333, 309 + 64 = 373
            x_base = (900 - 512) / 2 + 75 + CROP_OFFSET_384
            y_base = (900 - 512) / 2 + 115 + CROP_OFFSET_384
        else:
            # 900 -> image_size 직접 crop
            x_base = (900 - image_size) / 2 + 75
            y_base = (900 - image_size) / 2 + 115
        crop_size = image_size

    elif label_type.lower() == 'aafos':
        # AAFOS는 더 작은 영역 (192x192 기준)
        if image_size == 384:
            x_base = (900 - 192) / 2 + 100 + CROP_OFFSET_384
            y_base = (900 - 192) / 2 + 20 + CROP_OFFSET_384
        else:
            x_base = (900 - 192) / 2 + 100
            y_base = (900 - 192) / 2 + 20
        crop_size = image_size // 2
    else:
        raise ValueError("label_type should be 'AAFOS' or 'ASOS'")

    return int(x_base), int(y_base), crop_size


def get_station_map(image_size, label_type='asos', origin_size=900):
    '''
    관측소의 이미지 좌표를 반환하는 함수

    512x512를 기준으로 좌표를 계산하고, 다른 크기는 512에서 변환합니다.
    이 방식은 verify_crop_384.ipynb에서 검증된 방식입니다.

    Parameters:
        image_size: 타겟 이미지 크기 (384, 512 등)
        label_type: 'asos' or 'aafos'
        origin_size: 원본 이미지 크기 (기본값 900, 2km 해상도)

    Returns:
        dict: {관측소_id: (x, y)} 형태의 좌표 딕셔너리

    Example:
        >>> asos_map = get_station_map(384, 'asos')
        >>> print(asos_map[93])  # (191, 78)
    '''
    # 512 기준 offset (관측소 중심 맞춤)
    OFFSET_512 = {
        'asos': (75, 115),
        'aafos': (100, 20)
    }

    # 512 -> 384 crop offset
    CROP_OFFSET_384 = 64

    label_type = label_type.lower()
    if label_type not in OFFSET_512:
        raise ValueError("label_type should be 'asos' or 'aafos'")

    # 좌표 딕셔너리 선택
    if label_type == 'asos':
        coord_dict = {**ASOS_LAND_COORD, **ASOS_COAST_COORD}
    else:
        coord_dict = AAFOS_COORD

    offset_x, offset_y = OFFSET_512[label_type]

    # 512 crop의 시작점 (origin_size 기준)
    base_512_x = (origin_size - 512) // 2 + offset_x
    base_512_y = (origin_size - 512) // 2 + offset_y

    # origin_size 기준 관측소 좌표 -> 512 기준 좌표
    map_512 = {}
    for k, (lat, lon) in coord_dict.items():
        x, y = coord_to_map(lat, lon, origin_size)
        map_512[k] = (x - base_512_x, y - base_512_y)

    # 타겟 크기에 맞게 변환
    if image_size == 512:
        return map_512
    elif image_size == 384:
        # 512에서 (64, 64) offset으로 crop
        return {k: (v[0] - CROP_OFFSET_384, v[1] - CROP_OFFSET_384)
                for k, v in map_512.items()}
    else:
        # 다른 크기: 512 중심 기준으로 계산
        offset = (512 - image_size) // 2
        return {k: (v[0] - offset, v[1] - offset)
                for k, v in map_512.items()}

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    import cv2
    save_idx = 0
            
    # for origin_size in [900, 1800]:
        
    #     image = np.load('assets/geometry_maps/watermap_1km_avg_3600.npy', allow_pickle=True)
    #     image = -image + 1.0
        
    #     x_base, y_base, image_size = get_crop_base(origin_size, label_type='ASOS')
    #     print(x_base, y_base, image_size)
    #     image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_CUBIC)

    #     plt.figure(figsize=(10, 10))
    #     plt.imshow(image, cmap='gray', vmin=-1, vmax=2)
        
        
    #     ASOS_LAND_MAP = {k: coord_to_map(*v, origin_size) for k, v in ASOS_LAND_COORD.items()}
    #     ASOS_COAST_MAP = {k: coord_to_map(*v, origin_size) for k, v in ASOS_COAST_COORD.items()}
    #     AAFOS_MAP = {k: coord_to_map(*v, origin_size) for k, v in AAFOS_COORD.items()}

        
    #     for k, v in ASOS_LAND_MAP.items():
    #         x, y = v[0]-x_base, v[1]-y_base
    #         plt.scatter(x, y, c='r', marker='o')
    #     for k, v in ASOS_COAST_MAP.items():
    #         x, y = v[0]-x_base, v[1]-y_base
    #         plt.scatter(x, y, c='b', marker='o')
    #     for k, v in AAFOS_MAP.items():
    #         x, y = v[0]-x_base, v[1]-y_base
    #         plt.scatter(x, y, c='g', marker='o')
        
    #     plt.savefig(f'results/test_dataset_projection_{save_idx}.png')
    #     save_idx += 1
        
    


