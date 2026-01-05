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
    origin image에서 크롭할 x_base, y_base, patch_size 산출
    
    예시:
        x_add, y_add, image_size = get_crop_base(origin_size=900, mode='asos')
        image = image[y_add:y_add+patch_size, x_add:x_add+patch_size]
    '''

    ratio = image_size // 384
    
    if label_type.lower() == 'asos':
        x_base = (900 - 384) / 2 + 75
        y_base = (900 - 384) / 2 + 115
    elif label_type.lower() == 'aafos':
        x_base = (900 - 256) / 2 + 100
        y_base = (900 - 256) / 2 + 20
        image_size = image_size // 2
    else:
        raise ValueError("mode should be 'AAFOS' or 'ASOS'")
        
    x_base *= ratio
    y_base *= ratio

    return int(x_base), int(y_base), image_size

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
        
    


