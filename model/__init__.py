try:
    from ._deeplab import DeepLabHeadV3Plus
    from .modeling import _load_model
    from .convlstm import ConvLSTM
    from .PredFormer_Binary_ST import PredFormer_Model
    from .backbone.resnet import resnet18
except:
    from _deeplab import DeepLabHeadV3Plus
    from modeling import _load_model
    from convlstm import ConvLSTM
    from PredFormer_Binary_ST import PredFormer_Model
    from backbone.resnet import resnet18

import torch
import torch.nn as nn
import math

class ConvLSTMSegmentation(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_layers, num_classes, 
                 num_frames, misc_channels=0, kernel_size=(3, 3)):
        """
        Args:
            in_channels: ConvLSTM Cell에 들어갈 'Time Step 당' 채널 수 (Dynamic + Static)
            num_frames: 시계열 데이터의 길이 (Time Step 수)
            misc_channels: 입력 데이터 뒷부분에 붙어있는 정적(Static) 채널의 수
        """
        super(ConvLSTMSegmentation, self).__init__()
        
        self.num_frames = num_frames
        self.misc_channels = misc_channels
        self.dynamic_channels_per_frame = (in_channels - misc_channels) // num_frames

        # ConvLSTM 초기화: 각 타임스텝당 채널 수 = dynamic_channels_per_frame + misc_channels
        self.convlstm = ConvLSTM(input_dim=self.dynamic_channels_per_frame + misc_channels,
                                 hidden_dim=hidden_dim,
                                 kernel_size=kernel_size,
                                 num_layers=num_layers,
                                 batch_first=True,
                                 bias=True,
                                 return_all_layers=False)
        
        # Classifier
        self.classifier = nn.Conv2d(hidden_dim, num_classes, kernel_size=1)

    def forward(self, x):
        # x: (Batch, Total_Channels, H, W)
        # Total_Channels = (Time * Dynamic_C) + Misc_C
        
        b, _, h, w = x.size()
        
        # 1. 데이터 분리 (Dynamic vs Static)
        if self.misc_channels > 0:
            # 뒷부분 misc_channels 만큼 잘라냄
            dynamic_x = x[:, :-self.misc_channels, :, :]   # (B, T*Dyn_C, H, W)
            static_x = x[:, -self.misc_channels:, :, :]    # (B, Misc_C, H, W)
        else:
            dynamic_x = x
            static_x = None

        # 2. Dynamic Data Reshape: (B, T*C, H, W) -> (B, T, C, H, W)
        # self.dynamic_channels_per_frame 변수를 사용하여 정확하게 reshape
        dynamic_x = dynamic_x.view(b, self.num_frames, self.dynamic_channels_per_frame, h, w)

        # 3. Static Data 확장 및 결합 (각 Time Step마다 붙여줌)
        if static_x is not None:
            # (B, Misc_C, H, W) -> (B, 1, Misc_C, H, W) -> (B, T, Misc_C, H, W)
            static_x = static_x.unsqueeze(1).expand(-1, self.num_frames, -1, -1, -1)
            
            # Channel 차원(dim=2)에서 결합 -> (B, T, Dyn_C + Misc_C, H, W)
            x_input = torch.cat([dynamic_x, static_x], dim=2)
        else:
            x_input = dynamic_x

        # 4. ConvLSTM 실행
        # x_input shape: (Batch, Time, Channel, Height, Width)
        _, last_states = self.convlstm(x_input)
        
        # 마지막 Hidden State 추출
        h = last_states[-1][0] # (Batch, hidden_dim, H, W)
        
        out = self.classifier(h)
        return out


# [추가] PredFormer를 Segmentation 모델처럼 동작하게 하는 래퍼 클래스 정의
class PredFormerSegmentation(nn.Module):
    def __init__(self, in_channels, num_classes, num_frames, misc_channels=0,
                 patch_size=8, dim=256, heads=8, dim_head=32,
                 dropout=0.0, attn_dropout=0.0, drop_path=0.0,
                 scale_dim=4, depth=1, Ndepth=12, image_size=512):
        """
        Args:
            in_channels: 전체 입력 채널 수 (Dynamic * Time + Static)
            num_classes: 출력 클래스 수
            num_frames: 시계열 데이터의 길이 (Time Step 수)
            misc_channels: 입력 데이터 뒷부분에 붙어있는 정적(Static) 채널의 수
            patch_size: 패치 크기
            image_size: 이미지 크기 (H, W)
        """
        super(PredFormerSegmentation, self).__init__()

        self.num_frames = num_frames
        self.misc_channels = misc_channels
        self.dynamic_channels_per_frame = (in_channels - misc_channels) // num_frames
        self.image_size = image_size

        # PredFormer 설정
        model_config = {
            'height': image_size,
            'width': image_size,
            'num_channels': self.dynamic_channels_per_frame + misc_channels,  # 각 타임스텝당 채널 수
            'pre_seq': num_frames,
            'patch_size': patch_size,
            'dim': dim,
            'heads': heads,
            'dim_head': dim_head,
            'dropout': dropout,
            'attn_dropout': attn_dropout,
            'drop_path': drop_path,
            'scale_dim': scale_dim,
            'depth': depth,
            'Ndepth': Ndepth,
            'out_channels': num_classes
        }

        self.predformer = PredFormer_Model(model_config)

    def forward(self, x):
        # x: (Batch, Total_Channels, H, W)
        # Total_Channels = (Time * Dynamic_C) + Misc_C

        b, _, h, w = x.size()

        # 1. 데이터 분리 (Dynamic vs Static)
        if self.misc_channels > 0:
            # 뒷부분 misc_channels 만큼 잘라냄
            dynamic_x = x[:, :-self.misc_channels, :, :]   # (B, T*Dyn_C, H, W)
            static_x = x[:, -self.misc_channels:, :, :]    # (B, Misc_C, H, W)
        else:
            dynamic_x = x
            static_x = None

        # 2. Dynamic Data Reshape: (B, T*C, H, W) -> (B, T, C, H, W)
        dynamic_x = dynamic_x.view(b, self.num_frames, self.dynamic_channels_per_frame, h, w)

        # 3. Static Data 확장 및 결합 (각 Time Step마다 붙여줌)
        if static_x is not None:
            # (B, Misc_C, H, W) -> (B, 1, Misc_C, H, W) -> (B, T, Misc_C, H, W)
            static_x = static_x.unsqueeze(1).expand(-1, self.num_frames, -1, -1, -1)

            # Channel 차원(dim=2)에서 결합 -> (B, T, Dyn_C + Misc_C, H, W)
            x_input = torch.cat([dynamic_x, static_x], dim=2)
        else:
            x_input = dynamic_x

        # 4. PredFormer 실행
        # x_input shape: (Batch, Time, Channel, Height, Width)
        out = self.predformer(x_input)  # (B, T, num_classes, H, W)

        # 마지막 타임스텝의 출력을 반환 (또는 평균을 사용할 수도 있음)
        out = out[:, -1, :, :, :]  # (B, num_classes, H, W)

        return out


class ResNetConvLSTMSegmentation(nn.Module):
    """
    ResNet18 backbone으로 feature extraction 후 ConvLSTM에 입력하는 모델
    - 각 timestep마다 ResNet18으로 feature 추출
    - Feature sequence를 ConvLSTM으로 temporal modeling
    - Decoder로 원래 해상도 복원

    Sliding Window 모드:
    - window_size > 1이면 여러 timestep을 묶어서 하나의 프레임으로 처리
    - stride로 윈도우 이동 간격 조절 (overlap 가능)
    """
    def __init__(self, in_channels, hidden_dim, num_layers, num_classes,
                 num_timesteps, misc_channels=0, kernel_size=(3, 3),
                 pretrained=True, feature_layer='layer3',
                 window_size=1, stride=1):
        """
        Args:
            in_channels: 전체 입력 채널 수 (Dynamic * Time + Static)
            hidden_dim: ConvLSTM hidden dimension
            num_layers: ConvLSTM layer 수
            num_classes: 출력 클래스 수
            num_timesteps: 원본 시계열 데이터의 timestep 수 (예: time_range 길이)
            misc_channels: Static 채널 수
            kernel_size: ConvLSTM kernel size
            pretrained: ImageNet pretrained weights 사용 여부
            feature_layer: feature 추출할 layer ('layer2', 'layer3', 'layer4')
            window_size: 각 프레임이 포함하는 timestep 수 (기본 1 = 개별 처리)
            stride: 윈도우 이동 간격 (기본 1, stride < window_size면 overlap)
        """
        super(ResNetConvLSTMSegmentation, self).__init__()

        self.num_timesteps = num_timesteps
        self.misc_channels = misc_channels
        self.window_size = window_size
        self.stride = stride
        self.feature_layer = feature_layer

        # 원본 timestep당 채널 수
        self.channels_per_timestep = (in_channels - misc_channels) // num_timesteps

        # Sliding window로 생성되는 프레임 수
        self.num_frames = (num_timesteps - window_size) // stride + 1

        # 각 프레임(윈도우)의 채널 수
        self.channels_per_frame = self.channels_per_timestep * window_size

        # Feature dimension 및 scale factor 설정
        feature_dims = {'layer2': 128, 'layer3': 256, 'layer4': 512}
        scale_factors = {'layer2': 8, 'layer3': 16, 'layer4': 32}
        self.feature_dim = feature_dims[feature_layer]
        self.scale_factor = scale_factors[feature_layer]

        # ResNet18 backbone 로드
        backbone = resnet18(pretrained=pretrained)

        # 입력 채널 수 변경 (윈도우 내 dynamic 채널 + static 채널)
        backbone_input_channels = self.channels_per_frame + misc_channels
        self.backbone_conv1 = nn.Conv2d(backbone_input_channels, 64,
                                         kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone_bn1 = backbone.bn1
        self.backbone_relu = backbone.relu
        self.backbone_maxpool = backbone.maxpool
        self.backbone_layer1 = backbone.layer1
        self.backbone_layer2 = backbone.layer2

        if feature_layer in ['layer3', 'layer4']:
            self.backbone_layer3 = backbone.layer3
        if feature_layer == 'layer4':
            self.backbone_layer4 = backbone.layer4

        # ConvLSTM: feature dimension을 입력으로 받음
        self.convlstm = ConvLSTM(
            input_dim=self.feature_dim,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            num_layers=num_layers,
            batch_first=True,
            bias=True,
            return_all_layers=False
        )

        # Decoder: Upsampling + Classification
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, num_classes, kernel_size=1)
        )

        # Upsampling은 forward에서 동적으로 처리 (입력 크기에 맞춤)

    def extract_features(self, x):
        """단일 프레임에서 feature 추출"""
        x = self.backbone_conv1(x)
        x = self.backbone_bn1(x)
        x = self.backbone_relu(x)
        x = self.backbone_maxpool(x)

        x = self.backbone_layer1(x)
        x = self.backbone_layer2(x)

        if self.feature_layer in ['layer3', 'layer4']:
            x = self.backbone_layer3(x)
        if self.feature_layer == 'layer4':
            x = self.backbone_layer4(x)

        return x

    def forward(self, x):
        # x: (B, Total_Channels, H, W)
        b, _, h, w = x.size()

        # 1. 데이터 분리 (Dynamic vs Static)
        if self.misc_channels > 0:
            dynamic_x = x[:, :-self.misc_channels, :, :]
            static_x = x[:, -self.misc_channels:, :, :]
        else:
            dynamic_x = x
            static_x = None

        # 2. Dynamic Data Reshape: (B, T*C, H, W) -> (B, T, C, H, W)
        #    T = num_timesteps (원본 timestep 수)
        #    C = channels_per_timestep (timestep당 채널 수)
        dynamic_x = dynamic_x.view(b, self.num_timesteps, self.channels_per_timestep, h, w)

        # 3. Sliding Window로 각 프레임 생성 및 feature extraction
        features = []
        for f in range(self.num_frames):
            # 윈도우 시작 timestep
            start_t = f * self.stride
            # 윈도우 내 timestep들의 채널을 concat
            window_timesteps = dynamic_x[:, start_t:start_t + self.window_size, :, :, :]  # (B, window_size, C, H, W)
            # (B, window_size, C, H, W) -> (B, window_size * C, H, W)
            frame = window_timesteps.reshape(b, self.channels_per_frame, h, w)

            # Static 채널 결합
            if static_x is not None:
                frame = torch.cat([frame, static_x], dim=1)

            # ResNet backbone으로 feature 추출
            feat = self.extract_features(frame)  # (B, feat_dim, H', W')
            features.append(feat)

        # 4. Feature sequence 생성: (B, T, feat_dim, H', W')
        feature_seq = torch.stack(features, dim=1)

        # 5. ConvLSTM 실행
        _, last_states = self.convlstm(feature_seq)
        h_out = last_states[-1][0]  # (B, hidden_dim, H', W')

        # 6. Decoder
        out = self.decoder(h_out)

        # 7. 원래 해상도로 Upsample
        out = nn.functional.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)

        return out


def get_model(n_channel, patch_size, class_num=2, model_type='deeplab', misc_channels=3,
              channels_per_timestep=16, window_size=2, stride=1):
    """
    Args:
        n_channel: 전체 입력 채널 수
        patch_size: 패치 크기
        class_num: 출력 클래스 수
        model_type: 'deeplab', 'convlstm', 'predformer', 'resnet_convlstm'
        misc_channels: Static 채널 수 (elevation, vegetation, watermap 등)
        channels_per_timestep: 한 timestep당 채널 수 (16ch 데이터 기준 16)
        window_size: Sliding window 크기 (기본 2 = 개별 timestep 처리)
        stride: Sliding window 이동 간격 (기본 1)

    Examples:
        time_range = [-18, -15, -12] (3 timesteps), window_size=1, stride=1
          → 3 frames, 각 19채널 (16 + 3 misc)

        time_range = [-21, -18, -15, -12] (4 timesteps), window_size=2, stride=1
          → 3 frames (overlap), 각 35채널 (32 + 3 misc)
          → Frame 0: t=-21, t=-18
          → Frame 1: t=-18, t=-15 (overlap)
          → Frame 2: t=-15, t=-12 (overlap)

        time_range = [-27, ..., -12] (6 timesteps), window_size=2, stride=2
          → 3 frames (no overlap), 각 35채널
          → Frame 0: t=-27, t=-24
          → Frame 1: t=-21, t=-18
          → Frame 2: t=-15, t=-12
    """

    # 원본 timestep 수 계산
    num_timesteps = (n_channel - misc_channels) // channels_per_timestep

    # ConvLSTM/PredFormer용 num_frames 계산 (sliding window 적용 전)
    num_frames = (num_timesteps - window_size) // stride + 1
    channels_per_frame = channels_per_timestep * window_size + misc_channels

    if model_type == 'resnet_convlstm':
        hidden_dim = 64
        num_layers = 2

        model = ResNetConvLSTMSegmentation(
            in_channels=n_channel,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=class_num,
            num_timesteps=num_timesteps,
            misc_channels=misc_channels,
            pretrained=True,
            feature_layer='layer3',
            window_size=window_size,
            stride=stride
        )

    elif model_type == 'convlstm':
        hidden_dim = 64
        num_layers = 2

        model = ConvLSTMSegmentation(in_channels=n_channel,
                                     hidden_dim=hidden_dim,
                                     num_layers=num_layers,
                                     num_classes=class_num,
                                     num_frames=num_frames,
                                     misc_channels=misc_channels)
    elif model_type == 'predformer':
        model = PredFormerSegmentation(in_channels=n_channel,
                                       num_classes=class_num,
                                       num_frames=num_frames,
                                       misc_channels=misc_channels,
                                       patch_size=8,
                                       dim=256,
                                       heads=8,
                                       dim_head=32,
                                       dropout=0.0,
                                       attn_dropout=0.0,
                                       drop_path=0.0,
                                       scale_dim=4,
                                       depth=1,
                                       Ndepth=12,
                                       image_size=patch_size)
        
    elif model_type == 'deeplab':
        model = _load_model('deeplabv3plus', 'resnet18', 1, output_stride=None, pretrained_backbone=True)
        model.backbone.conv1 = nn.Conv2d(n_channel, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) # 거의 1/2 압축
        model.classifier = DeepLabHeadV3Plus(in_channels=512, low_level_channels=64, num_classes=1, aspp_dilate=[6, 12, 18])
        model.classifier.classifier = nn.Conv2d(112, class_num, 3, 1, 1)
        model.cuda()

    else:
        raise ValueError(f"Unknown model_type: {model_type}. Choose 'deeplab', 'convlstm', 'predformer', or 'resnet_convlstm'.")

    model.cuda()

    def forward_by_patch(self, x:torch.Tensor, patch_size:int, overlap:int=None, mode:str='cosine', eps:float=1e-8, latlon:bool=False):
        b, c, h, w = x.shape

        if h == patch_size:
            return self.forward(x)

        overlap = patch_size // 2 if overlap is None else overlap
        stride = patch_size - overlap
        pred_map = torch.zeros([b, class_num, h, w], device=x.device)
        mask = torch.zeros([b, class_num, h, w], device=x.device)

        patch_center = h // 2
        patch_centers = set([patch_center])
        for i in range(patch_center, patch_size // 2, -stride):
            patch_centers.add(i)
        patch_centers.add(patch_size // 2)
        for i in range(patch_center, h - patch_size // 2, stride):
            patch_centers.add(i)
        patch_centers.add(h - patch_size // 2)
        patch_idx = sorted(patch_centers)

        # 1D 가중치 벡터 생성 (edge blending)
        if mode == 'linear':
            weight_1d = torch.linspace(0, 1, overlap, device=x.device)
        elif mode == 'cosine':
            weight_1d = 0.5 * (1 - torch.cos(torch.linspace(0, math.pi, overlap, device=x.device)))

        # 2D 가중치 마스크 생성
        weight_mask = torch.ones([b, class_num, patch_size, patch_size], device=x.device)
        # 상단/하단 edge
        weight_mask[:, :, :overlap, :] *= weight_1d.unsqueeze(1)
        weight_mask[:, :, -overlap:, :] *= weight_1d.flip(0).unsqueeze(1)
        # 좌/우 edge
        weight_mask[:, :, :, :overlap] *= weight_1d.unsqueeze(0)
        weight_mask[:, :, :, -overlap:] *= weight_1d.flip(0).unsqueeze(0)

        for i in patch_idx:
            for j in patch_idx:
                patch = x[:, :, i-patch_size//2:i+patch_size//2, j-patch_size//2:j+patch_size//2]
                pred = self.forward(patch)
                # 가중치 적용
                pred = pred * weight_mask
                pred_map[:, :, i-patch_size//2:i+patch_size//2, j-patch_size//2:j+patch_size//2] += pred
                mask[:, :, i-patch_size//2:i+patch_size//2, j-patch_size//2:j+patch_size//2] += weight_mask

        # plt.imshow(mask[0, 0].cpu().detach().numpy())
        # plt.colorbar()
        # plt.show()

        # plt.imshow(pred_map[0, 0].cpu().detach().numpy())
        # plt.colorbar()
        # plt.show()

        pred_map /= (mask + eps)
        return pred_map

    model.forward_by_patch = lambda x, patch_size=patch_size, overlap=None, mode='cosine', eps=1e-8, latlon=False: forward_by_patch(model, x, patch_size, overlap, mode, eps, latlon)

    # x = torch.randn(4, 19, 1024, 1024).cuda()
    # pred = model.forward_by_patch(x, 256)
    # plt.imshow(pred[0, 0].cpu().detach().numpy())
    # plt.show()

    return model

if __name__ == "__main__":
    model = get_model(21, 256).cuda()
    print(model)
    x = torch.randn(4, 19, 1024, 1024).cuda()
    pred = model.forward_by_patch(x, patch_size=256, latlon=True, image_size=512)
    print(pred.shape)
    
