try:
    from ._deeplab import DeepLabHeadV3Plus
    from .modeling import _load_model
    from .convlstm import ConvLSTM
    from .PredFormer_Binary_ST import PredFormer_Model
except:
    from _deeplab import DeepLabHeadV3Plus
    from modeling import _load_model
    from convlstm import ConvLSTM
    from PredFormer_Binary_ST import PredFormer_Model

import torch
import torch.nn as nn
import math

# [추가] ConvLSTM을 Segmentation 모델처럼 동작하게 하는 래퍼 클래스 정의
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
        self.dynamic_channels_per_frame = in_channels - misc_channels

        # ConvLSTM 초기화
        self.convlstm = ConvLSTM(input_dim=in_channels, # 여기는 (Dynamic + Misc) 합산 채널이 들어갑니다.
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



def get_model(n_channel, patch_size, class_num=2, model_type='deeplab'):
    """
    Args:
        n_channel: 전체 입력 채널 수
        patch_size: 패치 크기
        class_num: 출력 클래스 수
        model_type: 'deeplab', 'convlstm', 'predformer'
    """

    num_frames = 3
    misc_channels = 3
    dynamic_total = n_channel - misc_channels
    channels_per_frame = (dynamic_total // num_frames) + misc_channels

    if model_type == 'convlstm':
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
        model.backbone.conv1 = nn.Conv2d(n_channel, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.classifier = DeepLabHeadV3Plus(in_channels=512, low_level_channels=64, num_classes=1, aspp_dilate=[6, 12, 18])
        model.classifier.classifier = nn.Conv2d(112, class_num, 3, 1, 1)
        model.cuda()

    else:
        raise ValueError(f"Unknown model_type: {model_type}. Choose 'convlstm' or 'predformer'.")

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
    
