try:
    from ._deeplab import DeepLabHeadV3Plus
    from .modeling import _load_model
except:
    from _deeplab import DeepLabHeadV3Plus
    from modeling import _load_model


import torch
import torch.nn as nn
import math

def get_model(n_channel, patch_size, class_num=2):
    model = _load_model('deeplabv3plus', 'resnet18', 1, output_stride=None, pretrained_backbone=True)
    model.backbone.conv1 = nn.Conv2d(n_channel, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.classifier = DeepLabHeadV3Plus(in_channels=512, low_level_channels=64, num_classes=1, aspp_dilate=[6, 12, 18])
    model.classifier.classifier = nn.Conv2d(112, class_num, 3, 1, 1)
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
    
