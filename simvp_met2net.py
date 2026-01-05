"""
Met²Net: A Decoupled Two-Stage Spatio-Temporal Forecasting Model
논문에 맞게 수정된 SimVP 기반 모델

핵심 수정 사항:
1. Stage 1: Translator 동결, Encoder/Decoder가 공유 잠재 공간 학습 (Reconstruction)
2. Stage 2: Encoder/Decoder 동결, Translator가 변수 간 상호작용 학습 (Prediction)
3. Momentum EMA로 부드러운 파라미터 전이
4. Self-attention 기반 다변수 융합 (CIMidNet)
"""

import torch
from torch import nn
import torch.nn.functional as F
from timm.layers import trunc_normal_

from openstl.modules import ConvSC
from openstl.models.simvp_ema.ciatt_modules import CIMidNet


def sampling_generator(N, reverse=False):
    """Encoder/Decoder의 downsampling/upsampling 패턴 생성"""
    samplings = [False, True] * (N // 2)
    if reverse:
        return list(reversed(samplings[:N]))
    return samplings[:N]


class Encoder(nn.Module):
    """변수별 독립 Encoder - 각 기상 변수를 개별 모달리티로 처리"""
    
    def __init__(self, C_in, C_hid, N_S, spatio_kernel, act_inplace=True):
        super(Encoder, self).__init__()
        samplings = sampling_generator(N_S)
        self.enc = nn.Sequential(
            ConvSC(C_in, C_hid, spatio_kernel, downsampling=samplings[0], act_inplace=act_inplace),
            *[ConvSC(C_hid, C_hid, spatio_kernel, downsampling=s, act_inplace=act_inplace) 
              for s in samplings[1:]]
        )

    def forward(self, x):
        """
        Args:
            x: (B*T, 1, H, W) - 단일 변수의 시공간 데이터
        Returns:
            latent: (B*T, C_hid, H', W') - 잠재 표현
            enc1: 첫 번째 인코더 출력 (skip connection용)
        """
        enc1 = self.enc[0](x)
        latent = enc1
        for i in range(1, len(self.enc)):
            latent = self.enc[i](latent)
        return latent, enc1


class Decoder(nn.Module):
    """변수별 독립 Decoder - 잠재 공간에서 원본 공간으로 복원"""
    
    def __init__(self, C_hid, C_out, N_S, spatio_kernel, act_inplace=True):
        super(Decoder, self).__init__()
        samplings = sampling_generator(N_S, reverse=True)
        self.dec = nn.Sequential(
            *[ConvSC(C_hid, C_hid, spatio_kernel, upsampling=s, act_inplace=act_inplace) 
              for s in samplings[:-1]],
            ConvSC(C_hid, C_hid, spatio_kernel, upsampling=samplings[-1], act_inplace=act_inplace)
        )
        self.readout = nn.Conv2d(C_hid, C_out, 1)

    def forward(self, hid, enc1=None):
        """
        Args:
            hid: (B*T, C_hid, H', W') - 잠재 표현
            enc1: skip connection (optional)
        Returns:
            Y: (B*T, C_out, H, W) - 복원된 출력
        """
        for i in range(0, len(self.dec) - 1):
            hid = self.dec[i](hid)
        if enc1 is not None:
            Y = self.dec[-1](hid + enc1)
        else:
            Y = self.dec[-1](hid)
        Y = self.readout(Y)
        return Y


class Met2Net_Model(nn.Module):
    """
    Met²Net: Decoupled Two-Stage Spatio-Temporal Forecasting Model
    
    논문의 핵심 아이디어:
    - 각 기상 변수를 독립적인 모달리티로 취급
    - 변수별 독립 Encoder/Decoder로 representation inconsistency 해결
    - 2단계 학습으로 task inconformity 해결
    - Self-attention (CIMidNet)으로 다변수 간 상호작용 학습
    
    Stage 1 (Reconstruction): 
        - Translator 동결
        - Encoder/Decoder가 공유 잠재 공간 학습
        - Loss: MSE(reconstructed, target)
        
    Stage 2 (Prediction):
        - Encoder/Decoder 동결 (EMA로 업데이트)
        - Translator가 시공간 특징 및 변수 간 상호작용 학습
        - Loss: MSE(latent_pred, latent_target) + MSE(pred, target)
    """
    
    def __init__(self, in_shape, hid_S=16, hid_T=256, N_S=4, N_T=4,
                 spatio_kernel_enc=3, spatio_kernel_dec=3, 
                 act_inplace=True, momentum_ema=0.9, **kwargs):
        """
        Args:
            in_shape: [T, C, H, W] - 입력 텐서 shape
            hid_S: Encoder/Decoder의 hidden channel 수
            hid_T: Translator의 hidden dimension
            N_S: Encoder/Decoder의 layer 수
            N_T: Translator의 layer 수
            momentum_ema: EMA momentum (0.9 ~ 0.999)
        """
        super(Met2Net_Model, self).__init__()
        
        self.momentum = momentum_ema
        self.hid_S = hid_S
        
        T, C, H, W = in_shape
        self.T = T
        self.C = C
        
        # Downsampled spatial size
        self.H_latent = int(H / 2 ** (N_S / 2))
        self.W_latent = int(W / 2 ** (N_S / 2))
        
        act_inplace = False  # 안정성을 위해 False
        
        # ============================================
        # Stage 1용 모듈 (학습 대상: Encoder/Decoder)
        # ============================================
        # Query Encoder/Decoder: Stage 1에서 학습됨
        self.enc_q_list = nn.ModuleList([
            Encoder(1, hid_S, N_S, spatio_kernel_enc, act_inplace=act_inplace) 
            for _ in range(C)
        ])
        self.dec_q_list = nn.ModuleList([
            Decoder(hid_S, 1, N_S, spatio_kernel_dec, act_inplace=act_inplace) 
            for _ in range(C)
        ])
        
        # ============================================
        # Stage 2용 모듈 (학습 대상: Translator)
        # ============================================
        # Key Encoder/Decoder: EMA로 업데이트 (Stage 2에서 동결)
        self.enc_k_list = nn.ModuleList([
            Encoder(1, hid_S, N_S, spatio_kernel_enc, act_inplace=act_inplace) 
            for _ in range(C)
        ])
        self.dec_k_list = nn.ModuleList([
            Decoder(hid_S, 1, N_S, spatio_kernel_dec, act_inplace=act_inplace) 
            for _ in range(C)
        ])
        
        # ============================================
        # Translator: Self-attention 기반 다변수 융합
        # ============================================
        # Query Translator: Stage 2에서 학습됨
        self.translator_q = CIMidNet(
            in_channels=T * hid_S, 
            d_model=hid_T, 
            n_layers=N_T, 
            heads=8
        )
        # Key Translator: EMA로 업데이트 (Stage 1에서 사용)
        self.translator_k = CIMidNet(
            in_channels=T * hid_S, 
            d_model=hid_T, 
            n_layers=N_T, 
            heads=8
        )
        
        # 초기화
        self._init_weights()
        
    def _init_weights(self):
        """Query -> Key 초기 동기화"""
        # Encoder 동기화
        for enc_q, enc_k in zip(self.enc_q_list, self.enc_k_list):
            for param_q, param_k in zip(enc_q.parameters(), enc_k.parameters()):
                param_k.requires_grad = False
                param_k.data.copy_(param_q.data)
        
        # Decoder 동기화
        for dec_q, dec_k in zip(self.dec_q_list, self.dec_k_list):
            for param_q, param_k in zip(dec_q.parameters(), dec_k.parameters()):
                param_k.requires_grad = False
                param_k.data.copy_(param_q.data)
        
        # Translator 동기화
        for param_q, param_k in zip(self.translator_q.parameters(), self.translator_k.parameters()):
            param_k.requires_grad = False
            param_k.data.copy_(param_q.data)

    @torch.no_grad()
    def _momentum_update_encoder(self):
        """EMA update: enc_k = m * enc_k + (1-m) * enc_q"""
        m = self.momentum
        for enc_q, enc_k in zip(self.enc_q_list, self.enc_k_list):
            for param_q, param_k in zip(enc_q.parameters(), enc_k.parameters()):
                param_k.data.mul_(m).add_(param_q.data, alpha=1.0 - m)

    @torch.no_grad()
    def _momentum_update_decoder(self):
        """EMA update: dec_k = m * dec_k + (1-m) * dec_q"""
        m = self.momentum
        for dec_q, dec_k in zip(self.dec_q_list, self.dec_k_list):
            for param_q, param_k in zip(dec_q.parameters(), dec_k.parameters()):
                param_k.data.mul_(m).add_(param_q.data, alpha=1.0 - m)

    @torch.no_grad()
    def _momentum_update_translator(self):
        """EMA update: translator_k = m * translator_k + (1-m) * translator_q"""
        m = self.momentum
        for param_q, param_k in zip(self.translator_q.parameters(), self.translator_k.parameters()):
            param_k.data.mul_(m).add_(param_q.data, alpha=1.0 - m)

    def _encode(self, x, encoder_list):
        """
        변수별 독립 인코딩
        Args:
            x: (B*T, C, H, W)
            encoder_list: C개의 Encoder 모듈
        Returns:
            z: (B, C, T*hid_S, H', W') - 잠재 표현
        """
        B_T, C, H, W = x.shape
        B = B_T // self.T
        
        h_list = []
        for i in range(C):
            h, _ = encoder_list[i](x[:, i:i+1, :, :])  # (B*T, hid_S, H', W')
            h_list.append(h)
        
        H_, W_ = h_list[0].shape[-2], h_list[0].shape[-1]
        
        # Stack and reshape: (B*T, C, hid_S, H', W') -> (B, C, T*hid_S, H', W')
        z = torch.stack(h_list, dim=1)  # (B*T, C, hid_S, H', W')
        z = z.reshape(B, self.T, C, -1, H_, W_)  # (B, T, C, hid_S, H', W')
        z = z.permute(0, 2, 1, 3, 4, 5)  # (B, C, T, hid_S, H', W')
        z = z.reshape(B, C, -1, H_, W_)  # (B, C, T*hid_S, H', W')
        
        return z, H_, W_

    def _decode(self, z, decoder_list, T_out, H_, W_):
        """
        변수별 독립 디코딩
        Args:
            z: (B, C, T*hid_S, H', W')
            decoder_list: C개의 Decoder 모듈
            T_out: 출력 시간 스텝 수
        Returns:
            y: (B, T_out, C, H, W)
        """
        B, C = z.shape[:2]
        
        # Reshape for decoding
        z = z.reshape(B, C, self.T, -1, H_, W_)[:, :, :T_out]  # (B, C, T_out, hid_S, H', W')
        z = z.reshape(B * T_out, C, -1, H_, W_)  # (B*T_out, C, hid_S, H', W')
        
        rec_list = []
        for i in range(C):
            rec = decoder_list[i](z[:, i, :, :, :])  # (B*T_out, 1, H, W)
            rec_list.append(rec)
        
        y = torch.stack(rec_list, dim=1)  # (B*T_out, C, 1, H, W)
        y = y.squeeze(2)  # (B*T_out, C, H, W)
        
        H, W = y.shape[-2], y.shape[-1]
        y = y.reshape(B, T_out, C, H, W)  # (B, T_out, C, H, W)
        
        return y

    def forward(self, x_raw, y_raw, **kwargs):
        """
        2단계 학습을 한 forward pass에서 수행
        
        Args:
            x_raw: (B, T, C, H, W) - 입력 시퀀스
            y_raw: (B, T_out, C, H, W) - 타겟 시퀀스
            
        Returns:
            pre_y: (B, T_out, C, H, W) - 예측 출력
            loss: 총 손실
            loss_rec: Stage 1 reconstruction loss
            loss_latent: Stage 2 latent space loss
            loss_pre: Stage 2 prediction loss
        """
        B, T, C, H, W = x_raw.shape
        _, T_out, _, _, _ = y_raw.shape
        
        x = x_raw.view(B * T, C, H, W)
        y = y_raw.view(B * T_out, C, H, W)
        
        # ============================================
        # Stage 1: Reconstruction (Encoder/Decoder 학습)
        # - Translator_k는 동결 상태 (학습 안 함)
        # - Encoder_q, Decoder_q가 공유 잠재 공간 학습
        # ============================================
        
        # 1-1. Encoder_q로 입력 인코딩
        z_x, H_, W_ = self._encode(x, self.enc_q_list)  # (B, C, T*hid_S, H', W')
        
        # 1-2. Translator_k로 시공간 변환 (동결됨, no_grad)
        with torch.no_grad():
            z_y_stage1 = self.translator_k(z_x)  # (B, C, T*hid_S, H', W')
        
        # 1-3. Decoder_q로 복원
        rec_y = self._decode(z_y_stage1, self.dec_q_list, T_out, H_, W_)
        
        # Stage 1 Loss: Reconstruction
        loss_rec = F.mse_loss(rec_y, y_raw)
        
        # ============================================
        # Stage 2: Prediction (Translator 학습)
        # - Encoder_k, Decoder_k는 EMA로 업데이트됨
        # - Translator_q가 변수 간 상호작용 학습
        # ============================================
        
        # EMA 업데이트 (Encoder/Decoder를 천천히 동기화)
        self._momentum_update_encoder()
        self._momentum_update_decoder()
        
        # 2-1. Encoder_k로 입력 인코딩 (동결됨)
        with torch.no_grad():
            z_x_k, _, _ = self._encode(x, self.enc_k_list)
        
        # 2-2. Translator_q로 시공간 변환 (학습됨)
        z_y_stage2 = self.translator_q(z_x_k)  # (B, C, T*hid_S, H', W')
        
        # 2-3. GT의 잠재 표현 얻기 (Encoder_k 사용)
        with torch.no_grad():
            z_y_gt, _, _ = self._encode(y, self.enc_k_list)
            # T_out에 맞게 reshape
            z_y_gt = z_y_gt.reshape(B, C, T_out, -1, H_, W_)
            z_y_gt = z_y_gt.reshape(B, C, -1, H_, W_)  # T_out * hid_S
        
        # Stage 2 잠재 공간도 T_out에 맞게 조정
        z_y_stage2_for_loss = z_y_stage2.reshape(B, C, self.T, -1, H_, W_)[:, :, :T_out]
        z_y_stage2_for_loss = z_y_stage2_for_loss.reshape(B, C, -1, H_, W_)
        
        # Stage 2 Loss: Latent space alignment
        loss_latent = F.mse_loss(z_y_stage2_for_loss, z_y_gt)
        
        # Translator EMA 업데이트
        self._momentum_update_translator()
        
        # 2-4. Decoder_k로 최종 예측 생성
        with torch.no_grad():
            # Decoder_k는 동결이지만 출력은 필요
            pass
        pre_y = self._decode(z_y_stage2, self.dec_k_list, T_out, H_, W_)
        
        # Stage 2 Loss: Final prediction
        loss_pre = F.mse_loss(pre_y, y_raw)
        
        # 총 손실 (논문에서는 loss_rec + loss_latent 사용)
        loss = loss_rec + loss_latent + 0.1 * loss_pre
        
        return pre_y, loss, loss_rec, loss_latent, loss_pre

    @torch.no_grad()
    def sample(self, batch_x):
        """
        추론 모드: 입력만으로 예측 생성
        
        Args:
            batch_x: (B, T, C, H, W)
        Returns:
            pred_y: (B, T, C, H, W)
        """
        B, T, C, H, W = batch_x.shape
        x = batch_x.view(B * T, C, H, W)
        
        # Encoder_q로 인코딩
        z_x, H_, W_ = self._encode(x, self.enc_q_list)
        
        # Translator_q로 시공간 변환
        z_y = self.translator_q(z_x)
        
        # Decoder_q로 디코딩
        pred_y = self._decode(z_y, self.dec_q_list, T, H_, W_)
        
        return pred_y


# Backward compatibility alias
SimVP_Model = Met2Net_Model


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Test configuration
    B, T, C, H, W = 4, 4, 19, 128, 128
    T_out = 4
    
    x = torch.randn((B, T, C, H, W)).to(device)
    y = torch.randn((B, T_out, C, H, W)).to(device)
    
    model = Met2Net_Model(
        in_shape=[T, C, H, W],
        hid_S=16,
        hid_T=128,
        N_S=4,
        N_T=4,
        momentum_ema=0.9
    ).to(device)
    
    print("Testing forward pass...")
    pre_y, loss, loss_rec, loss_latent, loss_pre = model(x, y)
    print(f"Output shape: {pre_y.shape}")
    print(f"Losses - Total: {loss.item():.4f}, Rec: {loss_rec.item():.4f}, "
          f"Latent: {loss_latent.item():.4f}, Pre: {loss_pre.item():.4f}")
    
    print("\nTesting sample (inference) mode...")
    pred = model.sample(x)
    print(f"Sample output shape: {pred.shape}")
