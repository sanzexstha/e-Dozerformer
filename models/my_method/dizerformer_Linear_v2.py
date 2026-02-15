import torch
from torch import nn
from einops import rearrange

# Implementation from other source code.
from models.my_method.dozerformer_EncDec import dozerformer_Encoder, dozerformer_Decoder

from models.REVIN import RevIN
from models.my_method.build_model_util import series_decomp_multi, series_decomp_multi_learnable
from models.my_method.trend import AdaptiveSTFusion, TinyStableSTFusion, AdaptiveSTFusionV2, SKFusionST


class MoEFusion(nn.Module):
    def __init__(self, C, K=4, drop=0.05):
        super().__init__()
        self.K = K

        # gate produces mixture weights over experts: [B,Q,K]
        self.gate = nn.Sequential(
            nn.Linear(2*C, 2*C),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(2*C, K),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)  # start uniform after softmax

        # K experts each outputs a correction to the sum: [B,Q,C]
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2*C, 2*C),
                nn.GELU(),
                nn.Dropout(drop),
                nn.Linear(2*C, C),
            ) for _ in range(K)
        ])
        for e in self.experts:
            nn.init.zeros_(e[-1].weight)
            nn.init.zeros_(e[-1].bias)

        self.alpha = nn.Parameter(torch.tensor(0.1))  # strength of expert correction

    def forward(self, s, t):
        # s,t: [B,Q,C]
        z = torch.cat([s, t], dim=-1)      # [B,Q,2C]
        sum_pred = s + t

        g = self.gate(z)                   # [B,Q,K]
        w = torch.softmax(g, dim=-1)       # [B,Q,K]

        # expert corrections
        corr = 0.0
        for k, e in enumerate(self.experts):
            corr = corr + w[..., k:k+1] * e(z)   # [B,Q,1] * [B,Q,C]

        return sum_pred + self.alpha * corr
import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F

class SKScaleCorrectFusion(nn.Module):
    """
    Selective-Kernel-style adaptive fusion + scale correction.
    Inputs: trend, seasonal in [B,Q,C]
    Output: fused in [B,Q,C]
    """
    def __init__(self, channels, r=8, U=4, eps=1e-6,
                 pool_mode="mean_q",      # how to pool context
                 weight_mode="per_channel",# "per_channel" or "global"
                 rms_mode="c",            # "c" or "global"
                 detach_alpha=True,
                 clamp_alpha=(0.5, 2.0),
                 drop=0.1):
        super().__init__()
        self.C = channels
        self.eps = eps
        self.pool_mode = pool_mode
        self.weight_mode = weight_mode
        self.rms_mode = rms_mode
        self.detach_alpha = detach_alpha
        self.clamp_alpha = clamp_alpha

        # u = max(D/r, U)
        u = max(channels // r, U)

        # Z = FC(S)
        self.fc_reduce = nn.Sequential(
            nn.Linear(channels, u),
            nn.GELU(),
            nn.Dropout(drop),
        )

        # produce a and b
        if weight_mode == "per_channel":
            self.fc_a = nn.Linear(u, channels)
            self.fc_b = nn.Linear(u, channels)
        elif weight_mode == "global":
            self.fc_a = nn.Linear(u, 1)
            self.fc_b = nn.Linear(u, 1)
        else:
            raise ValueError("weight_mode must be 'per_channel' or 'global'")

    def _pool(self, x_sum):
        # x_sum: [B,Q,C] -> S: [B,C]
        if self.pool_mode == "mean_q":
            return x_sum.mean(dim=1)
        elif self.pool_mode == "mean_q_abs":
            return x_sum.abs().mean(dim=1)
        else:
            raise ValueError("pool_mode must be 'mean_q' or 'mean_q_abs'")

    def _rms(self, x):
        # x: [B,Q,C]
        if self.rms_mode == "c":
            # per-channel RMS over batch+time -> [1,1,C]
            return torch.sqrt(x.pow(2).mean(dim=(0, 1), keepdim=True) + self.eps)
        elif self.rms_mode == "global":
            # scalar RMS -> [1,1,1]
            return torch.sqrt(x.pow(2).mean(dim=(0, 1, 2), keepdim=True) + self.eps)
        else:
            raise ValueError("rms_mode must be 'c' or 'global'")

    def forward(self, trend, seasonal):
        # trend, seasonal: [B,Q,C]
        x_sum = trend + seasonal                      # [B,Q,C]
        S = self._pool(x_sum)                         # [B,C]
        Z = self.fc_reduce(S)                         # [B,u]

        a_logits = self.fc_a(Z)                       # [B,C] or [B,1]
        b_logits = self.fc_b(Z)                       # [B,C] or [B,1]

        # Softmax over the "branch" dimension (2 branches)
        w = torch.stack([a_logits, b_logits], dim=1)  # [B,2,C] or [B,2,1]
        w = F.softmax(w, dim=1)
        a, b = w[:, 0], w[:, 1]                       # [B,C] or [B,1]

        # broadcast to [B,Q,C]
        a = a.unsqueeze(1)                            # [B,1,C] or [B,1,1]
        b = b.unsqueeze(1)

        x_mix = a * trend + b * seasonal              # [B,Q,C]

        # scale correction to match x_sum magnitude
        rms_sum = self._rms(x_sum)
        rms_mix = self._rms(x_mix)
        alpha = (rms_sum / (rms_mix + self.eps))

        if self.detach_alpha:
            alpha = alpha.detach()
        if self.clamp_alpha is not None:
            alpha = alpha.clamp(self.clamp_alpha[0], self.clamp_alpha[1])

        out = alpha * x_mix
        return out, (a, b, alpha)



class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.mode = configs.mode
        assert self.mode in ["pretrain", 'finetune', "forecasting"], "Error mode."
        self.patch_size = configs.patch_size
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.in_channel = configs.data_dim
        self.embed_dim = configs.embed_dim
        # self.mask_ratio = configs.mask_ratio
        self.encoder_depth = configs.encoder_depth
        self.decoder_depth = configs.decoder_depth
        self.decoder_embed_dim = configs.decoder_embed_dim
        self.dropout = configs.dropout
        self.epoch = 0
        # self.d_model = 256
        configs.activation = 'gelu'
        # self.prediction = TwoRegimeFusion(self.in_channel)
        # self.prediction = MoEFusion(self.in_channel)

        # self.st_fusion = AdaptiveSTFusion(C=self.in_channel, r=8, U=4, drop=0.1, pool="mean")

        self.revin_layer = RevIN(self.in_channel, affine=True, subtract_last=False)
        self.revin_layer_dec = RevIN(self.in_channel, affine=True, subtract_last=False)

        # Decomposition
        self.decomp_multi = series_decomp_multi(configs.moving_avg)
        # Seasonal encoder and decoder
        self.encoder_seasonal = dozerformer_Encoder(configs, mode='Seasonal')
        self.decoder_seasonal = dozerformer_Decoder(configs, mode='Seasonal')

        self.output_layer_2 = nn.Conv2d(in_channels=self.embed_dim,
                                      out_channels=1,
                                      kernel_size=(1, 1))
        self.output_layer_1 = nn.Linear(configs.seq_len, configs.pred_len)
        self.trend_model = nn.Linear(configs.seq_len, configs.pred_len)
        # self.sk_fusion = SKScaleCorrectFusion(self.in_channel)
        #
        # self.attention_mlp = nn.Sequential(
        #     nn.Linear(self.in_channel * 2, self.in_channel),
        #     nn.GELU(),
        #     nn.Dropout(0.1),
        #     nn.Linear(self.in_channel, self.in_channel),
        #     nn.Sigmoid()
        # )
        # self.st_fusion = AdaptiveSTFusionV2(C=self.in_channel, r=8, U=4, drop=0.1, pool="mean")
        # self.st_fusion = SKFusionST(self.in_channel)

        # self.beta_mlp = nn.Sequential(
        #     nn.Linear(4 * self.in_channel, self.in_channel),
        #     nn.GELU(),
        #     nn.Dropout(0.1),
        #     nn.Linear(self.in_channel, self.in_channel),
        # )

        # self._init_eia_weights()
        # self.corr = nn.Sequential(
        #     nn.Linear(2 * self.in_channel, self.in_channel),
        #     nn.GELU(),
        #     nn.Linear(self.in_channel, self.in_channel),
        # )
        # nn.init.zeros_(self.corr[-1].weight)
        # nn.init.zeros_(self.corr[-1].bias)

        # self.fuse_logit = nn.Sequential(
        #     torch.zeros(1, 1, self.in_channel),
        #     # nn.Linear(self.in_channel * 2, self.in_channel),
        #     nn.GELU(),
        #     nn.Dropout(0.1),
        #     # nn.Linear(self.in_channel, self.in_channel),
        #     nn.Sigmoid()
        # )
        # self.fuse_logit = nn.Parameter(torch.zeros(1, 1, self.in_channel))
        #
        # self.fuse_mlp = nn.Sequential(
        #     nn.GELU(),
        #     nn.Dropout(p=0.1),
        #     nn.Sigmoid()
        # )
        # self._init_fuse_weights()

        # self.fuse_logit = nn.Parameter(torch.zeros(1, 1, self.in_channel))  # global prior
        # self.delta_mlp = nn.Sequential(
        #     nn.Linear(2 * self.in_channel, self.in_channel),
        #     nn.GELU(),
        #     nn.Dropout(0.1),
        #     nn.Linear(self.in_channel, self.in_channel)
        # )


        # self.fuse_logit = nn.Parameter(torch.zeros(1, 1, self.in_channel))  # init 0 => 0.5

    def _init_fuse_weights(self):
        for layer in self.fuse_logit:
            if isinstance(layer, nn.Linear):
                nn.init.zeros_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def _init_eia_weights(self):
        for layer in self.attention_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.zeros_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)


    def forward(self, x_enc, x_mark_enc, seq_y_mark, x_dec, x_label, phase,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None
                ) -> torch.tensor:

        x_norm = self.revin_layer(x_enc, 'norm')

        x_enc, trend_enc = self.decomp_multi(x_norm)

        # Encoder
        encoder_output = self.encoder_seasonal(x_enc, x_label, x_mark_enc, phase)

        encoder_output = self.encoder_seasonal.encoder_segment.concat(encoder_output)
        encoder_output = rearrange(encoder_output, 'b emb seq_len ts_d -> b emb ts_d seq_len')

        seasonal_predict = self.output_layer_1(encoder_output)
        seasonal_predict = self.output_layer_2(seasonal_predict)
        seasonal_predict = rearrange(seasonal_predict, 'b 1 ts_d seq_len -> b seq_len ts_d')

        # Trend
        trend_enc = rearrange(trend_enc, 'b seq_len ts_d -> b ts_d seq_len')
        trend_predict = self.trend_model(trend_enc)
        trend_predict = rearrange(trend_predict, 'b ts_d seq_len -> b seq_len ts_d')

        # # Concate Trend and Seasonal
        # final_predict = seasonal_predict + trend_predict
        # fusion_weights = self.attention_mlp(torch.cat([seasonal_predict, trend_predict], dim=-1))
        # final_predict = 2 * (fusion_weights * seasonal_predict + (1 - fusion_weights) * trend_predict)
        # seasonal_predict + trend_predict
        # fusion_weights = self.attention_mlp()
        # final_predict = 2 * (fusion_weights * seasonal_predict + (1 - fusion_weights) * trend_predict)
        # new
        # final_predict, (a, b, alpha) = self.sk_fusion(trend_predict, seasonal_predict)







        # delta = self.delta_mlp(torch.cat([y1, y2], dim=-1))
        # β = torch.sigmoid(self.fuse_logit + delta)
        # y3 = 2 * (β * y1 + (1 - β) * y2)
        # y1 = seasonal_predict
        # y2 = trend_predict
        # delta = self.delta_mlp(torch.cat([y1, y2], dim=-1))
        # w = torch.sigmoid(self.fuse_logit + delta)
        # final_predict = 2 * (w * y1 + (1 - w) * y2)
        # final_predict = None
        final_predict = seasonal_predict + trend_predict
        # final_predict = self.st_fusion(seasonal_predict, trend_predict)

        # final_predict = self.prediction(seasonal_predict, trend_predict)

        # Inverse Revin
        final_predict = self.revin_layer(final_predict, 'denorm')
        return final_predict

