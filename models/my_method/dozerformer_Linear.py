import torch
from torch import nn
from einops import rearrange

# Implementation from other source code.
from models.my_method.dozerformer_EncDec import dozerformer_Encoder, dozerformer_Decoder

from models.REVIN import RevIN
from models.my_method.build_model_util import series_decomp_multi, series_decomp_multi_learnable
from models.my_method.trend import TrendMSResidual


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

        self.revin_layer = RevIN(self.in_channel, affine=True, subtract_last=False)
        self.revin_layer_dec = RevIN(self.in_channel, affine=True, subtract_last=False)

        # Decomposition
        self.decomp_multi = series_decomp_multi(configs.moving_avg)
        # self.decomp_multi = series_decomp_multi_learnable(
        #     kernel_size=configs.moving_avg,
        #     channels=self.in_channel,
        #     per_channel=True  # recommended for ETTh1
        # )

        # Seasonal encoder and decoder
        self.encoder_seasonal = dozerformer_Encoder(configs, mode='Seasonal')
        self.decoder_seasonal = dozerformer_Decoder(configs, mode='Seasonal')

        self.output_layer_2 = nn.Conv2d(in_channels=self.embed_dim,
                                      out_channels=1,
                                      kernel_size=(1, 1))
        self.output_layer_1 = nn.Linear(configs.seq_len, configs.pred_len)

        self.fuse_logit = nn.Parameter(torch.zeros(1, 1, self.in_channel))  # init 0 => 0.5

        self.trend_model = nn.Linear(configs.seq_len, configs.pred_len)
        self.hour_embed_out = nn.Embedding(24, self.in_channel)
        self.dow_embed_out = nn.Embedding(7, self.in_channel)
        self.out_time_scale = nn.Parameter(torch.tensor(0.1))

        self.refine_ln = nn.LayerNorm(self.in_channel)

        self.refine_dw = nn.Conv1d(self.in_channel, self.in_channel, kernel_size=5,
                                   padding=2, groups=self.in_channel, bias=True)
        self.refine_pw = nn.Conv1d(self.in_channel, self.in_channel, kernel_size=1, bias=True)

        # critical: start as no-op
        nn.init.zeros_(self.refine_pw.weight)
        nn.init.zeros_(self.refine_pw.bias)

        self.refine_scale = nn.Parameter(torch.tensor(1.0))  # scale is safe because pw is zero-init
        self.resid_linear = nn.Linear(self.seq_len, self.pred_len, bias=True)
        self.resid_scale = nn.Parameter(torch.tensor(0.1))  # start small

        # naive branch weight (starts near 0 => mostly no effect)
        self.naive_logit = nn.Parameter(torch.tensor(-3.0))  # sigmoid(-3) ≈ 0.047

        # nn.init.normal_(self.hour_embed.weight, std=0.02)
        # nn.init.normal_(self.dow_embed.weight, std=0.02)

    def forward(self, x_enc, x_mark_enc, seq_y_mark, x_dec, x_label,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None
                ) -> torch.tensor:
        # x_enc: (B, L, D)
        # hour = x_mark_enc[..., 3].long().to("cuda:0")
        # dow = x_mark_enc[..., 2].long().to("cuda:0")
        #
        # hour_emb = self.hour_embed(hour)  # (B, L, D)
        # dow_emb = self.dow_embed(dow)  # (B, L, D)
        # mark_pred: (B, pred_len, 4) = [month, day, weekday, hour]


        x_norm = self.revin_layer(x_enc, 'norm')

        x_enc, trend_enc = self.decomp_multi(x_norm)
        # x_enc = x_enc + hour_emb + dow_emb
        # Encoder
        encoder_output = self.encoder_seasonal(x_enc, x_label)

        encoder_output = self.encoder_seasonal.encoder_segment.concat(encoder_output)
        encoder_output = rearrange(encoder_output, 'b emb seq_len ts_d -> b emb ts_d seq_len')

        seasonal_predict = self.output_layer_1(encoder_output)
        seasonal_predict = self.output_layer_2(seasonal_predict)
        seasonal_predict = rearrange(seasonal_predict, 'b 1 ts_d seq_len -> b seq_len ts_d')


        # Trend
        trend_enc = rearrange(trend_enc, 'b seq_len ts_d -> b ts_d seq_len')
        trend_predict = self.trend_model(trend_enc)
        trend_predict = rearrange(trend_predict, 'b ts_d seq_len -> b seq_len ts_d')




        w = torch.sigmoid(self.fuse_logit)  # (1,1,D)

        #
        final_predict = w * seasonal_predict + (1 - w) * trend_predict
        x = self.refine_ln(final_predict)  # (B,pred,D)
        y = x.transpose(1, 2)  # (B,D,pred)

        corr = self.refine_pw(self.refine_dw(y)).transpose(1, 2)  # (B,pred,D)

        final_predict = final_predict + torch.tanh(self.refine_scale) * corr

        # )
        # ---- Branch 3: naive (daily + weekly) ----
        # daily naive: repeat last 24 hours

        # Concate Trend and Seasonal
        # final_predict = seasonal_predict + trend_predict
        # Inverse Revin
        final_predict = self.revin_layer(final_predict, 'denorm')
        return final_predict

