import torch
from torch import nn
import torch.nn.functional as F

class TrendMSResidual(nn.Module):
    """
    trend_enc: (B, L=720, D)
    returns trend_pred: (B, pred=96, D)
    """
    def __init__(self, seq_len, pred_len, scales=(1, 6, 24, 168), d_hidden=128, dropout=0.0):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.scales = scales

        # baseline per-channel mapping (strong)
        self.base = nn.Linear(seq_len, pred_len)

        # residual heads per scale
        self.res_heads = nn.ModuleList()
        for s in scales:
            Ls = seq_len // s
            self.res_heads.append(nn.Sequential(
                nn.Linear(Ls, d_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_hidden, pred_len)
            ))

        # per-channel gate (computed from full-res history)
        self.gate = nn.Sequential(
            nn.Linear(seq_len, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # residual scale starts small => won’t hurt baseline
        self.res_scale = nn.Parameter(torch.tensor(0.005))

    def forward(self, trend_enc):
        B, L, D = trend_enc.shape
        assert L == self.seq_len

        x = trend_enc.transpose(1, 2)  # (B, D, L)

        base = self.base(x)  # (B, D, pred)

        # multi-scale residual
        res_sum = 0.0
        for s, head in zip(self.scales, self.res_heads):
            if s == 1:
                x_s = x  # (B, D, L)
            else:
                # avg pool along time: (B*D,1,L) -> (B*D,1,L//s)
                x_s = x.reshape(B * D, 1, L)
                x_s = F.avg_pool1d(x_s, kernel_size=s, stride=s)
                x_s = x_s.reshape(B, D, L // s)

            res_sum = res_sum + head(x_s)  # (B, D, pred)

        gate = self.gate(x)  # (B, D, 1)

        trend = base + torch.tanh(self.res_scale) * gate * res_sum
        return trend.transpose(1, 2)  # (B, pred, D)
