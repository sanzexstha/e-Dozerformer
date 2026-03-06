import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveSTFusion(nn.Module):
    """
    Adaptive fusion block (Selective-Kernel style) for seasonal/trend fusion.

    Inputs:
      seasonal: (B, L, C)
      trend:    (B, L, C)
    Output:
      fused:    (B, L, C)

    Matches equations:
      x_vt = x_temp + x_var
      S = GP(x_vt)
      Z = FC(S)
      a,b = Softmax( FC_a(Z), FC_b(Z) )
      x = a*x_temp + b*x_var
    """
    def __init__(self, C, r=8, U=4, drop=0.0, pool="mean"):
        super().__init__()
        u = max(C // r, U)

        self.pool = pool
        self.fc1 = nn.Linear(C, u)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop) if drop and drop > 0 else nn.Identity()

        # two separate FCs for a and b (per-channel)
        self.fc_a = nn.Linear(u, C)
        self.fc_b = nn.Linear(u, C)

        # safe init: start close to equal weights (a=b=0.5)
        nn.init.zeros_(self.fc_a.weight); nn.init.zeros_(self.fc_a.bias)
        nn.init.zeros_(self.fc_b.weight); nn.init.zeros_(self.fc_b.bias)

        # optional normalization on output (often helps stability)
        # self.out_norm = nn.LayerNorm(C)

    def _global_pool(self, x):
        # x: (B,L,C) -> (B,C)
        if self.pool == "mean":
            return x.mean(dim=1)
        elif self.pool == "avgmax":
            return 0.5 * (x.mean(dim=1) + x.amax(dim=1))
        else:
            raise ValueError(f"Unknown pool: {self.pool}")

    def forward(self, seasonal, trend):
        # (17) sum
        x_vt = seasonal + trend                      # (B,L,C)

        # (18) global pooling
        S = self._global_pool(x_vt)                  # (B,C)

        # (19) compact guidance
        Z = self.drop(self.act(self.fc1(S)))         # (B,u)

        # (20) two heads + softmax across {a,b}
        a_logit = self.fc_a(Z)                       # (B,C)
        b_logit = self.fc_b(Z)                       # (B,C)

        w = torch.stack([a_logit, b_logit], dim=1)   # (B,2,C)
        w = F.softmax(w, dim=1)                      # (B,2,C) => a+b=1

        a = w[:, 0, :].unsqueeze(1)                  # (B,1,C)
        b = w[:, 1, :].unsqueeze(1)                  # (B,1,C)

        # (21) weighted fusion
        out = 2 * (a * seasonal + b * trend)               # (B,L,C)
        return out

import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyStableSTFusion(nn.Module):
    """
    Very stable adaptive fusion for seasonal/trend when C is small (e.g., 7).

    seasonal, trend: (B,L,C)
    returns: (B,L,C)

    Uses:
      S = pool(seasonal + trend)  -> (B,C)
      logits = Linear(C -> 2C)    -> (B,2,C)
      softmax over 2 to get a,b (a+b=1)
      y = a*seasonal + b*trend
      y = y0 + alpha*(y-y0)       (baseline-preserving)
    """
    def __init__(self, C=7, drop=0.1, pool="avgmax", clamp=(0.05, 0.95)):
        super().__init__()
        self.pool = pool
        self.clamp = clamp

        self.pre = nn.LayerNorm(C)
        self.logits = nn.Linear(C, 2 * C)
        self.drop = nn.Dropout(drop)

        # init to equal weights => logits ~ 0 => softmax => 0.5/0.5
        nn.init.zeros_(self.logits.weight)
        nn.init.zeros_(self.logits.bias)

        # fixed temperature is more stable for tiny C
        self.tau = 1.0

        # residual strength (start small)
        self.alpha = nn.Parameter(torch.tensor(-2.5))  # sigmoid ~ 0.075

        self.out_norm = nn.LayerNorm(C)

    def _pool(self, x):
        if self.pool == "mean":
            return x.mean(dim=1)
        elif self.pool == "avgmax":
            return 0.5 * (x.mean(dim=1) + x.amax(dim=1))
        else:
            raise ValueError(f"Unknown pool: {self.pool}")

    def forward(self, seasonal, trend):
        # safe baseline anchor
        y0 = 0.5 * (seasonal + trend)

        x_vt = seasonal + trend
        S = self.pre(self._pool(x_vt))               # (B,C)

        logits = self.drop(self.logits(S))           # (B,2C)
        logits = logits.view(S.size(0), 2, -1)       # (B,2,C)

        w = F.softmax(logits / self.tau, dim=1)      # (B,2,C)
        a = w[:, 0, :].unsqueeze(1)                  # (B,1,C)
        b = w[:, 1, :].unsqueeze(1)

        # optional clamp to avoid hard switching (helps stability a lot)
        if self.clamp is not None:
            lo, hi = self.clamp
            a = a.clamp(lo, hi)
            b = 1.0 - a

        y = a * seasonal + b * trend

        g = torch.sigmoid(self.alpha)                # small at start
        out = y0 + g * (y - y0)

        return out


class AdaptiveSTFusionV2(nn.Module):
    def __init__(self, C, r=20, U=1, drop=0.1, pool="mean", dmax=0.25):
        super().__init__()
        u = max(C // r, U)
        self.pool = pool
        self.dmax = dmax

        self.in_norm_s = nn.LayerNorm(C)
        self.in_norm_t = nn.LayerNorm(C)

        # gate uses [GP(s), GP(t), GP(s-t)] => 3C
        self.fc1 = nn.Linear(3 * C, u)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop) if drop and drop > 0 else nn.Identity()
        self.fc_delta = nn.Linear(u, C)

        nn.init.zeros_(self.fc_delta.weight)
        nn.init.zeros_(self.fc_delta.bias)

        # optional output norm + safe residual scaling
        self.out_norm = nn.LayerNorm(C)
        self.res_scale = nn.Parameter(torch.tensor(0.1))  # small & learnable

    def _gp(self, x):
        if self.pool == "mean":
            return x.mean(dim=1)
        elif self.pool == "avgmax":
            return 0.5 * (x.mean(dim=1) + x.amax(dim=1))
        else:
            raise ValueError(self.pool)

    def forward(self, seasonal, trend):
        s = seasonal
        t = trend

        S = torch.cat([self._gp(s), self._gp(t), self._gp(s - t)], dim=-1)  # (B,3C)
        Z = self.drop(self.act(self.fc1(S)))                                # (B,u)

        delta = torch.tanh(self.fc_delta(Z)) * self.dmax                    # (B,C)
        a = (0.5 + delta).unsqueeze(1)                                      # (B,1,C)
        b = (0.5 - delta).unsqueeze(1)                                      # (B,1,C)

        mix = 2 * (a * s + b * t)
        # base = 0.5 * (s + t)
        # out = base + self.res_scale * (mix - base)
        return mix


class SKFusionST(nn.Module):
    """
    Use SK-style Fuse+Select to fuse seasonal and trend.
    default r = 8, L_min=4,
    for ETTh2, L_min=2, better, 336, 720 pred_len
    """
    def __init__(self, C, r=8, L_min=6, drop=0.1):
        super().__init__()
        d_hidden = max(C // r, L_min)
        # L_min = 10 for

        self.fc_reduce = nn.Linear(C, d_hidden, bias=False)
        self.bn_reduce = nn.BatchNorm1d(d_hidden)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop) if drop and drop > 0 else nn.Identity()

        # 2 branches -> 2 logits heads
        self.fc_a = nn.Linear(d_hidden, C, bias=False)
        self.fc_b = nn.Linear(d_hidden, C, bias=False)

        nn.init.zeros_(self.fc_a.weight)
        nn.init.zeros_(self.fc_b.weight)

    def forward(self, seasonal, trend):
        # (B,L,C)
        U = seasonal + trend                    # Eq(1)
        s = U.mean(dim=1)                       # GAP over time -> (B,C)

        z = self.fc_reduce(s)
        z = self.bn_reduce(z)
        z = self.act(z)
        z = self.drop(z)

        a_logit = self.fc_a(z)                  # (B,C)
        b_logit = self.fc_b(z)                  # (B,C)

        logits = torch.stack([a_logit, b_logit], dim=1)   # (B,2,C)
        w = torch.softmax(logits, dim=1)                  # (B,2,C)

        # out = g * U1 + (1 - g) * U2

        a = w[:, 0].unsqueeze(1)               # (B,1,C)
        b = w[:, 1].unsqueeze(1)               # (B,1,C)

        out =  a * seasonal + b * trend

        return out

class FusionTransfer(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.transfer_st = nn.Linear(2*C, C)
        self.transfer_ts = nn.Linear(2*C, C)
        self.proj = nn.Linear(2*C, C)

    def forward(self, S, T):
        ST = torch.cat([S, T], dim=-1)
        TS = torch.cat([T, S], dim=-1)

        delta_S = self.transfer_st(ST)
        delta_T = self.transfer_ts(TS)

        S_new = S + delta_S
        T_new = T + delta_T

        FT = torch.concat([S_new, T_new], dim=-1)
        Final = self.proj(FT)

        return Final


class SKFusion_v2(nn.Module):
    """
    Selective Kernel Fusion for seasonal and trend time-series components.

    Mirrors the three-stage SK pipeline from the paper:
        Split  →  two branches are the seasonal & trend inputs (pre-computed)
        Fuse   →  element-wise sum → global avg pool → FC bottleneck → z
        Select →  softmax attention over branches, channel-wise weighted sum

    Args:
        ts_d (int):  Number of time-series variates / channels (C in the paper).
        r    (int):  Reduction ratio that controls the bottleneck dimension d.
                     d = max(ts_d // r, min_d).  Default: 2.
        min_d (int): Minimum bottleneck dimension L in the paper. Default: 32.
        M    (int):  Number of branches. Currently 2 (seasonal + trend).
    """
    def __init__(self, ts_d, r=8, min_d=7):
        super().__init__()

        self.M = 2                          # seasonal + trend
        self.ts_d = ts_d

        # Bottleneck dimension  d = max(C/r, L)
        d = max(ts_d // r, min_d)
        self.d = d

        # Fuse: shared FC layer  W ∈ R^{d × C}   (Eq. 3 in paper)
        # BN + ReLU mimic  δ(B(Ws))
        self.fc   = nn.Linear(ts_d, d, bias=False)
        self.bn   = nn.BatchNorm1d(d)

        # Select: one attention matrix per branch  A, B ∈ R^{C × d}  (Eq. 5)
        self.attn = nn.Linear(d, self.M * ts_d, bias=False)
        nn.init.zeros_(self.attn.weight)

    def forward(self, seasonal, trend):
        B, T, C = seasonal.shape
        # Element-wise sum across the M branches  (Eq. 1)
        U = seasonal + trend                            # (B, T, C)

        # Global average pooling over the time dimension → s ∈ R^{B × C}
        # Eq. 2: s_c = (1 / T) Σ_t U_c(t)
        s = U.mean(dim=1)                               # (B, C)

        # Compact feature  z = δ(BN(W s))  (Eq. 3)
        z = F.relu(self.bn(self.fc(s)))                 # (B, d)

        # SELECT
        # Project z → M × C attention logits  (Eq. 5)
        logits = self.attn(z)                           # (B, M*C)
        logits = logits.view(B, self.M, C)              # (B, M, C)

        # Softmax over the M branches (channel-wise)
        weights = F.softmax(logits, dim=1)              # (B, M, C)

        a = weights[:, 0, :]                            # (B, C)  — seasonal weight
        b = weights[:, 1, :]                            # (B, C)  — trend weight

        # Weighted sum over branches  V_c = a_c · seasonal_c + b_c · trend_c
        # Broadcast: (B, 1, C) * (B, T, C)
        V = a.unsqueeze(1) * seasonal + b.unsqueeze(1) * trend  # (B, T, C)

        return V