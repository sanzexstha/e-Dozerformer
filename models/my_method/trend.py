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
    def __init__(self, C, r=8, L_min=4, drop=0.1):
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
import torch
import torch.nn as nn
import torch.nn.functional as F

class SKFusionST_v2(nn.Module):
    """
    SK-style fusion with:
    - global channel gate (like your original)
    - local time-aware gate (depthwise temporal conv)
    - optional energy-consistent residual path to avoid hurting reconstruction
    """
    def __init__(self, C, r=8, L_min=4, drop=0.1, k=9, alpha=0.5, tau=1.0, use_energy_residual=True):
        super().__init__()
        d_hidden = max(C // r, L_min)

        # global gate (channel-wise)
        self.fc_reduce = nn.Linear(C, d_hidden, bias=True)   # bias=True often helps here
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop) if drop and drop > 0 else nn.Identity()
        self.fc_a = nn.Linear(d_hidden, C, bias=True)
        self.fc_b = nn.Linear(d_hidden, C, bias=True)

        # local gate (time-wise, channel-wise): depthwise conv over time
        # input: (B, C, L) -> output: (B, C, L)
        self.dwconv = nn.Conv1d(C, C, kernel_size=k, padding=k//2, groups=C, bias=True)
        self.pw = nn.Conv1d(C, 2*C, kernel_size=1, bias=True)  # -> logits for a,b per time

        # init so it starts close to equal mixing
        nn.init.zeros_(self.fc_a.weight); nn.init.zeros_(self.fc_b.weight)
        nn.init.zeros_(self.fc_a.bias);   nn.init.zeros_(self.fc_b.bias)
        nn.init.zeros_(self.pw.weight);   nn.init.zeros_(self.pw.bias)

        self.alpha = alpha   # blend global vs local gate
        self.tau = tau       # softmax temperature
        self.use_energy_residual = use_energy_residual

    def forward(self, seasonal, trend):
        # seasonal, trend: (B, L, C)
        B, L, C = seasonal.shape
        U = seasonal + trend                     # reconstruction backbone

        # -------- global gate (B, 2, C) --------
        s = U.mean(dim=1)                        # (B, C)
        z = self.drop(self.act(self.fc_reduce(s)))   # (B, d_hidden)
        a_g = self.fc_a(z)                       # (B, C)
        b_g = self.fc_b(z)                       # (B, C)
        w_g = torch.softmax(torch.stack([a_g, b_g], dim=1) / self.tau, dim=1)  # (B,2,C)
        w_g = w_g.unsqueeze(2)                   # (B,2,1,C)

        # -------- local gate (B, 2, L, C) --------
        x = U.transpose(1, 2)                    # (B, C, L)
        x = self.act(self.dwconv(x))
        logits = self.pw(x)                      # (B, 2C, L)
        logits = logits.view(B, 2, C, L).permute(0, 1, 3, 2)   # (B,2,L,C)
        w_l = torch.softmax(logits / self.tau, dim=1)          # (B,2,L,C)

        # -------- blend gates --------
        # broadcast global to (B,2,L,C)
        w = self.alpha * w_l + (1 - self.alpha) * w_g.expand(-1, -1, L, -1)

        a = w[:, 0]   # (B,L,C)
        b = w[:, 1]   # (B,L,C)

        fused = a * seasonal + b * trend         # gated fusion

        # energy-consistent residual: keep reconstruction backbone, learn only a correction
        if self.use_energy_residual:
            # correction is (fused - U); start near 0 due to init
            return U + (fused - U)
        else:
            return fused


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



import torch
import torch.nn as nn
import torch.nn.functional as F

class SKFusion_EIA_ST(nn.Module):
    """
    SK-style gating (pool -> bottleneck -> select) adapted for seasonal/trend,
    enforcing a+b=1 via beta and using EIA scaling to preserve energy.

    Inputs: seasonal, trend: (B, L, C)
    Output: fused: (B, L, C)
    """
    def __init__(self, C, r=8, U=4, k_pool=96, act="gelu", drop=0.0,
                 use_ln=True, eia_scale=True):
        super().__init__()
        self.k_pool = k_pool
        self.eia_scale = eia_scale

        u = max(C // r, U)

        self.use_ln = use_ln
        self.ln_s = nn.LayerNorm(C) if use_ln else nn.Identity()
        self.ln_t = nn.LayerNorm(C) if use_ln else nn.Identity()

        self.fc_reduce = nn.Linear(C, u, bias=True)
        self.act = nn.GELU() if act == "gelu" else nn.ReLU(inplace=True) if act == "relu" else nn.Identity()
        self.drop = nn.Dropout(drop) if drop and drop > 0 else nn.Identity()

        # single head -> beta
        self.fc_beta = nn.Linear(u, C, bias=True)

        # safe init: start near beta=0.5 (so output ≈ seasonal+trend with EIA)
        nn.init.zeros_(self.fc_beta.weight)
        nn.init.zeros_(self.fc_beta.bias)

    def forward(self, seasonal, trend):
        # (B, L, C)
        assert seasonal.shape == trend.shape
        B, L, C = seasonal.shape

        # Normalize only for gating stability (optional but helps Weather)
        sN = self.ln_s(seasonal)
        tN = self.ln_t(trend)

        # SK "Fuse": combine branches to get descriptor
        U = sN + tN  # (B, L, C)

        # horizon-aware pooling (Weather-safe)
        k = min(self.k_pool, L)
        S = U[:, -k:, :].mean(dim=1)  # (B, C)

        # bottleneck
        Z = self.fc_reduce(S)
        Z = self.act(Z)
        Z = self.drop(Z)

        # SK "Select": beta in (0,1), and complementary weights
        beta = torch.sigmoid(self.fc_beta(Z)).unsqueeze(1)  # (B, 1, C)
        a = beta
        b = 1.0 - beta

        # decomposition-friendly fusion
        out = a * seasonal + b * trend

        # EIA energy invariance (so beta=0.5 -> seasonal+trend)
        if self.eia_scale:
            out = 2.0 * out

        return out


import torch
import torch.nn as nn

class HybridSKFusion(nn.Module):
    """
    Hybrid SK fusion for seasonal/trend (time-series):
      - Global SK gate (per-channel, stable): g_g  (B,1,C)
      - Local  SK gate (per-time, adaptive):  g_l  (B,L,C)
      - Combine gates in signed EIA form:
          out = (s+t) + g*(s-t),  g in [-1,1]

    Inputs:
      seasonal, trend: (B,L,C)
    Output:
      out:            (B,L,C)
    """
    def __init__(
        self,
        C: int,
        r_global: int = 8,
        Lmin_global: int = 4,
        hidden_mul_local: int = 2,
        drop: float = 0.0,
        lam_local: float = 1.0,     # strength of local correction
        gate_clip: float = 1.0      # keep 1.0; <1 makes gate more conservative
    ):
        super().__init__()
        self.lam_local = lam_local
        self.gate_clip = gate_clip

        # -------- Global SK gate (Fuse: U -> GAP -> bottleneck -> g_g) --------
        d_g = max(C // r_global, Lmin_global)
        self.g_fc1 = nn.Linear(C, d_g, bias=True)
        self.g_act = nn.GELU()
        self.g_drop = nn.Dropout(drop) if drop and drop > 0 else nn.Identity()
        self.g_fc2 = nn.Linear(d_g, C, bias=True)   # outputs g_g logits -> tanh

        # Start near SUM: g_g ~ 0
        nn.init.zeros_(self.g_fc2.weight)
        nn.init.zeros_(self.g_fc2.bias)

        # -------- Local gate (Fuse: concat(s,t) per time -> MLP -> g_l) --------
        H = hidden_mul_local * C
        self.l_fc1 = nn.Linear(2 * C, H, bias=True)
        self.l_act = nn.GELU()
        self.l_drop = nn.Dropout(drop) if drop and drop > 0 else nn.Identity()
        self.l_fc2 = nn.Linear(H, C, bias=True)

        # Start near SUM: g_l ~ 0
        nn.init.zeros_(self.l_fc2.weight)
        nn.init.zeros_(self.l_fc2.bias)

    def forward(self, seasonal: torch.Tensor, trend: torch.Tensor) -> torch.Tensor:
        # seasonal, trend: (B,L,C)
        U = seasonal + trend          # (B,L,C)
        D = seasonal - trend          # (B,L,C)

        # ---- global gate ----
        s = U.mean(dim=1)                                # (B,C)  GAP over time
        zg = self.g_drop(self.g_act(self.g_fc1(s)))       # (B,d_g)
        g_g = torch.tanh(self.g_fc2(zg)).unsqueeze(1)     # (B,1,C) in [-1,1]

        # ---- local gate ----
        yC = torch.cat([seasonal, trend], dim=-1)         # (B,L,2C)
        zl = self.l_drop(self.l_act(self.l_fc1(yC)))      # (B,L,H)
        g_l = torch.tanh(self.l_fc2(zl))                  # (B,L,C) in [-1,1]

        # ---- combine gates (bounded) ----
        g = torch.tanh(g_g + self.lam_local * g_l)        # (B,L,C)
        if self.gate_clip != 1.0:
            g = self.gate_clip * g                        # conservative gating if <1

        # ---- energy-invariant fusion (EIA form) ----
        out = U + g * D                                   # (B,L,C)
        return out


import torch
import torch.nn as nn

class EIADiffGate(nn.Module):
    """
    Stable EIA-style fusion:
      out = (seasonal + trend) + g * (seasonal - trend)
    where g in [-1, 1] (learned, time-dependent).
    seasonal, trend: (B,L,C)
    """
    def __init__(self, C, hidden_mul=2, drop=0.0):
        super().__init__()
        H = hidden_mul * C

        self.net = nn.Sequential(
            nn.Linear(2*C, H),
            nn.GELU(),
            nn.Dropout(drop) if drop and drop > 0 else nn.Identity(),
            nn.Linear(H, C),
        )

        # Start at SUM: g ~ 0
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, seasonal, trend):
        U = seasonal + trend                # (B,L,C)
        D = seasonal - trend                # (B,L,C)

        yC = torch.cat([seasonal, trend], dim=-1)  # (B,L,2C)
        g = torch.tanh(self.net(yC))        # (B,L,C) in [-1,1]

        out = U + g * D
        return out


class SKFusionResidual(nn.Module):
    """
    out = (seasonal + trend) + alpha * ( a*seasonal + b*trend - (seasonal + trend) )
    Starts as exact SUM and learns small deviations.
    """
    def __init__(self, C, r=8, L_min=4, drop=0.1, temp=1.0):
        super().__init__()
        d = max(C // r, L_min)

        self.fc_reduce = nn.Linear(C, d, bias=True)
        self.norm = nn.LayerNorm(d)          # BN often unstable for gating
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop) if drop and drop > 0 else nn.Identity()

        self.fc_a = nn.Linear(d, C, bias=True)
        self.fc_b = nn.Linear(d, C, bias=True)

        # Start near SUM: logits ~0 => a=b=0.5
        nn.init.zeros_(self.fc_a.weight); nn.init.zeros_(self.fc_a.bias)
        nn.init.zeros_(self.fc_b.weight); nn.init.zeros_(self.fc_b.bias)

        self.alpha = nn.Parameter(torch.tensor(0.0))  # start as exact SUM
        self.temp = temp

    def forward(self, seasonal, trend):
        U = seasonal + trend              # safe baseline
        s = U.mean(dim=1)                 # (B,C)

        z = self.drop(self.act(self.norm(self.fc_reduce(s))))  # (B,d)

        logits = torch.stack([self.fc_a(z), self.fc_b(z)], dim=1)  # (B,2,C)
        w = torch.softmax(logits / self.temp, dim=1)
        a = w[:, 0].unsqueeze(1)
        b = w[:, 1].unsqueeze(1)

        fused = a * seasonal + b * trend
        out = U + self.alpha * (fused - U)
        return out

