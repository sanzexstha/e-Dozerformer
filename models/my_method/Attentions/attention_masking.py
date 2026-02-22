import torch
import matplotlib.pyplot as plt


class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask

class ExtremeMask():
    """
    Builds a content-aware mask where attention is allowed only between tokens
    that contains the same label.
    """
    def __init__(self, x_label, device=None):
        if device is None:
            device = x_label.device

        with torch.no_grad():
            labels = x_label[..., 0]               # (B, L)
            mask = labels.unsqueeze(2).eq(labels.unsqueeze(1))  # (B, L, L) bool
            self._mask = mask.to(device) # (B,L,L)

    @property
    def mask(self):
        return self._mask


class DozerExtremeOnlyMask():
    """
    Combines (OR operator) an extremes-only content mask with a structural dozer mask.
    """
    def __init__(self, x_label, dozer_mask, device=None, extreme_value=1):
        if device is None:
            device = x_label.device

        with torch.no_grad():
            labels = x_label[..., 0].to(torch.long)      # (B, L)
            extreme = (labels == extreme_value)          # (B, L) bool
            extreme_mask = extreme.unsqueeze(2) & extreme.unsqueeze(1)  # (B,L,L)

            # normalize dozer_mask to (B,L,L)
            if dozer_mask.dim() == 2:                    # (L,L)
                dozer_b = dozer_mask.unsqueeze(0).expand(extreme_mask.size(0), -1, -1)
            else:
                raise ValueError(f"Unexpected dozer_mask shape: {dozer_mask.shape}. Both mask should be of the same shape")

            base = extreme_mask | dozer_b                # (B,L,L)
            self._mask = base.to(device)   # (B,L,L)

    @property
    def mask(self):
        return self._mask


class ExtremeDozerSparseMask:
    """
    Sparse mask:
        1) extreme_mask & dozer_mask
        2) remove query rows where label == 1
    """
    def __init__(self, extreme_mask, dozer_mask, x_label, device=None):

        if device is None:
            device = x_label.device

        with torch.no_grad():
            B, L, _ = extreme_mask.shape

            # Normalize dozer mask
            if dozer_mask.dim() == 2:  # (L, L)
                dozer_mask = dozer_mask.unsqueeze(0).expand(B, -1, -1)

            # Step 1: structural AND
            combined_mask = extreme_mask & dozer_mask  # (B,L,L)

            # Step 2: remove query rows where label == 1
            query_keep = (x_label.squeeze(-1) == 0).unsqueeze(-1)  # (B,L,1)
            combined_mask = combined_mask & query_keep  # broadcast

            self._mask = combined_mask.to(device)  # (B,1,L,L)

    @property
    def mask(self):
        return self._mask

class ConditionalExtremeDozerMask:
    """
    Row-wise conditional mask:

    For each batch b and query index i:
      if label[b,i] == 0 (normal query):
          row = dozer_mask[b,i,:] & extreme_0_1[b,i,:]
      else (extreme query):
          row = extreme_1[b,i,:]

      extreme_0_1  : (B, L, L) bool   (your content/extreme constraint)
      extreme_1 : (B, L, L) bool   (e.g., extreme<->extreme)
    """
    def __init__(self, x_label, dozer_mask, extreme_0_1, extreme_1, device=None):
        if device is None:
            device = x_label.device

        with torch.no_grad():
            B, L, _ = extreme_0_1.shape

            # labels: (B, L) bool where True means "extreme query"
            labels = x_label[..., 0].to(torch.long)          # (B,L)
            is_extreme_q = (labels == 1)                     # (B,L) bool

            # normalize dozer_mask to (B, L, L)
            if dozer_mask.dim() == 2:                        # (L,L)
                dozer_b = dozer_mask.unsqueeze(0).expand(B, -1, -1)
            else:
                raise ValueError(f"Unexpected dozer_mask shape: {dozer_mask.shape}")

            # normal rows: dozer & extreme_0_1
            normal_rows = dozer_b & extreme_0_1              # (B,L,L)

            # choose per-row using broadcasting
            # selector: (B,L,1) -> broadcast across keys (L)
            sel = is_extreme_q.unsqueeze(-1)                 # (B,L,1)
            combined = torch.where(sel, extreme_1, normal_rows)  # (B,L,L)

            self._mask = combined.to(device)     # (B,L,L)

    @property
    def mask(self):
        return self._mask


class ExtremeOnlyMask:
    """
    Builds an attention mask that is True only when BOTH query and key tokens are extreme (label==1).
    """
    def __init__(self, x_label, device=None, extreme_value=1):
        if device is None:
            device = x_label.device

        with torch.no_grad():
            labels = x_label[..., 0].to(torch.long)          # (B, L)
            extreme = (labels == extreme_value)              # (B, L) bool
            mask = extreme.unsqueeze(2) & extreme.unsqueeze(1)  # (B, L, L) bool
            self._mask = mask.to(device)        # (B, L, L)

    @property
    def mask(self):
        return self._mask

def show_mask(m):
    return m.detach().cpu().numpy()

def build_dozer_mask(L_Q, L_K, local_window=None, stride=None, device=None):
    device = device or "cpu"
    with torch.no_grad():
        i = torch.arange(L_Q, device=device)[:, None]   # [L_Q, 1]
        j = torch.arange(L_K, device=device)[None, :]   # [1, L_K]
        d = (i - j).abs()                               # [L_Q, L_K]

        mask = torch.zeros((L_Q, L_K), device=device, dtype=torch.bool)

        if local_window:
            w = local_window // 2
            mask |= (d <= w)

        if stride:
            s = stride + 1
            mask |= (d % s == 0)

        return mask

def generate_full_mask(B, L_Q, L_K, device=None):
    with torch.no_grad():
        mask = torch.ones((L_Q, L_K), device=device, dtype=torch.bool)
        return mask.repeat(B, 1, 1)

import torch

class ExtremeAndDozerMask:
    """
    Content-aware Extreme mask AND dozer) mask.
    """
    def __init__(self, x_label, dozer_mask, device=None):
        if device is None:
            device = x_label.device

        with torch.no_grad():
            labels = x_label[..., 0]  # (B, L)

            # (B, L, L): True when same label
            adapt_mask = labels.unsqueeze(2).eq(labels.unsqueeze(1))

            # normalize dozer_mask to (B, L, L)
            if dozer_mask.dim() == 2:                 # (L, L)
                B = labels.shape[0]
                dozer_b = dozer_mask.unsqueeze(0).expand(B, -1, -1)
            else:
                raise ValueError(f"Unexpected dozer_mask shape: {dozer_mask.shape}")

            base_mask = adapt_mask & dozer_b          # (B, L, L)
            self._mask = base_mask.to(device)  # (B,L,L)

    @property
    def mask(self):
        return self._mask


class SparseMask:
    def __init__(self, x_label, local_window, stride, device, B, L_Q, L_K):
        self.x_label = x_label
        self.local_window = local_window
        self.stride = stride
        self.device = device
        self.B = B
        self.L_Q = L_Q
        self.L_K = L_K
        self._mask = None

    def generate_mask(self, mask='dozer'):
        if mask == 'extreme_mask':
            self._mask = ExtremeMask(self.x_label).mask

        else:
            dozer_mask = build_dozer_mask(
                self.L_Q, self.L_K,
                local_window=self.local_window,
                stride=self.stride,
                device=self.device
            )

            if mask == 'dozer':
                self._mask = dozer_mask.unsqueeze(0).repeat(self.B, 1, 1)
            elif mask == 'dozer_ext_only':
                self._mask = DozerExtremeOnlyMask(self.x_label, dozer_mask).mask
            elif mask == 'dozer_ext_0':
                extreme_0_1 = ExtremeMask(self.x_label).mask
                extreme_1 = ExtremeOnlyMask(self.x_label).mask
                self._mask = ConditionalExtremeDozerMask(self.x_label, dozer_mask, extreme_0_1, extreme_1).mask
            elif mask == 'dozer_ext_0_v2':
                self._mask = ConditionalExtremeDozerMaskV2(self.x_label, dozer_mask, self.local_window).mask
            elif mask == 'dozer_ext_0_v3':
                self._mask = ConditionalExtremeDozerMaskV3(self.x_label, dozer_mask, self.local_window, ext_key_window=12).mask
            elif mask == 'dozer_ext_0_v4':
                self._mask = DozerExtremeHaloMaskV3(self.x_label, dozer_mask).mask
            elif mask == 'dozer_ext_null':
                self._mask = ExtremeDozerSparseMask(ExtremeMask(self.x_label).mask, dozer_mask, self.x_label).mask
            elif mask == 'dozer_AND_ext':
                self._mask = ExtremeAndDozerMask(self.x_label, dozer_mask).mask
            elif mask == 'full_mask':
                self._mask = generate_full_mask(self.B, self.L_Q, self.L_K, self.device)
            else:
                raise ValueError(f"Unknown mask type: {mask}")

        return self._mask

    def visualize_mask(self, mask='dozer'):
        MASK_TYPES = ['extreme_mask', 'dozer', 'dozer_ext_only', 'full_mask', 'dozer_ext_0', 'dozer_ext_null', 'dozer_AND_ext', 'dozer_ext_0_v2', 'dozer_ext_0_v3', 'dozer_ext_0_v4']

        if mask == 'all':
            results = {}
            for mask_type in MASK_TYPES:
                m = self.generate_mask(mask_type)
                results[mask_type] = m.detach().cpu().numpy()
            return results

        m = self.generate_mask(mask)
        return m.detach().cpu().numpy()

        # # If batched (3D), take the first batch element
        # mask_2d = self._mask[0] if self._mask.dim() == 3 else self._mask
        # mask_np = mask_2d.cpu().numpy()
        #
        # plt.figure(figsize=(10, 8))
        # plt.imshow(mask_np, aspect='auto', cmap='Blues', interpolation='none')
        # plt.colorbar(label='Mask Value')
        # plt.title(title)
        # plt.xlabel('Key Position (L_K)')
        # plt.ylabel('Query Position (L_Q)')
        # plt.tight_layout()
        # plt.show()

import torch

def build_local_band_mask(L_Q, L_K, local_window, device):
    """
    Returns a (L_Q, L_K) bool mask for the local window band only.
    """
    if not local_window:
        # If no local window specified, treat as "no restriction"
        return torch.ones((L_Q, L_K), device=device, dtype=torch.bool)

    i = torch.arange(L_Q, device=device)[:, None]
    j = torch.arange(L_K, device=device)[None, :]
    d = (i - j).abs()
    w = local_window // 2
    return (d <= w)


class ConditionalExtremeDozerMaskV2:
    """
    Desired behavior:
    - If query is EXTREME: attend only within local-window band (no stride far links).
    - If query is NORMAL: follow dozer mask, but block extreme KEYS unless they are local-window-close.
    """
    def __init__(self, x_label, dozer_mask, local_window, device=None, extreme_value=1):
        """
        x_label: (B, L, ...) where x_label[...,0] holds 0/1 labels
        dozer_mask: (L_Q, L_K) bool (structural: local OR stride)
        local_window: int, must match the local window used to define "near"
        """
        if device is None:
            device = x_label.device

        with torch.no_grad():
            labels = x_label[..., 0].to(torch.long)           # (B, L_Q) assuming L_Q == L
            B, L_Q = labels.shape
            # infer L_K from dozer_mask
            if dozer_mask.dim() != 2:
                raise ValueError(f"Expected dozer_mask (L_Q,L_K), got {tuple(dozer_mask.shape)}")
            L_K = dozer_mask.size(1)
            if dozer_mask.size(0) != L_Q:
                raise ValueError(f"dozer_mask first dim {dozer_mask.size(0)} != labels length {L_Q}")

            q_is_ext = (labels == extreme_value)              # (B, L_Q)
            # key labels (assume same timeline); if you ever have different key labels, pass them separately
            k_is_ext = (labels == extreme_value)              # (B, L_K) if L_K==L_Q; else this assumption breaks
            if L_K != L_Q:
                raise ValueError("L_K != L_Q: need separate key labels to handle this case correctly.")

            # Expand dozer to batch
            dozer_b = dozer_mask.to(device).unsqueeze(0).expand(B, -1, -1)   # (B,L_Q,L_K)

            # Local band (only the local-window section of dozer)
            local_band = build_local_band_mask(L_Q, L_K, local_window, device=device)  # (L_Q,L_K)
            local_b = local_band.unsqueeze(0).expand(B, -1, -1)                          # (B,L_Q,L_K)

            # 1) For EXTREME queries: only local band, nothing else
            mask_extreme_queries = local_b                                             # (B,L_Q,L_K)

            # 2) For NORMAL queries: dozer, but if key is extreme, require local band
            # Allow = dozer AND ( (key is not extreme) OR (key is extreme AND local) )
            k_ext_b = k_is_ext.unsqueeze(1).expand(B, L_Q, L_K)                         # (B,L_Q,L_K)
            allow_extreme_keys_only_if_local = (~k_ext_b) | (k_ext_b & local_b)
            mask_normal_queries = dozer_b & allow_extreme_keys_only_if_local            # (B,L_Q,L_K)

            # Select per-query-row based on whether query token is extreme
            q_ext_b = q_is_ext.unsqueeze(2).expand(B, L_Q, L_K)                          # (B,L_Q,L_K)
            self._mask = torch.where(q_ext_b, mask_extreme_queries, mask_normal_queries).to(device)

    @property
    def mask(self):
        return self._mask

import torch

def build_distance_mask(L_Q, L_K, max_dist, device):
    i = torch.arange(L_Q, device=device)[:, None]
    j = torch.arange(L_K, device=device)[None, :]
    d = (i - j).abs()
    return (d <= max_dist)  # (L_Q, L_K) bool


class ConditionalExtremeDozerMaskV3:
    """
    - Extreme queries: attend ONLY within local band (w_local).
    - Normal queries:
        * non-extreme keys: follow dozer mask
        * extreme keys: allow if within w_ext (>= w_local), otherwise block
    """
    def __init__(
        self,
        x_label,
        dozer_mask,         # (L,L) bool
        local_window,       # int
        ext_key_window=None,# int OR None: allowed window for extreme keys when query is normal
        device=None,
        extreme_value=1
    ):
        if device is None:
            device = x_label.device

        with torch.no_grad():
            labels = x_label[..., 0].to(torch.long)      # (B, L)
            B, L = labels.shape

            if dozer_mask.dim() != 2 or dozer_mask.size(0) != L or dozer_mask.size(1) != L:
                raise ValueError(f"Expected dozer_mask (L,L) matching labels length {L}, got {tuple(dozer_mask.shape)}")

            # radii
            if not local_window:
                raise ValueError("local_window must be set for this mask design.")
            w_local = local_window // 2

            # if not provided, default: allow extreme keys only in local band
            if ext_key_window is None:
                w_ext = w_local
            else:
                w_ext = ext_key_window // 2 if ext_key_window >= 2 else 0
                # ensure it's not smaller than local
                w_ext = max(w_ext, w_local)

            # query/key flags
            q_is_ext = (labels == extreme_value)          # (B,L)
            k_is_ext = (labels == extreme_value)          # (B,L)

            # expand dozer mask to batch
            dozer_b = dozer_mask.to(device).unsqueeze(0).expand(B, -1, -1)  # (B,L,L)

            # distance masks
            local_band = build_distance_mask(L, L, w_local, device=device)  # (L,L)
            local_b = local_band.unsqueeze(0).expand(B, -1, -1)             # (B,L,L)

            ext_band = build_distance_mask(L, L, w_ext, device=device)      # (L,L)
            ext_b = ext_band.unsqueeze(0).expand(B, -1, -1)                 # (B,L,L)

            # 1) EXTREME queries -> ONLY local band
            mask_extreme_queries = local_b                                  # (B,L,L)

            # 2) NORMAL queries:
            # non-extreme keys -> dozer
            # extreme keys     -> allowed only if within ext_b
            k_ext_b = k_is_ext.unsqueeze(1).expand(B, L, L)                 # (B,L,L)
            allow_keys = (~k_ext_b) | (k_ext_b & ext_b)                     # (B,L,L)
            mask_normal_queries = dozer_b & allow_keys                      # (B,L,L)

            # select per query row
            q_ext_b = q_is_ext.unsqueeze(2).expand(B, L, L)                 # (B,L,L)
            self._mask = torch.where(q_ext_b, mask_extreme_queries, mask_normal_queries)

    @property
    def mask(self):
        return self._mask

import torch

def _distance_matrix(L, device):
    i = torch.arange(L, device=device)[:, None]
    j = torch.arange(L, device=device)[None, :]
    return (i - j).abs()  # (L, L)

class DozerExtremeHaloMaskV3:
    """
    NORMAL queries:
      dozer
      OR (extreme keys within extreme_key_radius_for_normal)
      OR (halo-normal within halo_nn_radius, only if both tokens are in halo)

    EXTREME queries:
      dozer (local + stride)
      OR (extra neighbor band within ext_query_neighbor_radius)
    """
    def __init__(
        self,
        x_label,                          # (B, L, 1+) labels in x_label[...,0]
        dozer_mask,                       # (L, L) bool (local OR stride)
        extreme_value=1,

        # normal-query extras
        extreme_key_radius_for_normal=4,  # normal q -> extreme k reach (distance in tokens)
        halo_radius=3,                    # token is in halo if within this distance to any extreme token
        halo_nn_radius=2,                 # within-halo normal-normal reach

        # NEW: extreme-query local expansion
        ext_query_neighbor_radius=2,      # extra band radius for extreme queries (>= local_radius)
        device=None,
    ):
        device = device or x_label.device

        with torch.no_grad():
            labels = x_label[..., 0].to(torch.long)  # (B, L)
            B, L = labels.shape

            if dozer_mask.dim() != 2 or dozer_mask.shape != (L, L):
                raise ValueError(f"dozer_mask must be (L,L) with L={L}, got {tuple(dozer_mask.shape)}")

            q_is_ext = (labels == extreme_value)  # (B, L)
            k_is_ext = (labels == extreme_value)  # (B, L)

            dozer_b = dozer_mask.to(device).unsqueeze(0).expand(B, -1, -1)  # (B,L,L)

            d = _distance_matrix(L, device=device)      # (L,L)
            d_b = d.unsqueeze(0).expand(B, -1, -1)      # (B,L,L)

            # -------------------------
            # (A) Normal queries: extra reach to EXTREME KEYS
            # -------------------------
            ext_reach_b = (d_b <= extreme_key_radius_for_normal)            # (B,L,L)
            k_ext_b = k_is_ext.unsqueeze(1).expand(B, L, L)                 # (B,L,L)
            add_ext_keys = k_ext_b & ext_reach_b                            # (B,L,L)

            # -------------------------
            # (B) Halo: tokens near ANY extreme token
            # -------------------------
            big = L + 1
            masked_d = d_b.masked_fill(~k_ext_b, big)                       # distances only to extreme keys
            min_d_to_ext, _ = masked_d.min(dim=2)                           # (B,L)
            in_halo = (min_d_to_ext <= halo_radius)                         # (B,L)

            halo_band_b = (d_b <= halo_nn_radius)                           # (B,L,L)
            q_halo_b = in_halo.unsqueeze(2).expand(B, L, L)
            k_halo_b = in_halo.unsqueeze(1).expand(B, L, L)
            add_halo_nn = q_halo_b & k_halo_b & halo_band_b                 # (B,L,L)

            mask_normal = dozer_b | add_ext_keys | add_halo_nn

            # -------------------------
            # (C) Extreme queries: dozer + expanded neighbor band
            # -------------------------
            # extreme queries attend not only to dozer (local+stride)
            # but also to neighboring keys around local band (thicker band).
            ext_neighbor_b = (d_b <= ext_query_neighbor_radius)             # (B,L,L)
            mask_extreme = dozer_b | ext_neighbor_b

            # select per query row
            q_ext_b = q_is_ext.unsqueeze(2).expand(B, L, L)
            self._mask = torch.where(q_ext_b, mask_extreme, mask_normal)

    @property
    def mask(self):
        return self._mask


# B, L, _ = extreme_0_1.shape
#
# combined_mask = torch.zeros_like(extreme_0_1)
#
# for b in range(B):
#     for i in range(L):
#         if labels[b, i, 0] == 1:
#             continue
#         combined_mask[b, i, :] = extreme_0_1[b, i, :] & dozer_mask[b, i, :]

# labels = x_label[:, :, 0]  # [batch_size, L_Q]

# # Yifan's implementation
# extreme_0_1 = labels.unsqueeze(2).eq(labels.unsqueeze(1))
# batch_size = extreme_0_1.shape[0]
# dozer_mask = dozer_mask.repeat(batch_size, 1, 1)
# a_0_1 = extreme_0_1.detach().cpu().numpy()
# for i in range(extreme_0_1.shape[1]):
#     if labels[i] == 1:
#         continue
#     combined_mask = extreme_0_1[:, i, :] & dozer_mask[:, i, :]
# c_0_1 = combined_mask.detach().cpu().numpy()


def build_dozer_mask_batched(
    labels,                 # (B, L, 1) or (B, L)
    local_window=None,
    stride=None,
    precursor_window=None,
    recovery_window=None,
    extreme_to_extreme=True,
    is_training=True,
    device=None
):
    device = device or labels.device

    # squeeze last dim if needed → (B, L)
    if labels.dim() == 3:
        labels = labels.squeeze(-1)

    B, L = labels.shape

    with torch.no_grad():
        i = torch.arange(L, device=device)[None, :, None]  # (1, L, 1)
        j = torch.arange(L, device=device)[None, None, :]  # (1, 1, L)
        d = (i - j).abs()                                  # (1, L, L) → broadcasts over B

        # base mask — same for all samples in batch
        mask = torch.zeros((1, L, L), device=device, dtype=torch.bool)
        mask = mask.expand(B, L, L).clone()                # (B, L, L)

        #  existing dozer
        if local_window:
            w = local_window // 2
            mask |= (d <= w)

        if stride:
            s = stride + 1
            mask |= (d % s == 0)

        #  extreme components
        labels_bool = labels.bool()                        # (B, L)

        # extreme_Q: (B, L, 1)  extreme_K: (B, 1, L)
        extreme_Q = labels_bool[:, :, None]                # (B, L, 1)
        extreme_K = labels_bool[:, None, :]                # (B, 1, L)

        causal    = (i >= j)                               # (1, L, L)
        lookahead = (i < j)                                # (1, L, L)

        # 1. PRECURSOR LOOK-AHEAD (training only)
        if precursor_window and is_training:
            near_future_extreme = extreme_K & lookahead & ((j - i) <= precursor_window)
            mask |= near_future_extreme                    # (B, L, L)

        # 2. RECOVERY LOOK-BACK (always)
        if recovery_window:
            past_extreme_nearby = extreme_K & causal & ((i - j) <= recovery_window)
            mask |= past_extreme_nearby

        # 3. EXTREME-TO-EXTREME (always)
        if extreme_to_extreme:
            both_extreme_causal = extreme_Q & extreme_K & causal
            mask |= both_extreme_causal

    return mask  # (B, L, L) — True means can attend

# base_mask = build_dozer_mask_batched(
#     labels=x_label,
#     local_window=self.local_window,
#     stride=self.stride,
#     precursor_window=2,
#     recovery_window=4,
#     extreme_to_extreme=True,
#     is_training=True
# )