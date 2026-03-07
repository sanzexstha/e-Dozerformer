import torch
from flash_sparse_attn import flash_sparse_attn_func_auto
from flash_sparse_attn.utils.mask import create_mask
import math

# Setup
batch_size, seq_len, num_heads, num_kv_heads, head_dim = 1, 256, 2, 1, 64
window_size = 128
device = torch.device('cuda')
dtype = torch.bfloat16
min_dtype = torch.finfo(dtype).min  # dtype minimum value

# Input tensors
query = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
key = torch.randn(batch_size, seq_len, num_kv_heads, head_dim, device=device, dtype=dtype)
value = torch.randn(batch_size, seq_len, num_kv_heads, head_dim, device=device, dtype=dtype)

# Create bias for sparse attention
attn_bias = torch.randn(batch_size, num_kv_heads, 1, seq_len, device=device, dtype=dtype)

# Generate dynamic mask based on bias
if seq_len > window_size:
    attn_mask = create_mask(
        attention_bias=attn_bias,
        attention_mask=None,
        batch_size=batch_size,
        query_len=seq_len,
        key_len=seq_len,
        window_size=window_size,
        min_dtype=min_dtype,
    )

import torch
import torch.nn as nn
from flash_sparse_attn import flash_sparse_attn_func_auto

class FlashSparseAttnModule(nn.Module):
    def __init__(self, window_size=128, is_causal=True, head_dim=64):
        super().__init__()
        self.window_size = window_size
        self.is_causal = is_causal
        self.head_dim = head_dim
        self.flash_sparse_attn = flash_sparse_attn_func_auto(backend="cuda")

    def forward(self, query, key, value, attn_mask, attn_bias):
        return self.flash_sparse_attn(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            attn_bias=attn_bias,
            is_causal=self.is_causal,
            softmax_scale=1.0 / (self.head_dim ** 0.5),
        )

from fvcore.nn import FlopCountAnalysis

model = FlashSparseAttnModule(window_size=128, is_causal=True, head_dim=64).cuda()

inputs = (query, key, value, attn_mask, attn_bias)
flops = FlopCountAnalysis(model, inputs)

print("Total (default) flops:", flops.total())
print("By operator:", flops.by_operator())
print("Unsupported ops:", flops.unsupported_ops())
