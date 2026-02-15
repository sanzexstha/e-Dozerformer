import torch
import torch.nn as nn
from math import sqrt
from einops import repeat
from flash_sparse_attn import flash_sparse_attn_func_auto
from utils.tools import mask_to_bias

class DozerAttention(nn.Module):
    def __init__(self, local_window, stride, rand_rate, vary_len, pred_len,
                 in_channel, mask_flag=True, scale=None,
                 attention_dropout=0.1, output_attention=False):
        super(DozerAttention, self).__init__()
        self.scale = scale
        self.local_window = local_window
        self.stride = stride
        self.rand_rate = rand_rate
        self.vary_len = vary_len
        self.mask_flag = mask_flag
        self.pred_len = pred_len
        self.in_channel = in_channel
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.base_mask = None
        self.adapt_mask = None
        # self.register_buffer("flops_accum", torch.zeros(1))


    def forward(self, queries, keys, values, x_label, attn_mask):
        # queries: [B, L_Q, H, D]
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape
        batch_size, _, _ = x_label.shape
        orig_dtype = queries.dtype

        scale = self.scale or 1. / sqrt(D)

        # Build Dozer-style mask
        # base local/stride mask: [L_Q, L_K]
        dozer_mask = torch.zeros(L_Q, L_K, device=queries.device, dtype=torch.bool)


        if L_Q == L_K:
            if self.local_window:
                for w_idx in range(self.local_window//2+1):
                    dozer_mask = torch.diagonal_scatter(dozer_mask, torch.ones(L_Q - w_idx), w_idx)
                    dozer_mask = torch.diagonal_scatter(dozer_mask, torch.ones(L_Q - w_idx), -w_idx)

            if self.stride:
                stride = self.stride + 1
                for w_idx in range(0, L_Q, stride):
                    dozer_mask = torch.diagonal_scatter(dozer_mask, torch.ones(L_Q - w_idx), w_idx)
                    dozer_mask = torch.diagonal_scatter(dozer_mask, torch.ones(L_Q - w_idx), -w_idx)
            b = dozer_mask.detach().cpu().numpy()
            # vectorized content-aware mask
            labels = None
            labels = x_label[:, :, 0].to(torch.bool)  # [batch_size, L_Q]
            adapt_mask = labels.unsqueeze(2).eq(labels.unsqueeze(1))  # [batch_size, L_Q, L_K], bool
            base_mask = None

            # labels: [B, L] (from x_label[:, :, 0])
            # labels = x_label[:, :, 0].to(torch.bool)  # if label are 0/1
            #
            # # only tokens with label==1 are considered "extreme"
            # extreme = (labels == 1)    # [B, L] bool
            #
            # # extremes-only content mask: allow attention only among "1" tokens
            # extreme_mask = extreme.unsqueeze(2) & extreme.unsqueeze(1)  # [B, L, L] bool

            # OR with structural dozer mask
            # base_mask = extreme_mask | dozer_mask.unsqueeze(0)  # [B, L, L] bool

            # combine locality + content
            # adapt mask perform better than combination with dozer_mask
            # base_mask = adapt_mask | dozer_mask.unsqueeze(0)  # [batch_size, L_Q, L_K]
            full_mask = torch.ones(L_Q, L_K, device=queries.device, dtype=torch.bool)
            # base_mask = full_mask.unsqueeze(0).expand(batch_size, -1, -1)
            base_mask = dozer_mask.unsqueeze(0).expand(batch_size, -1, -1).clone()

            # base_mask = adapt_mask  # [batch_size, L_Q, L_K]

            # expand to B = batch_size * in_channel
            adapt_dozer_mask = repeat(base_mask, 'b seg_num c -> (b ts_d) seg_num c', ts_d=self.in_channel)

            # Final mask for FSA: [B, H, L_Q, L_K] (bool)
            # attn_mask = adapt_dozer_mask.unsqueeze(1)c
            attn_mask = adapt_dozer_mask.unsqueeze(1).expand(-1, H, -1, -1)

            adapt_mask = repeat(adapt_mask, 'b seg_num c -> (b ts_d) seg_num c', ts_d=self.in_channel)
            adapt_mask = adapt_mask.unsqueeze(1).expand(-1, H, -1, -1)



        # Run Flash Sparse Attention
        flash_sparse_attn_func = flash_sparse_attn_func_auto(backend="cuda")
        target_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        queries = queries.to(target_dtype)
        keys    = keys.to(target_dtype)
        values  = values.to(target_dtype)
        attn_bias = mask_to_bias(adapt_mask, queries)  # query is your Q tensor

        # active = attn_mask.to(torch.bool).sum()

        # flops = 2 * H * active * D

        attn = flash_sparse_attn_func(
            query=queries,
            key=keys,
            value=values,
            attn_mask=attn_mask,  # bool, [B, H, L_Q, L_K]
            attn_bias=attn_bias,
            softmax_scale=scale,
        )
        # self.flops_accum = flops / 1e6

        attn = attn.to(orig_dtype)

        if self.output_attention:
            return attn, None
        return attn, None


class DozerAttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(DozerAttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, x_label, attn_mask):
        x = torch.clone(queries)
        # Batch size, Seq len, embed_dim
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        # Batch size, Seq len, head, embed_dim/head
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            x_label,
            attn_mask
        )

        out = out.view(B, L, -1)
        out = self.out_projection(out)

        return out, attn


