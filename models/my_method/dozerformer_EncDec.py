import math
import torch
from torch import nn
from einops import rearrange
from models.my_method.Attentions.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from models.my_method.Attentions.SelfAttention_Family import FullAttention, AttentionLayer
from models.my_method.Attentions.flash_dozer import DozerAttention, DozerAttentionLayer
from models.my_method.build_model_util import DI_embedding, TS_Segment, series_decomp_multi
from math import ceil


import torch
from einops import rearrange, reduce, repeat

def labels_to_segments(y, patch_size, in_channel=None):
    B, L, C = y.shape
    seg_num = L // patch_size  # number of segments

    # reshape into (B, S, patch_size, 1)
    y_seg = rearrange(y, 'b (seg_num seg_len) c -> b seg_num seg_len c',
                      seg_num=seg_num, seg_len=patch_size)

    # count number of 1s in each patch
    count_ones = reduce(y_seg, 'b seg_num seg_len c -> b seg_num c', 'sum')  # (B, S, 1)

    # threshold rule: if more than 1 '1' in the patch
    y_major = (count_ones >= 1).to(y.dtype)

    # if in_channel is not None:
    #     y_major = repeat(y_major, 'b seg_num c -> (b ts_d) seg_num c', ts_d=in_channel)  # (out_batch, S, 1)

    return y_major



class dozerformer_Encoder(nn.Module):
    def __init__(self, configs, mode):
        super().__init__()
        self.patch_size = configs.patch_size if mode == 'Seasonal' else configs.trend_patch_size
        self.in_channel = configs.data_dim
        self.seq_len = configs.seq_len
        self.batch_size = configs.batch_size
        self.cycle_len = configs.cycle
        self.embed_dim = configs.embed_dim
        self.d_model = configs.embed_dim*configs.patch_size
        # self.d_model = 512
        # self.d_ff = 512
        self.d_ff = configs.d_ff*configs.patch_size
        # Embedding是非常重要的问题
        self.encoder_val_embedding = DI_embedding(configs.patch_size, configs.embed_dim, configs.dropout)
        self.encoder_segment = TS_Segment(configs.seq_len, configs.patch_size)
        self.encoder_pos_embed = nn.Parameter(torch.randn(1,
                                                            configs.embed_dim,
                                                            self.encoder_segment.seg_num,
                                                            configs.patch_size,
                                                            self.in_channel
                                                            ))
        self.encoder_pre_norm = nn.LayerNorm(self.d_model)
        self.encoder_norm = nn.LayerNorm(self.d_model)
        # Attention
        self.encoder = Encoder(
            [EncoderLayer(
                DozerAttentionLayer(
                    DozerAttention(configs.local_window, configs.stride, configs.rand_rate,
                                    configs.vary_len, self.encoder_segment.seg_num, self.in_channel,
                                    False,
                                    attention_dropout=configs.dropout,
                                    output_attention=configs.output_attention),
                    self.d_model,
                    configs.n_heads),
                d_model=self.d_model,
                d_ff=self.d_ff,
                dropout=configs.dropout,
                activation=configs.activation
            ) for l in range(configs.encoder_depth)
            ],
            norm_layer=None
        )

        self.fc_embed1 = nn.Linear(self.patch_size * configs.embed_dim, 512)
        self.fc_embed2 = nn.Linear(512, self.patch_size * configs.embed_dim)
        self.seg_num = self.seq_len // self.patch_size


        self.patch_embedding = nn.Parameter(torch.zeros(self.seg_num, self.d_model))
        self.phase_embedding = nn.Embedding(self.cycle_len, self.d_model)
        nn.init.xavier_normal_(self.phase_embedding.weight)
        self.joint_embedding = nn.Embedding(self.cycle_len, self.seg_num  * self.d_model)
        nn.init.xavier_normal_(self.joint_embedding.weight)
        nn.init.xavier_normal_(self.patch_embedding)


        # self.projection = nn.Linear(self.d_model, self.patch_size * configs.embed_dim)


    def forward(self, x_enc, x_label, x_mark_enc, phase):
        embeddings = self.encoder_val_embedding(rearrange(x_enc, 'b seq_len ts_d -> b 1 seq_len ts_d'))
        # Segment
        patches = self.encoder_segment(embeddings)
        identity = patches
        # Add pos
        patches = patches + self.encoder_pos_embed
        patches = rearrange(patches, 'b d_model seg_num seg_len ts_d -> (b ts_d) seg_num (seg_len d_model)')
        # Lrelu = nn.LeakyReLU(0.1)
        # relu = nn.ReLU()
        # T = nn.Tanh()
        # # # EFE embedding
        # patches = self.fc_embed2(T(self.fc_embed1(patches)))
        B, L, d_model = patches.shape

        # phase: (B,)
        phase_pre = phase.repeat_interleave(self.in_channel)  # (B*ts_d,) = (224,)
        phase_post = phase_pre.view(-1, 1).expand(-1, L)  # (224, 30) => (B, L)

        patch_emb = self.patch_embedding.expand(B, L, -1)  # Patch embedding
        phase_emb = self.phase_embedding(phase_post)  # Phase embedding
        joint_emb = self.joint_embedding(phase_pre).reshape(B, L, self.d_model)  # Joint Patch-Phase embedding

        patches =  patches + patch_emb + phase_emb + joint_emb

        # PreNorm
        patches = self.encoder_pre_norm(patches)
        #majority vote [batch, num , 1]
        patches_label = labels_to_segments(x_label, patch_size=self.patch_size, in_channel=self.in_channel)  # -> (224,30,1)

        encoder_output, attns = self.encoder(patches, patches_label)

        # PostNorm
        encoder_output = self.encoder_norm(encoder_output)
        # encoder_output = self.projection(encoder_output)

        # skip connection
        encoder_output = rearrange(encoder_output,
                                   '(b ts_d) seg_num (seg_len d_model) -> b d_model seg_num seg_len ts_d',
                                   seg_len=self.patch_size, ts_d=self.in_channel)
        # encoder_output = self.encoder_segment.concat(encoder_output)

        encoder_output = encoder_output + identity

        return encoder_output


class dozerformer_Decoder(nn.Module):
    def __init__(self, configs, mode):
        super().__init__()
        self.patch_size = configs.patch_size if mode == 'Seasonal' else configs.trend_patch_size
        self.in_channel = configs.data_dim

        d_model = configs.embed_dim*configs.patch_size
        d_ff = configs.d_ff*configs.patch_size
        pred_segs = ceil(configs.pred_len/configs.patch_size)
        self.decoder_val_embedding = DI_embedding(configs.patch_size, configs.decoder_embed_dim, configs.dropout)
        self.decoder_cross_segment = TS_Segment(configs.seq_len, configs.patch_size)
        self.decoder_segment = TS_Segment(configs.label_len + configs.pred_len, configs.patch_size)
        self.decoder_pos_embed = nn.Parameter(torch.randn(1,
                                                        configs.decoder_embed_dim,
                                                        self.decoder_segment.seg_num,
                                                        configs.patch_size,
                                                        self.in_channel
                                                        ))
        self.decoder_pre_norm = nn.LayerNorm(d_model)
        self.decoder_norm = nn.LayerNorm(d_model)
        # Attention
        self.decoder = Decoder(
            [
                DecoderLayer(
                    # 注意mask flag。True and False.
                    # Decoder中第一个attention
                    DozerAttentionLayer(
                        DozerAttention(configs.local_window, configs.stride, configs.rand_rate, configs.vary_len,
                                       pred_segs,
                                       False,
                                       attention_dropout=configs.dropout,
                                       output_attention=False),
                        d_model,
                        configs.n_heads),
                    # Decoder中第二个attention。Q是decoder的输入，KV是Encoder的输入。CrossAttention
                    DozerAttentionLayer(
                        DozerAttention(configs.local_window, configs.stride, configs.rand_rate, configs.vary_len,
                                       pred_segs,
                                       False,
                                       attention_dropout=configs.dropout,
                                       output_attention=False),
                        d_model,
                        configs.n_heads),
                    d_model=d_model,
                    d_ff=d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.decoder_depth)
            ],
            norm_layer=None,
            projection=None
        )


    def forward(self, x_dec, cross):
        '''
        x: the output of last decoder layer
        cross: the output of the corresponding encoder layer
        '''
        # Embedding
        embeddings = self.decoder_val_embedding(rearrange(x_dec, 'b seq_len ts_d -> b 1 seq_len ts_d'))
        # Segment
        patches = self.decoder_segment(embeddings)
        identity = patches
        # Add pos
        patches = patches + self.decoder_pos_embed

        cross = rearrange(cross, 'b d_model seg_num seg_len ts_d -> (b ts_d) seg_num (seg_len d_model)')
        patches = rearrange(patches, 'b d_model seg_num seg_len ts_d -> (b ts_d) seg_num (seg_len d_model)')
        # decoder
        patches = self.decoder_pre_norm(patches)
        decoder_output = self.decoder(patches, cross)
        decoder_output = self.decoder_norm(decoder_output)

        # skip connection
        decoder_output = rearrange(decoder_output,
                                    '(b ts_d) seg_num (seg_len d_model) -> b d_model seg_num seg_len ts_d',
                                    seg_len=self.patch_size, ts_d=self.in_channel)
        # decoder_output = self.decoder_segment.concat(decoder_output)
        decoder_output = decoder_output + identity
        return decoder_output



