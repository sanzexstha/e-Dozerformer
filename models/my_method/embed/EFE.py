import torch
import torch.nn as nn

class EFE(nn.Module):
    def __init__(self, lag_s, embed_dim):
        super().__init__()
        self.dense = nn.Linear(lag_s + 2, embed_dim)
        self.activation = nn.Tanh()
        self.lag_s = lag_s

    def forward(self, stream, rain):
        # stream: (b, seq_len, 1)
        # rain:   (b, seq_len, 1)
        b, t, _ = rain.shape

        # pad and unfold — no Python loop
        rain_padded = torch.nn.functional.pad(rain.squeeze(-1), (self.lag_s, 0), value=0)  # (b, seq_len+s)
        lagged_rain = rain_padded.unfold(1, self.lag_s + 1, 1)  # (b, seq_len, s+1)

        efe_input = torch.cat([stream, lagged_rain], dim=-1)  # (b, seq_len, s+2)
        return self.activation(self.dense(efe_input))  # (b, seq_len, embed_dim)