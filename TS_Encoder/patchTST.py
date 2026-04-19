import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from math import sqrt, log

from .emb_layers import DataEmbedding


class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask
    
def l2norm(t):
    return F.normalize(t, dim = -1)

class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()
        
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, attn_bias):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            attn_bias
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn

class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1,
                 output_attention=False, configs=None,
                 attn_scale_init=20):
        super(FullAttention, self).__init__()
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
       
        self.scale = scale

    def forward(self, queries, keys, values, attn_mask, attn_bias):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)
        
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        if attn_bias is not None:
            attn_bias = attn_bias.permute(0, 3, 1, 2)
            A = self.dropout(torch.softmax(scores * scale + attn_bias, dim=-1))
        else:
            A = self.dropout(torch.softmax(scores * scale, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)

class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.BatchNorm1d(d_model)
        self.norm2 = nn.BatchNorm1d(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, attn_bias=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            attn_bias=attn_bias
        )
        x = x + self.dropout(new_x)
        #print(x.shape)
        y = x = self.norm1(x.permute(0, 2, 1)).permute(0, 2, 1)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        y = x + y
        y = self.norm2(y.permute(0, 2, 1)).permute(0, 2, 1)
        return y, attn

class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, attn_bias=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, attn_bias=attn_bias)
                attns.append(attn)

        if self.norm is not None:
            # x = self.norm(x)
            x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)

        return x, attns


class PatchTST(nn.Module):
    def __init__(self, args, in_len, out_len, patch_len):
        super(PatchTST, self).__init__()

        self.patch_size = patch_len
        self.stride = 4
        self.patch_num = (in_len - self.patch_size) // self.stride + 1

        self.seq_len = in_len
        self.pred_len = out_len
        self.d_model = args.hidden_dim
        self.n_heads = getattr(args, 'n_heads', 4)
        self.n_layers = getattr(args, 'n_layers', 3)
        self.dropout = getattr(args, 'dropout', 0.1)
        self.activation = getattr(args, 'activation', 'gelu')
        self.embed = 'fixed' 

        # a hard fix here
        if in_len == 312:
            self.freq = 't'  # minute-level
        elif in_len == 7: #Environment
            self.freq = 'w'
        else:
            self.freq = 'h'  # hour-level

        # Embedding
        self.enc_embedding = DataEmbedding(
            c_in=self.patch_size,
            d_model=self.d_model,
            embed_type=self.embed,
            freq=self.freq,
            dropout=self.dropout
        )

        # Encoder
        encoder_layers = []
        for _ in range(self.n_layers):
            encoder_layers.append(
                EncoderLayer(
                    attention=AttentionLayer(
                        FullAttention(
                            mask_flag=False,
                            factor=getattr(args, 'factor', 3),
                            attention_dropout=self.dropout,
                            output_attention=False  # hidden states 不需要 attn
                        ),
                        d_model=self.d_model,
                        n_heads=self.n_heads
                    ),
                    d_model=self.d_model,
                    d_ff=2 * self.d_model, 
                    dropout=self.dropout,
                    activation=self.activation
                )
            )
        
        # self.encoder = Encoder(encoder_layers, norm_layer=nn.LayerNorm(self.d_model))
        self.encoder = Encoder(encoder_layers, norm_layer=None)

        # Projection head for forecasting
        self.proj = nn.Linear(self.d_model * self.patch_num, out_len)

    def forward(self, x_enc, return_hidden=True):
        """
        Args:
            x_enc: [B, L, M] — input time series
            return_hidden: if True, return encoder hidden states; else return forecast

        Returns:
            if return_hidden:
                hidden: [B, M * P, d_model]
            else:
                pred: [B, M, pred_len]
        """
        B, L, M = x_enc.shape  # M = number of variates (channels)

        # Patching: [B, L, M] -> [B, M, L] -> [B, M, P, patch_len] -> [(B*M), P, patch_len]
        x_enc = rearrange(x_enc, 'b l m -> b m l')
        x_enc = x_enc.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        x_enc = rearrange(x_enc, 'b m n p -> (b m) n p')

        # Embedding + Encoder
        enc_out = self.enc_embedding(x_enc)  # [(B*M), P, d_model]
       # print(enc_out.shape)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)  # [(B*M), P, d_model]

        if return_hidden:
            # Reshape to [B, M * P, d_model]
            enc_out = enc_out.view(B, M, self.patch_num, self.d_model)
            enc_out = enc_out.flatten(1, 2)  # [B, M*P, d_model]
            return enc_out

        # Forecasting head
        enc_out = enc_out.reshape(B * M, -1)  # [(B*M), P * d_model]
        enc_out = self.proj(enc_out)          # [(B*M), pred_len]
        enc_out = rearrange(enc_out, '(b m) l -> b m l', b=B, m=M)  # [B, M, pred_len]
        return enc_out