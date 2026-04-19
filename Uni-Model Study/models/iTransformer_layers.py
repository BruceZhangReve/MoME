import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import GatingNetwork_v3


# This is a down sampling strategy used in informer, whereas it is uncessary for itransformer
class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=2,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))# [B, L, N] -> [B, N, L]???
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x




class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu", num_experts=5, topk=2, if_moe=False):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        activation_layer = nn.ReLU() if activation == "relu" else nn.GELU()
        self.if_moe = if_moe 
        self.topk = topk
        #for moe:
        if self.if_moe:
            self.Gating = GatingNetwork_v3(d_model, num_experts)
            self.experts_r = nn.ModuleList([
                    nn.Sequential(
                    nn.Conv1d(in_channels=2*d_model, out_channels=d_ff, kernel_size=1),
                    activation_layer,
                    nn.Dropout(dropout),
                    nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1),
                    nn.Dropout(dropout)
                ) for _ in range(num_experts)
            ])
        else:
            self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
            self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)

        #remark: Conv1d(kernel_size=1) is equivalent to Linear1(x) → ReLU → Linear2(x)

    def forward(self, x, cls_emd=None, prob_matrix=None,attn_mask=None, tau=None, delta=None):
        """
        x: [bs, n_vars, d_model]
        cls_emd: [n_clusters, d_model]
        prob_matrix: [n_vars, n_clusters]
        """
        bs = x.shape[0]
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        if not self.if_moe:
            gate_scores = None
            y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
            y = self.dropout(self.conv2(y).transpose(-1, 1))
            return self.norm2(x + y), attn, gate_scores
        else:
            bs, n_vars, d_model = y.shape; n_clusters, _ = cls_emd.shape
            gate_scores = self.Gating(x, cls_emd) #[n_vars, n_clusters], y:[bs, n_vars, d_model]

            y = y.unsqueeze(2).expand(-1, -1, n_clusters, -1)  # [bs, n_vars, K, d_model]
            z = cls_emd.view(1, 1, n_clusters, -1).expand_as(y)  # [bs, n_vars, K, d_model]
            yz_cat = torch.cat([y, z], dim=-1)  # [bs, n_vars, K, 2*d_model]
            # reshape for Conv1d: (K, bs * n_vars, 2*d_model) → (K, 2*d_model, bs * n_vars)
            yz_cat = yz_cat.permute(2, 0, 1, 3).reshape(n_clusters, -1, 2 * d_model).permute(0, 2, 1)

            #expert_r_outputs = torch.stack([expert_r(y.transpose(-1, -2)).transpose(-1, -2) for expert_r in self.experts_r], dim=2)  # [bs, n_vars, n_clusters, d_model]
            expert_r_outputs = torch.stack(
                [expert(yz_cat[k]).permute(1, 0) for k, expert in enumerate(self.experts_r)], dim=2
                )  # [bs * n_vars, K, d_model]
            expert_r_outputs = expert_r_outputs.view(bs, n_vars, n_clusters, d_model)  # [bs, n_vars, K, d_model]

            y_r = torch.einsum('nk,bnkd->bnd', gate_scores, expert_r_outputs)  # [bs, n_vars, d_model]
            return self.norm2(x + y_r), attn, gate_scores


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None, if_moe=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer
        self.if_moe = if_moe

    def forward(self, x, cls_emd=None, prob_matrix=None,attn_mask=None, topk=None, tau=None, delta=None):
        attns = []; gate_score_lis = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn, gate_scores = attn_layer(x, cls_emd=cls_emd, prob_matrix=prob_matrix, attn_mask=attn_mask, tau=tau, delta=delta)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn, gate_scores = attn_layer(x, cls_emd=cls_emd, prob_matrix=prob_matrix, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)
        
        if self.if_moe:
            gate_score_lis.append(gate_scores)

        return x, attns,  gate_score_lis


class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask,
            tau=tau, delta=None
        )[0])
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask,
            tau=tau, delta=delta
        )[0])

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x
