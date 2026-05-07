import torch
import torch.nn as nn
import torch.nn.functional as F
from models.attention import FullAttention, AttentionLayer
from models.emb_layers import DataEmbedding_inverted
from models.layers import RevIN, Cat_Embed, MixedProjector
import numpy as np
import math
#Adapted from paper https://arxiv.org/abs/2310.06625

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.n_cluster = args.n_cluster
        self.n_vars = args.batch_size if args.data in ["M4", "stock"] else args.data_dim
        self.individual = args.individual
        self.seq_len = args.in_len
        self.d_model = args.d_model
        self.device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
        self.encoder = iTransformer_backbone(args)
        self.categorical = args.categorical


    def forward(self, x_seq, if_update=False):       # [bs, seq_len, n_vars]

        x_seq = x_seq.permute(0,2,1) # [bs, n_vars, seq_len]

        x = x_seq

        x, gate_score_lis = self.encoder(x) #out shape:[bs, target_window, nvars]

        out = x

        return out, gate_score_lis, None #[:, :self.out_len, :]   # [bs, out_len, n_vars]
    

class iTransformer_backbone(nn.Module):
    def __init__(self, args):
        super(iTransformer_backbone, self).__init__()
        self.seq_len = args.in_len
        self.pred_len = args.out_len
        self.output_attention = args.output_attention
        self.use_norm = args.ravin_norm
        self.enc_embedding = DataEmbedding_inverted(args.in_len, args.d_model, args.dropout)
        self.class_strategy = args.class_strategy
        self.revin_layer = RevIN(args.data_dim, affine=True, subtract_last=False)
        self.top_k = args.topk
        self.moe_mode = args.moe_mode
        self.n_experts = args.n_experts
        self.device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, args.factor, attention_dropout=args.dropout,
                                      output_attention=args.output_attention), args.d_model, args.n_heads),
                    args.d_model,
                    args.n_heads,
                    dropout=args.dropout,
                    activation=args.activation,
                    num_experts=self.n_experts,
                    topk=args.topk,
                    moe_mode=self.moe_mode
                ) for l in range(args.n_layers)
            ],
            norm_layer=torch.nn.LayerNorm(args.d_model), moe_mode =args.moe_mode
        )

        if self.moe_mode == 'multi_expert':
            self.Gating = nn.Linear(args.d_model, self.n_experts, bias=False)
            self.experts = nn.ModuleList([nn.Linear(args.d_model, args.out_len, bias=True) for _ in range(self.n_experts)])
        else:
            self.projector = nn.Linear(args.d_model, args.out_len, bias=True)


    def forecast(self, x_enc):
        """
        #x_enc: [bs, seq_len, n_vars]
        """
        gate_score_lis = None

        if self.use_norm:
            x_enc = self.revin_layer(x_enc, 'norm')

        B, _, N = x_enc.shape

        # Embedding: [bs, seq_len, n_vars] -> [bs, n_vars, d_model]
        enc_out = self.enc_embedding(x_enc, None)

        # Multivariate Attention: [bs, n_vars, d_model] -> [bs, n_vars, d_model]
        enc_out, attns, gate_score_lis = self.encoder(enc_out, attn_mask=None)
        
        # Projection head: [bs, n_vars, d_model] -> [bs, out_len, n_vars]
        if self.moe_mode == "multi_expert":
            gate_logits = self.Gating(enc_out) #[B, N, E]
            gate_logits = F.softmax(gate_logits, dim=-1)
            weights, selected_experts = torch.topk(gate_logits, self.top_k)  # [bs, n_vars, topk]
            output = torch.zeros(B, N, self.pred_len, device=enc_out.device, dtype=enc_out.dtype)
            for i, expert in enumerate(self.experts):
                batch_idx, token_idx, kth = torch.where(selected_experts == i)
                #print(expert(x[batch_idx, token_idx]).shape) #[N_i, d_model]
                output[batch_idx, token_idx] += (
                    weights[batch_idx, token_idx, kth][:, None] * expert(enc_out[batch_idx, token_idx])
                    )
            enc_out = output.permute(0, 2, 1)
        else:
            enc_out = self.projector(enc_out).permute(0, 2, 1)

        if self.use_norm:
            enc_out = self.revin_layer(enc_out, 'denorm')

        return enc_out, attns, gate_score_lis

    def forward(self, x_enc):
        '''
        input:
            x_enc: [bs, nvars, seq_len]
        return: 
            x_enc: [bs, nvars, target_window]
        '''       
        x_enc = x_enc.permute(0,2,1) # [bs, seq_len, nvars]
        out, attns, gate_score_lis = self.forecast(x_enc) # out shape: [bs, out_len, n_vars]

        return out, gate_score_lis # [B,S,N] , [B, n_cluster, d_model]





class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None, moe_mode='single_expert'):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer
        self.moe_mode = moe_mode

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        attns = []; gate_score_lis = []
        if self.conv_layers is not None:
            # Since it's always None we didn't modify this part
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn, gate_scores = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn, gate_scores = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)
                if gate_scores is not None:
                    #print("HERE")
                    gate_score_lis.append(gate_scores.mean(0))

        if self.norm is not None:
            x = self.norm(x)

        return x, attns, gate_score_lis
    

class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu", num_experts=5, topk=2, moe_mode='single_expert'):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        activation_layer = nn.ReLU() if activation == "relu" else nn.GELU()

        self.moe_mode = moe_mode
        self.top_k = topk
        self.n_experts = num_experts
        #for moe:
        if self.moe_mode == "multi_expert":
            self.Gating = nn.Linear(d_model, self.n_experts, bias=False)
            self.experts = nn.ModuleList([
                    nn.Sequential(
                    nn.Linear(d_model, d_ff),
                    activation_layer,
                    nn.Dropout(dropout),
                    nn.Linear(d_ff, d_model),
                    nn.Dropout(dropout)
                ) for _ in range(num_experts)
            ])
        else:
            self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
            self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)

        #remark: Conv1d(kernel_size=1) is equivalent to Linear1(x) → ReLU → Linear2(x)

    def forward(self, x, attn_mask=None, tau=None, delta=None):
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

        if self.moe_mode == "multi_expert":
            gate_logits = self.Gating(y) #[B, N, E]
            gate_logits = F.softmax(gate_logits, dim=-1)
            weights, selected_experts = torch.topk(gate_logits, self.top_k)  # [bs, n_vars, topk]
            output = torch.zeros_like(y) #[bs, n_vars, d_model]
            for i, expert in enumerate(self.experts):
                batch_idx, token_idx, kth = torch.where(selected_experts == i)
                #print(expert(x[batch_idx, token_idx]).shape) #[N_i, d_model]
                output[batch_idx, token_idx] += (
                    weights[batch_idx, token_idx, kth][:, None] * expert(y[batch_idx, token_idx])
                )
            y = output

            return self.norm2(x + y), attn, gate_logits
        else:
            gate_logits = None
            y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
            y = self.dropout(self.conv2(y).transpose(-1, 1))

            return self.norm2(x + y), attn, gate_logits
    
