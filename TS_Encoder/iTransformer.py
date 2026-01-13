import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import FullAttention, AttentionLayer
from .emb_layers import DataEmbedding_inverted
from .layers import RevIN
import numpy as np
import math
#Adapted from paper https://arxiv.org/abs/2310.06625


class iTransformer(nn.Module):
    def __init__(self, n_vars, seq_len, d_model, dropout, topk, moe_mode, 
                 n_experts, n_heads, n_layers):
        super(iTransformer, self).__init__()
        self.n_vars = n_vars
        self.seq_len = seq_len
        self.d_model = d_model
        #self.device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
        self.encoder = iTransformer_backbone(n_vars, seq_len, d_model, dropout, topk, moe_mode, n_experts,
                                             n_heads, n_layers)


    def forward(self, x_seq, if_update=False):       # [bs, seq_len, n_vars]

        x_seq = x_seq.permute(0,2,1) # [bs, n_vars, seq_len]

        x, gate_score_lis, expert_embedding_lis = self.encoder(x_seq) #out shape:[bs, target_window, nvars]
        #note that the 2nd and 3rd output would be meaningless, if it's single-channel time-series

        out = x.permute(0,2,1) #[B,C,d]

        return out #[B,C,d]; 
    

class iTransformer_backbone(nn.Module):
    def __init__(self, n_vars, in_len, d_model, dropout, topk, moe_mode, n_experts, n_heads, n_layers):
        super(iTransformer_backbone, self).__init__()

        self.output_attention = False # Whether or not return output attention
        self.factor = 1 #attn factor

        self.seq_len = in_len
        self.output_attention = False # Whether or not return output attention
        self.use_norm = True # Use ravin_norm
        self.enc_embedding = DataEmbedding_inverted(in_len, d_model, dropout)
        self.revin_layer = RevIN(n_vars, affine=True, subtract_last=False)
        self.top_k = topk
        self.moe_mode = moe_mode
        self.n_experts = n_experts
        self.pred_len = d_model
        print(self.moe_mode)
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, self.factor, attention_dropout=dropout,
                                      output_attention=self.output_attention), d_model, n_heads),
                    d_model,
                    n_heads,
                    dropout=dropout,
                    activation="gelu",
                    num_experts=self.n_experts,
                    topk=self.top_k,
                    moe_mode=self.moe_mode
                ) for l in range(n_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model), moe_mode=moe_mode
        )

        if self.moe_mode == 'multi_expert':
            self.Gating = nn.Linear(d_model, self.n_experts, bias=False)
            self.experts = nn.ModuleList([nn.Linear(d_model, d_model, bias=True) for _ in range(self.n_experts)])
        else:
            self.projector = nn.Linear(d_model, d_model, bias=True)


    def forecast(self, x_enc):
        """
        #x_enc: [bs, n_vars, seq_len]
        """

        if self.use_norm:
            x_enc = self.revin_layer(x_enc, 'norm')

        #B, _, N = x_enc.shape

        # Embedding: [bs, seq_len, n_vars] -> [bs, n_vars, d_model]
        B, N, _ = x_enc.shape
        x_enc = x_enc.permute(0, 2, 1) #
        #print(x_enc.shape)
        enc_out = self.enc_embedding(x_enc, None)

        # Multivariate Attention: [bs, n_vars, d_model] -> [bs, n_vars, d_model]
        enc_out, attns, gate_score_lis, expert_embedding_lis = self.encoder(enc_out, attn_mask=None)
        
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

        return enc_out, attns, gate_score_lis, expert_embedding_lis

    def forward(self, x_enc):
        '''
        input:
            x_enc: [bs, seq_len, nvars]
        return: 
            x_enc: [bs, d_model, nvars]
        '''       
        #x_enc = x_enc.permute(0,2,1) # [bs, seq_len, nvars]

        out, attns, gate_score_lis, expert_embedding_lis = self.forecast(x_enc) # out shape: [bs, d_model, n_vars]

        return out, gate_score_lis, expert_embedding_lis #[B,S,N] 


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None, moe_mode='single_expert'):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer
        self.moe_mode = moe_mode

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        attns = []; gate_score_lis = []; expert_embeddings_lis = []
        if self.conv_layers is not None:
            # Since it's always None we didn't modify this part
            #for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
            #    delta = delta if i == 0 else None
            #    x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
            #    x = conv_layer(x)
            #    attns.append(attn)
            #x, attn, gate_scores, expert_embeddings = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
            #attns.append(attn)
            pass
        else:
            for attn_layer in self.attn_layers:
                x, attn, gate_scores, expert_embeddings = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)
                if gate_scores is not None:
                    #print("HERE")
                    gate_score_lis.append(gate_scores.mean(0))
                if expert_embeddings is not None:
                    expert_embeddings_lis.append(expert_embeddings)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns, gate_score_lis, expert_embeddings_lis
    

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

            # We wish to use mean pooling to get expert embedding
            expert_embeddings = []

            for i, expert in enumerate(self.experts):
                batch_idx, token_idx, kth = torch.where(selected_experts == i)
                
                if len(batch_idx) > 0:  # only compute when the experts is selected
                    expert_input = y[batch_idx, token_idx]
                    expert_output = expert(expert_input)
                    output[batch_idx, token_idx] += (
                        weights[batch_idx, token_idx, kth][:, None] * expert_output
                    )
            
                    expert_embedding = torch.mean(expert_input, dim=0)  # [N_i, d] -> [d]
                    expert_embeddings.append(expert_embedding)
                else:
                    # If expert not selected, we use zero-vector as its embedding
                    expert_embeddings.append(torch.zeros_like(y[0, 0]))
            
            expert_embeddings = torch.stack(expert_embeddings, dim=0) #[E, d]
            y = output

            return self.norm2(x + y), attn, gate_logits, expert_embeddings
        
        else:
            gate_logits = None; expert_embeddings = None
            y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
            y = self.dropout(self.conv2(y).transpose(-1, 1))

            return self.norm2(x + y), attn, gate_logits, expert_embeddings
    
