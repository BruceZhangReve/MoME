import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import FullAttention, AttentionLayer
from .emb_layers import DataEmbedding_inverted
from .layers import RevIN
import numpy as np
import math
#Adapted from paper https://arxiv.org/abs/2310.06625


class MiTransformer(nn.Module):
    def __init__(self, n_vars, seq_len, d_model, dropout, topk, router_modulation, 
                 n_experts, n_heads, n_layers):
        super(MiTransformer, self).__init__()
        self.n_vars = n_vars
        self.modulation = True
        self.seq_len = seq_len
        self.d_model = d_model
        self.encoder = MiTransformer_backbone(n_vars, seq_len, d_model, dropout, topk, router_modulation, n_experts,
                                              n_heads, n_layers)

    def forward(self, x_seq, Ins_tk):       # [bs, seq_len, n_vars]

        x_seq = x_seq.permute(0,2,1) # [bs, n_vars, seq_len]

        x = self.encoder(x_seq, Ins_tk) #out shape:[bs, target_window, nvars]

        out = x.permute(0,2,1) #[B,C,d]

        return out #[B,C,d]; 
    

class MiTransformer_backbone(nn.Module):
    def __init__(self, n_vars, in_len, d_model, dropout, topk, router_modulation, n_experts, n_heads, n_layers):
        super(MiTransformer_backbone, self).__init__()

        self.output_attention = False # Whether or not return output attention
        self.factor = 1 #attn factor
        self.router_modulation = router_modulation
        self.seq_len = in_len
        self.output_attention = False # Whether or not return output attention
        self.use_norm = True # Use ravin_norm
        self.enc_embedding = DataEmbedding_inverted(in_len, d_model, dropout)
        self.revin_layer = RevIN(n_vars, affine=True, subtract_last=False)
        self.top_k = topk
        self.n_experts = n_experts
        self.pred_len = d_model

        self.projector = nn.Linear(d_model, d_model, bias=True)

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
                    router_modulation=self.router_modulation,
                ) for l in range(n_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.projector = nn.Linear(d_model, d_model, bias=True)


    def forecast(self, x_enc, Ins_tk):
        """
        #x_enc: [bs, n_vars, seq_len]
        """

        if self.use_norm:
            x_enc = self.revin_layer(x_enc, 'norm')

        # Embedding: [bs, seq_len, n_vars] -> [bs, n_vars, d_model]
        B, N, _ = x_enc.shape
        x_enc = x_enc.permute(0, 2, 1) 
        enc_out = self.enc_embedding(x_enc, None)

        # Multivariate Attention: [bs, n_vars, d_model] -> [bs, n_vars, d_model]
        enc_out, attns = self.encoder(enc_out, Ins_tk, attn_mask=None)
        
        # Projection head: [bs, n_vars, d_model] -> [bs, out_len, n_vars]
        enc_out = self.projector(enc_out).permute(0, 2, 1)

        if self.use_norm:
            enc_out = self.revin_layer(enc_out, 'denorm')

        return enc_out

    def forward(self, x_enc, Ins_tk):
        '''
        input:
            x_enc: [bs, seq_len, nvars]
        return: 
            x_enc: [bs, d_model, nvars]
        '''       

        out = self.forecast(x_enc, Ins_tk) # out shape: [bs, d_model, n_vars]

        return out


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, Ins_tk, attn_mask=None, tau=None, delta=None):
        attns = []
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x, Ins_tk, attn_mask=attn_mask, tau=tau, delta=delta)
            attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns
    

class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu", num_experts=5, topk=2, router_modulation=False):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 2 * d_model
        self.modulation = True
        self.hidden_dim = d_model
        self.router_modulation = router_modulation
        self.attention = attention
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

        self.top_k = topk
        self.n_experts = num_experts

        self.Gate = nn.Linear(d_model, self.n_experts, bias=False)
        self.experts = nn.ModuleList(
            [MoMeMLP(d_model, intermediate_size=d_ff) for _ in range(self.n_experts)]
        )

        # Externel Modulation
        if self.modulation == True:
            self.EiLM = nn.ModuleList(
                [EiLM(self.hidden_dim) for _ in range(self.n_experts)]
                )
            #self.router_modulator = nn.Linear(self.hidden_dim, self.n_experts, bias=False)
            if self.router_modulation:
                self.router_modulator = nn.Linear(self.hidden_dim, self.n_experts, bias=False)
        else:
            self.EiLM = None
            self.router_modulator = None

    def forward(self, x, Ins_tk, attn_mask=None, tau=None, delta=None):
        """
        x: [bs, n_vars, d_model]
        cls_emd: [n_clusters, d_model]
        prob_matrix: [n_vars, n_clusters]
        """
        B, C, d = x.shape

        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)
        y = x = self.norm1(x) #[B, N, d]

        tokens = x.reshape(-1, self.hidden_dim) # [B*N, d]
        router_logits = self.Gate(tokens) # [B*N, num_experts]
        if self.router_modulation:
            router_gamma = torch.mean(self.router_modulator(Ins_tk), dim=1)[0] # [B, N_i, hidden_dim] -> [B, N_i, num_experts] -> [B, num_experts] -> [num_experts]
            router_gamma.unsqueeze(0).expand_as(router_logits) # [B*C*P, num_experts]
            router_logits = router_gamma + router_logits # [B*C*P, num_experts]
        else:
            pass

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float32) #[B*C, E]
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1) #[B*C, E]

        final_hidden_states = torch.zeros(
            (B * C, self.hidden_dim), dtype=x.dtype, device=x.device
            )
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.n_experts).permute(2, 1, 0)

        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        if self.modulation == False:
            for expert_idx in expert_hit:
                expert_layer = self.experts[expert_idx]
                idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))

                current_state = tokens[None, top_x].reshape(-1, self.hidden_dim) #[N_i, d]
                current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None] #[N_i, d]

                final_hidden_states.index_add_(0, top_x, current_hidden_states.to(x.dtype))

            final_hidden_states = final_hidden_states.reshape(B, C, self.hidden_dim) #[B*C, d] -> [B, C, d]

            out =  final_hidden_states #[B, C, d]
        else:
            for expert_idx in expert_hit:
                expert_layer = self.experts[expert_idx]
                EiLM_layer = self.EiLM[expert_idx]

                idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))

                current_state = tokens[None, top_x].reshape(-1, self.hidden_dim) #[N_i, d]

                # Expert-Level Modulation #
                expert_output = expert_layer(current_state) #[N_i, d]
                expert_output = EiLM_layer(expert_output, Ins_tk) #[N_i, d], modulated
                 # Expert-Level Modulation #

                current_hidden_states = expert_output  * routing_weights[top_x, idx, None] #[N_i, d]

                final_hidden_states.index_add_(0, top_x, current_hidden_states.to(x.dtype))

            final_hidden_states = final_hidden_states.reshape(B, C, self.hidden_dim) #[B*C, d] -> [B, C, d]

            out = final_hidden_states #[B, C, d]

        y = out

        return self.norm2(x + y), attn

    

# Modified from transformers.models.mistral.modeling_mistral.MistralMLP 
class MoMeMLP(nn.Module):
    def __init__(self, hidden_dim, intermediate_size=None):
        super().__init__()
        self.hidden_size = hidden_dim
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = F.silu #ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class EiLM(nn.Module):
    """
    An expert-wise Linear Modulation Layer.
    seems B ==1 must hold...
    Input: Instruction tokens: [B, N_i, hidden_dim]; Tokens in main tasks received by an expert.
    Output: Modulated expert output for main task: [N_i, hidden_dim]
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.gamma_generator = nn.Linear(hidden_dim, 1, bias=False)
        self.beta_generator = nn.Linear(hidden_dim, hidden_dim, bias=False)
  
    def forward(self, x, Ins_tk):
        # x:[N_e, hidden_dim]
        B, _, _ = Ins_tk.shape;
        assert B==1, "batch_size must be 1, to enbale expert-level modulation!"

        gammas = torch.mean(self.gamma_generator(Ins_tk), dim=1)[0] # [B, N_i, hidden_dim] -> [B, 1] -> [1]
        betas = torch.mean(self.beta_generator(Ins_tk), dim=1)[0] # [B, N_i, hidden_dim] -> [B, hidden_dim] -> [hidden_dim]

        gammas = gammas.unsqueeze(0).expand_as(x)
        betas = betas.unsqueeze(0).expand_as(x)

        return (gammas * x) + betas
