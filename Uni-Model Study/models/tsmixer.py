import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import *
from models.patch_layer import *



class TSMixerC(nn.Module):
    def __init__(self, args):
        super(TSMixerC, self).__init__()
        self.n_vars = args.batch_size if args.data in ["M4", "stock"] else args.data_dim
        self.in_len = args.in_len
        self.out_len = args.out_len
        self.n_experts = args.n_experts
        self.d_ff = args.d_ff
        self.d_model = args.d_model
        self.device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
        self.moe_mode = args.moe_mode
        self.topk = args.topk
        self.mixer_layers = []
        self.n_mixer = args.n_layers
        for i in range(self.n_mixer):
            self.mixer_layers.append(MixerLayer(self.n_experts, self.n_vars, self.in_len, self.device, self.d_ff, args.dropout, self.moe_mode, self.topk)) 
        self.mixer_layers = nn.ModuleList(self.mixer_layers)
        self.temp_proj = TemporalProj(self.in_len, self.out_len, self.moe_mode, self.n_experts, self.topk) #self.moe_mode potentially

        self.rev_in = RevIN(args.data_dim, affine=True, subtract_last=False)
        
        self.categorical = args.categorical

        self.rev_in = RevIN(args.data_dim, affine=True, subtract_last=False)
        
    def forward(self, x, if_update=False):
        gate_score_lis = [] #x is of shape [bs, in_seq, n_vars]

        if self.categorical:
            #print("FIRST",x.shape,x.device)
            x = self.cat_embed(x.transpose(1,2)) 
            x = x.transpose(1,2) # now x is [bs, in_seq, n_vars]
            x_num = x[:, :, :16]      # numerical features
            x_cat = x[:, :, 16:]      # categorical features
            #print("SECONDE",x_num.shape, x_num.device)
            x_num = self.rev_in(x_num, mode="norm")
            x = torch.cat([x_num, x_cat], dim=2)  #[bs, in_seq, n_vars]
        else:
            # Originally Commented out
            x = self.rev_in(x, mode = "norm")

        for i in range(self.n_mixer):
            #x = self.mixer_layers[i](x, self.cluster_prob) #[bs, in_seq, n_vars]
            x, gate_logits = self.mixer_layers[i](x) #[bs, in_seq, n_vars]
            if gate_logits is not None:
                gate_score_lis.append(gate_logits.mean(0)) #[n_vars, n_experts]

        x = self.temp_proj(x) # x: [bs, out_seq, n_vars]

        out = self.rev_in(x, mode="denorm") # x: [bs, out_seq, n_vars]
        
        return out, gate_score_lis, None  #[bs, out_len, n_vars] 
        



class MixerLayer(nn.Module):
    def __init__(self, n_experts, n_vars, seq_len, device, ff_dim, dropout, moe_mode, topk):
        super(MixerLayer, self).__init__()
        self.moe_mode = moe_mode
        #print(moe_mode)
        self.n_experts = n_experts
        self.topk = topk

        self.mlp_time = MLPTime(seq_len, dropout, moe_mode, n_experts, topk)

        self.mlp_feat = MLPFeat(n_vars, ff_dim, dropout)

    def forward(self, x):
        # x has shape (B, L, C) 
        res_x = x; gate_score = None

        x = x.transpose(1, 2) #[bs, in_seq, n_vars] -> [bs, n_vars, in_seq]

        x, gate_logits = self.mlp_time(x) #[bs, n_vars, in_seq]

        x = x.transpose(1, 2) + res_x #[bs, n_vars, in_seq] -> [bs, in_seq, n_vars]
        
        res_x = x
        x = self.mlp_feat(x) + res_x #[bs, in_seq, n_vars] -> [bs, in_seq, n_vars]

        return x, gate_logits


class MLPTime(nn.Module):
    def __init__(self, seq_len, dropout_rate, moe_mode, n_experts, topk):
        super(MLPTime, self).__init__()
        self.moe_mode = moe_mode
        self.n_experts = n_experts
        self.top_k = topk
        
        if self.moe_mode == "multi_expert":
            self.Gating = nn.Linear(seq_len, self.n_experts, bias=False)
            self.experts = nn.ModuleList([nn.Linear(seq_len, seq_len) for _ in range(self.n_experts)])
        else:
            self.fc = nn.Linear(seq_len, seq_len)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        gate_logits = None

        if self.moe_mode == "multi_expert":
            gate_logits = self.Gating(x) #[B, N, E]
            gate_logits = F.softmax(gate_logits, dim=-1)
            weights, selected_experts = torch.topk(gate_logits, self.top_k)  # [bs, n_vars, topk]

            output = torch.zeros_like(x) #[bs, n_vars, L]
            for i, expert in enumerate(self.experts):
                batch_idx, token_idx, kth = torch.where(selected_experts == i)
                #print(expert(x[batch_idx, token_idx]).shape) #[N_i, d_model]
                output[batch_idx, token_idx] += (
                    weights[batch_idx, token_idx, kth][:, None] * expert(x[batch_idx, token_idx])
                    )
            x = output
        else:
            x = self.fc(x)

        x = self.relu(x)
        x = self.dropout(x)
        return x, gate_logits

class MLPFeat(nn.Module):
    def __init__(self, C, ff_dim, dropout_rate=0.1):
        super(MLPFeat, self).__init__()
        self.fc1 = nn.Linear(C, ff_dim)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(ff_dim, C)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x


#Whether we only change projection head or, we change internal layers
class TemporalProj(nn.Module):
    def __init__(self, in_dim, out_dim, moe_mode, n_experts, topk):
        super(TemporalProj, self).__init__()
        self.moe_mode = moe_mode
        self.n_experts = n_experts
        self.top_k = topk
        self.pred_len = out_dim
        
        if self.moe_mode == "multi_expert":
            self.Gating = nn.Linear(in_dim, self.n_experts, bias=False)
            self.experts = nn.ModuleList([nn.Linear(in_dim, out_dim) for _ in range(self.n_experts)])
        else:
            self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        # x: [bs, seq_len, n_vars]
        B, in_len, n_vars = x.shape
        x = x.transpose(1, 2)

        if self.moe_mode == "multi_expert":
            gate_logits = self.Gating(x) #[B, N, E]
            gate_logits = F.softmax(gate_logits, dim=-1)
            weights, selected_experts = torch.topk(gate_logits, self.top_k)  # [bs, n_vars, topk]
            
            #output = torch.zeros_like(x) #[bs, n_vars, L]
            output = torch.zeros(B, n_vars, self.pred_len, device=x.device, dtype=x.dtype)
            for i, expert in enumerate(self.experts):
                batch_idx, token_idx, kth = torch.where(selected_experts == i)
                #print(expert(x[batch_idx, token_idx]).shape) #[N_i, d_model]
                output[batch_idx, token_idx] += (
                    weights[batch_idx, token_idx, kth][:, None] * expert(x[batch_idx, token_idx])
                    )
            x = output

        else:
            x = self.fc(x)

        x = x.transpose(1, 2)
        return x  # x: [bs, out_len, n_vars]
    

class RevNorm(nn.Module):
    #Reversible Instance Normalization in PyTorch.
    def __init__(self, num_features, axis=-2, eps=1e-5, affine=True):
        super().__init__()
        self.axis = axis
        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features)) 

    def forward(self, x, mode):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _get_statistics(self, x):
        self.mean = x.mean(dim=self.axis, keepdim=True).detach() # [Batch, Input Length, Channel]
        self.stdev = torch.sqrt(x.var(dim=self.axis, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = (x - self.mean) / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / self.affine_weight
        x = x * self.stdev
        x = x + self.mean
        return x

