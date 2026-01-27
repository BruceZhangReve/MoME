import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import *
from models.patch_layer import *


class MLP(nn.Module):
    def __init__(self, args):
        super(MLP, self).__init__()
        self.n_vars = args.batch_size if args.data in ["M4", "stock"] else args.data_dim
        self.seq_len = args.in_len
        self.pred_len = args.out_len
        self.moe_mode = args.moe_mode
        self.top_k = args.topk
        self.n_experts = args.n_experts
        self.categorical = args.categorical

        if self.moe_mode == "multi_expert":
            self.Gating = nn.Linear(self.seq_len, self.n_experts, bias=False)
            self.experts = nn.ModuleList([nn.Linear(self.seq_len,self.pred_len) for _ in range(self.n_experts)])
        else:
            self.Linear = nn.Linear(self.seq_len,self.pred_len)

        #This is for Telecom datasets, where categorical channels are involved
        if args.categorical:
            self.cat_embed = Cat_Embed(cat_indices=[16,17], cat_vocab_size=[3,3])
            self.mixed_proj = MixedProjector(cat_indices=[16,17], vocab_sizes=[3,3])
            # Add two learnable log-variance parameters for dynamic loss scaling
            self.log_sigma_num = nn.Parameter(torch.zeros(()))
            self.log_sigma_cat = nn.Parameter(torch.zeros(()))
            #self.rev_in = RevIN(args.data_dim-2, affine=True, subtract_last=False)
        #else:
            #self.rev_in = RevIN(args.data_dim, affine=True, subtract_last=False)
        

    def forward(self, x, if_update=False):
        #x: [bs, in_seq, n_vars]
        B, in_len, n_vars = x.shape
        gate_score_lis = [] #x is of shape [bs, in_seq, n_vars]

        if self.categorical:
            #print("FIRST",x.shape,x.device)
            x = self.cat_embed(x.transpose(1,2)) 
            x = x.transpose(1,2) # now x is [bs, in_seq, n_vars]
            x_num = x[:, :, :16] # numerical features
            x_cat = x[:, :, 16:] # categorical features
            #print("SECONDE",x_num.shape, x_num.device)
            #x_num = self.rev_in(x_num, mode="norm")
            x = torch.cat([x_num, x_cat], dim=2)  #[bs, in_seq, n_vars]
       # else:
            # Originally Commented out
            #x = self.rev_in(x, mode = "norm")

        
        x = x.permute(0,2,1)
       
        if self.moe_mode == "multi_expert":
            gate_logits = self.Gating(x) #[B, N, E]
            gate_logits = F.softmax(gate_logits, dim=-1)
            weights, selected_experts = torch.topk(gate_logits, self.top_k)  # [bs, n_vars, topk]
            results = torch.zeros(B, n_vars, self.pred_len, device=x.device, dtype=x.dtype)
            for i, expert in enumerate(self.experts):
                batch_idx, token_idx, kth = torch.where(selected_experts == i)
                results[batch_idx, token_idx] += (
                    weights[batch_idx, token_idx, kth][:, None] * expert(x[batch_idx, token_idx])
                    )
        else:
            results = self.Linear(x)

        x = results.transpose(1, 2) # x: [bs, out_seq, n_vars]

        if self.categorical:
            x_num_ = x[:, :, :16]      # numerical features
            x_cat_ = x[:, :, 16:]      # categorical features
            #x_num_ = self.rev_in(x_num_, mode="denorm")
            x = torch.cat([x_num_, x_cat_], dim=2)  #[bs, in_seq, n_vars]
            x = self.mixed_proj(x.transpose(1,2)) 
        #else:
            #x = self.rev_in(x, mode="denorm") # x: [bs, out_seq, n_vars]

        if self.moe_mode == "No":
            return x, None, None
        elif self.moe_mode == "multi_expert":
            return x, [gate_logits.mean(dim=0)], None
        else:
            raise ValueError("Unknown moe_mode")
    
"""
class MLP(nn.Module):
    def __init__(self, args):
        super(MLP, self).__init__()
        self.n_vars = args.batch_size if args.data in ["M4", "stock"] else args.data_dim
        self.in_len = args.in_len
        self.out_len = args.out_len
        self.moe_mode = args.moe_mode
        self.topk = args.topk
        self.n_experts = args.n_experts
        self.d_model = args.d_model
        self.d_ff = args.d_ff
        self.device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
        self.mlp_layers = []
        self.n_mlp = args.n_layers

        self.categorical = args.categorical

        self.emb = nn.Linear(self.in_len, self.d_model)

        for i in range(self.n_mlp):
            self.mlp_layers.append(MLPLayer(self.d_model, self.d_ff, args.dropout, self.moe_mode, self.n_experts, self.topk)) 

        self.mlp_layers = nn.ModuleList(self.mlp_layers)
        #self.temp_proj = TemporalProj(self.d_model, self.out_len)
        self.temp_proj = TemporalProj(self.d_model, self.out_len, self.moe_mode, self.n_experts, self.topk) #self.moe_mode potentially


        #This is for Telecom datasets, where categorical channels are involved
        if args.categorical:
            self.cat_embed = Cat_Embed(cat_indices=[16,17], cat_vocab_size=[3,3])
            self.mixed_proj = MixedProjector(cat_indices=[16,17], vocab_sizes=[3,3])
            # Add two learnable log-variance parameters for dynamic loss scaling
            self.log_sigma_num = nn.Parameter(torch.zeros(()))
            self.log_sigma_cat = nn.Parameter(torch.zeros(()))
            self.rev_in = RevIN(args.data_dim-2, affine=True, subtract_last=False)
        else:
            self.rev_in = RevIN(args.data_dim, affine=True, subtract_last=False)
        

    def forward(self, x, if_update=False):
        gate_score_lis = [] #x is of shape [bs, in_seq, n_vars]
        #x: [bs, in_seq, n_vars]

        if self.categorical:
            #print("FIRST",x.shape,x.device)
            x = self.cat_embed(x.transpose(1,2)) 
            x = x.transpose(1,2) # now x is [bs, in_seq, n_vars]
            x_num = x[:, :, :16] # numerical features
            x_cat = x[:, :, 16:] # categorical features
            #print("SECONDE",x_num.shape, x_num.device)
            x_num = self.rev_in(x_num, mode="norm")
            x = torch.cat([x_num, x_cat], dim=2)  #[bs, in_seq, n_vars]
        else:
            # Originally Commented out
            x = self.rev_in(x, mode = "norm")

        x = self.emb(x.permute(0,2,1)) #[bs, in_seq, n_vars] -> [bs, n_vars, d_model]

        for i in range(self.n_mlp):
            x, gate_logits =  self.mlp_layers[i](x) #[bs, n_vars, d_model]
            #x, gate_score = self.mlp_layers[i](x) #[bs, n_vars, d_model]
            if gate_logits is not None:
                gate_score_lis.append(gate_logits.mean(0)) #[n_vars, n_experts]

        x = self.temp_proj(x) # x: [bs, n_vars, out_seq]

        x = x.transpose(1, 2) # x: [bs, out_seq, n_vars]

        if self.categorical:
            x_num_ = x[:, :, :16]      # numerical features
            x_cat_ = x[:, :, 16:]      # categorical features
            x_num_ = self.rev_in(x_num_, mode="denorm")
            x = torch.cat([x_num_, x_cat_], dim=2)  #[bs, in_seq, n_vars]
            x = self.mixed_proj(x.transpose(1,2)) 
        else:
            x = self.rev_in(x, mode="denorm") # x: [bs, out_seq, n_vars]

        return x, gate_score_lis, None


# We're gonna make changes on this
class MLPTime(nn.Module):
    def __init__(self, d_model, d_ff, dropout_rate):
        super(MLPTime, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        #x: [bs, n_vars, d_model]
        x = self.dropout1(self.relu(self.fc1(x)))
        x = self.dropout2(self.fc2(x))
        return x


#class MLPTime(nn.Module):
    #def __init__(self, seq_len, dropout_rate, moe_mode, n_experts, topk):
        #super(MLPTime, self).__init__()
        #self.moe_mode = moe_mode
        #self.n_experts = n_experts
        #self.top_k = topk
        
        #if self.moe_mode == "multi_expert":
        #    self.Gating = nn.Linear(seq_len, self.n_experts, bias=False)
        #    self.experts = nn.ModuleList([nn.Linear(seq_len, seq_len) for _ in range(self.n_experts)])
        #else:
        #    self.fc = nn.Linear(seq_len, seq_len)
        #self.relu = nn.ReLU()
        #self.dropout = nn.Dropout(p=dropout_rate)

    #def forward(self, x):
        #gate_logits = None

        #if self.moe_mode == "multi_expert":
        #    gate_logits = self.Gating(x) #[B, N, E]
        #    gate_logits = F.softmax(gate_logits, dim=-1)
        #    weights, selected_experts = torch.topk(gate_logits, self.top_k)  # [bs, n_vars, topk]

        #    output = torch.zeros_like(x) #[bs, n_vars, L]
        #    for i, expert in enumerate(self.experts):
        #        batch_idx, token_idx, kth = torch.where(selected_experts == i)
                #print(expert(x[batch_idx, token_idx]).shape) #[N_i, d_model]
        #        output[batch_idx, token_idx] += (
        #            weights[batch_idx, token_idx, kth][:, None] * expert(x[batch_idx, token_idx])
        #            )
        #    x = output
        #else:
        #    x = self.fc(x)

        #x = self.relu(x)
        #x = self.dropout(x)
        #return x, gate_logits

class MLPLayer(nn.Module):
    def __init__(self, d_model, d_ff, dropout, moe_mode, n_experts, topk):
        super(MLPLayer, self).__init__()
        self.moe_mode = moe_mode
        self.n_experts = n_experts
        self.top_k = topk
        #self.mlp_time = MLPTime(d_model, dropout, moe_mode, n_experts, topk)
        self.mlp_time = MLPTime(d_model, d_ff, dropout)
        #if self.moe_mode == "multi_expert":
        #    self.Gating = nn.Linear(d_model, self.n_experts, bias=False)
        #    self.experts = nn.ModuleList([ MLPTime(d_model, d_ff, dropout) for _ in range(self.n_experts)])
        #else:
        #    self.fc =  MLPTime(d_model, d_ff, dropout)

    
    def forward(self, x):
        # x: [bs, n_vars, d_model]
        res_x = x; gate_logits = None
        #x, gate_logits = self.mlp_time(x) #[bs, n_vars, in_seq]
        x = self.mlp_time(x) #[bs, n_vars, in_seq]
        x = x + res_x #[bs, n_vars, d_model] 
        return x, None


        #if self.moe_mode == "multi_expert":
        #    gate_logits = self.Gating(x) #[B, N, E]
        #    gate_logits = F.softmax(gate_logits, dim=-1)
        #    weights, selected_experts = torch.topk(gate_logits, self.top_k)  # [bs, n_vars, topk]

        #    output = torch.zeros_like(x) #[bs, n_vars, L]
        #    for i, expert in enumerate(self.experts):
        #        batch_idx, token_idx, kth = torch.where(selected_experts == i)
                #print(expert(x[batch_idx, token_idx]).shape) #[N_i, d_model]
        #        output[batch_idx, token_idx] += (
        #            weights[batch_idx, token_idx, kth][:, None] * expert(x[batch_idx, token_idx])
        #            )
        #    x = output
        #else:
        #    x = self.fc(x)

        #x = x + res_x #[bs, n_vars, d_model] 
        #return x, gate_logits

#class TemporalProj(nn.Module):
#    def __init__(self, in_dim, out_dim):
#        super(TemporalProj, self).__init__()
#        self.fc = nn.Linear(in_dim, out_dim)

#    def forward(self, x):
        # x: [bs, n_vars, d_model]
#        x = self.fc(x)
#        return x  #[bs, n_vars, pred_len]

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
        # x: [bs, n_vars, d_model]
        B, n_vars, d_model = x.shape

        if self.moe_mode == "multi_expert":
            gate_logits = self.Gating(x) #[B, N, E]
            gate_logits = F.softmax(gate_logits, dim=-1)
            weights, selected_experts = torch.topk(gate_logits, self.top_k)  # [bs, n_vars, topk]
            
            output = torch.zeros(B, n_vars, self.pred_len, device=x.device, dtype=x.dtype)
            for i, expert in enumerate(self.experts):
                batch_idx, token_idx, kth = torch.where(selected_experts == i)
                output[batch_idx, token_idx] += (
                   weights[batch_idx, token_idx, kth][:, None] * expert(x[batch_idx, token_idx])
                    )
            x = output
        else:
            x = self.fc(x)

        return x  # x: [bs, out_len, n_vars]
    

        
    

class RevNorm(nn.Module):
    #Reversible Instance Normalization in PyTorch, assue teporal dimension is indiced on -2
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
"""