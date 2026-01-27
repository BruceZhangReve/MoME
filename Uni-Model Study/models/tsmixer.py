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

        if self.categorical:
            x_num_ = x[:, :, :16]      # numerical features
            x_cat_ = x[:, :, 16:]      # categorical features
            x_num_ = self.rev_in(x_num_, mode="denorm")
            x = torch.cat([x_num_, x_cat_], dim=2)  #[bs, in_seq, n_vars]
            out = self.mixed_proj(x.transpose(1,2)) 
        else:
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


"""
class TSLinear(nn.Module):
    def __init__(self, L, T):
        super(TSLinear, self).__init__()
        self.fc = nn.Linear(L, T)

    def forward(self, x):
        return self.fc(x)

class TSMixerC(nn.Module):
    def __init__(self, args):
        super(TSMixerC, self).__init__()
        self.n_vars = args.batch_size if args.data in ["M4", "stock"] else args.data_dim
        self.in_len = args.in_len
        self.out_len = args.out_len
        self.n_cluster = args.n_cluster
        self.d_ff = args.d_ff
        self.d_model = args.d_model
        self.device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
        self.individual = args.individual
        self.if_moe = args.if_moe
        self.moe_head = args.moe_head
        self.mixer_layers = []
        self.n_mixer = args.n_layers
        for i in range(self.n_mixer):
            self.mixer_layers.append(MixerLayer(self.n_cluster, self.n_vars, self.in_len, self.device, self.individual, self.d_ff, args.dropout, args.if_moe)) 
        self.mixer_layers = nn.ModuleList(self.mixer_layers)
        self.temp_proj = TemporalProj(self.n_cluster, self.n_vars, self.in_len, self.out_len, self.device, self.individual, self.moe_head)

        if self.individual == "c":
            self.Cluster_assigner = Cluster_assigner(self.n_vars, self.n_cluster, self.in_len, self.d_ff, device=self.device)
            self.cluster_emb = self.Cluster_assigner.cluster_emb
        else:
            if not self.if_moe:
                self.cluster_emb = None #torch.empty(self.n_cluster, self.in_len).to(self.device)
            else:
                self.cluster_emb = nn.Parameter(torch.empty(self.n_cluster, self.in_len).to(self.device), requires_grad=True)
                nn.init.kaiming_uniform_(self.cluster_emb, a=math.sqrt(5))

        self.rev_in = RevIN(args.data_dim, affine=True, subtract_last=False)
        
        self.categorical = args.categorical
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
        if self.individual == "c":
            self.cluster_prob, cluster_emb = self.Cluster_assigner(x, self.cluster_emb)
        else:
            self.cluster_prob = None

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
            x, gate_score = self.mixer_layers[i](x, self.cluster_prob, self.cluster_emb) #[bs, in_seq, n_vars]

        x = self.temp_proj(x, self.cluster_prob) # x: [bs, out_seq, n_vars]

        # Originally Commented out
        if self.categorical:
            x_num_ = x[:, :, :16]      # numerical features
            x_cat_ = x[:, :, 16:]      # categorical features
            x_num_ = self.rev_in(x_num_, mode="denorm")
            x = torch.cat([x_num_, x_cat_], dim=2)  #[bs, in_seq, n_vars]
            #print("HERE", x.shape)
        else:
            x = self.rev_in(x, mode="denorm")

        if if_update and self.individual == "c":
            self.cluster_emb = nn.Parameter(cluster_emb, requires_grad=True)

        if self.if_moe:
            gate_score_lis.append(gate_score)
        
        if self.categorical:
            #This is for Telecom
            out = self.mixed_proj(x.transpose(1,2))
            return out, gate_score_lis, self.cluster_prob  #[bs, out_len, n_vars] 
        else:
            #This line is normally used
            return x, gate_score_lis, self.cluster_prob


# We're gonna make changes on this
class MLPTime(nn.Module):
    def __init__(self, n_cluster, n_vars, seq_len, device, individual, dropout_rate):
        super(MLPTime, self).__init__()
        if individual == "c":
            self.fc = Cluster_wise_linear(n_cluster, n_vars, seq_len, seq_len, device)
        else:
            self.fc = nn.Linear(seq_len, seq_len)
        self.individual = individual
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x, prob):
        if self.individual == "c":
            x = self.fc(x, prob)
        else:
            x = self.fc(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

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

class MixerLayer(nn.Module):
    def __init__(self, n_cluster, n_vars, seq_len, device, individual, ff_dim, dropout, if_moe):
        super(MixerLayer, self).__init__()
        self.if_moe = if_moe
        if self.if_moe:
            self.Gating = GatingNetwork_v3(seq_len, n_cluster)
            #self.experts_r = nn.ModuleList([
            #        MLPTime(n_cluster, n_vars, seq_len, device, individual, dropout)
            #        for _ in range(n_cluster)
            #])
            self.experts_r = nn.ModuleList([
                    MLPTime_modified(2*seq_len, dropout) for _ in range(n_cluster)
                    ])
        else:
            self.mlp_time = MLPTime(n_cluster, n_vars, seq_len, device, individual, dropout)

        self.mlp_feat = MLPFeat(n_vars, ff_dim, dropout)

    def batch_norm_2d(self, x):
        # x has shape (B, L, C) 
        #[bs, in_seq, n_vars]
        return (x - x.mean()) / x.std()
    
    def forward(self, x, prob, cls_emd):
        # x has shape (B, L, C) 
        res_x = x; gate_score = None

        #x = self.batch_norm_2d(x)
        x = x.transpose(1, 2) #[bs, in_seq, n_vars] -> [bs, n_vars, in_seq]

        if not self.if_moe:
            x = self.mlp_time(x, prob) #[bs, n_vars, in_seq], shared MLP across channels if non-ccm
        else:
            #print("HERE: ",x.shape, cls_emd.shape) #cls_emd: [n_clusters, in_seq]
            gate_score = self.Gating(x, cls_emd) #[n_vars, n_clusters]
            #expert_r_outputs = torch.stack([expert_r(x, prob) for expert_r in self.experts_r], dim=2)  # [bs, n_vars, n_clusters, in_seq]
            expert_r_outputs = torch.stack(
                [expert_r(torch.cat([x, cls_emd[k].expand_as(x)], dim=-1)) for k, expert_r in enumerate(self.experts_r)], dim=2) #[bs, nvars, n_clusters, in_seq]
            x = torch.einsum('nk,bnkd->bnd', gate_score, expert_r_outputs)  # [bs, n_vars, in_seq]

        x = x.transpose(1, 2) + res_x #[bs, n_vars, in_seq] -> [bs, in_seq, n_vars]
        res_x = x
        #x = self.batch_norm_2d(x)
        x = self.mlp_feat(x) + res_x #[bs, in_seq, n_vars] -> [bs, in_seq, n_vars]
        return x, gate_score

class TemporalProj(nn.Module):
    def __init__(self, n_cluster, n_vars, in_dim, out_dim, device, individual, moe_head):
        super(TemporalProj, self).__init__()
        if individual == "c":
            self.fc = Cluster_wise_linear(n_cluster, n_vars, in_dim, out_dim, device)
        else:
            if moe_head == 'ind':
                #TESTING, comment out if no good
                self.fc = Flatten_Head(n_vars, in_dim, out_dim)
            else:
                self.fc = nn.Linear(in_dim, out_dim)

        self.individual = individual
    def forward(self, x, prob):
        # x: [bs, seq_len, n_vars]
        # mask: [n_vars, n_cluster]
        x = x.transpose(1, 2)
        if self.individual == "c":
            x = self.fc(x, prob)
        else:
            x = self.fc(x)
        x = x.transpose(1, 2)
        return x  #[n_cluster,seq_len]

        

class MLPTime_modified(nn.Module):
    def __init__(self, seq_len, dropout_rate):
        super(MLPTime_modified, self).__init__()
        self.fc = nn.Linear(seq_len, seq_len//2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class Flatten_Head(nn.Module):
    def __init__(self, n_vars, d_model, target_window, head_dropout=0.2):
        super().__init__()
        self.n_vars = n_vars
        self.linears = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.flattens = nn.ModuleList()
        for i in range(self.n_vars):
            self.flattens.append(nn.Flatten(start_dim=-2))
            self.linears.append(nn.Linear(d_model, target_window))
            self.dropouts.append(nn.Dropout(head_dropout))
            
    def forward(self, x):        # x: [bs, nvars, d_model];
        x_out = []
        for i in range(self.n_vars):
            z = x[:,i,:]          # z: [bs, d_model]
            z = self.linears[i](z)                    # z: [bs, target_window]
            z = self.dropouts[i](z)
            x_out.append(z)
        x = torch.stack(x_out, dim=1)                 # x: [bs, nvars, target_window]
        return x            #x: [bs, nvars, target_window]

"""