import torch
import torch.nn as nn
import torch.nn.functional as f
from models.layers import *
from models.attention import *
from models.patch_layer import *


class PatchTSTC(nn.Module):
    def __init__(self, args, baseline = False, if_decomposition=False):
        super(PatchTSTC, self).__init__()
        self.n_vars = args.batch_size if args.data in ["M4", "stock"] else args.data_dim
        self.in_len = args.in_len
        self.out_len = args.out_len
        self.patch_len = args.patch_len
        self.n_cluster = args.n_cluster
        self.d_model = args.d_model
        self.d_ff = args.d_ff
        self.individual = args.individual
        self.if_moe = args.if_moe
        self.moe_mode = args.moe_mode
        self.baseline = baseline
        self.device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
        if if_decomposition:
            self.decomp_module = series_decomp(kernel_size=25)
            self.encoder_trend = Patch_backbone(args, device=self.device)
            self.encoder_res = Patch_backbone(args, device=self.device)
        if self.individual == "c":
            self.Cluster_assigner = Cluster_assigner(self.n_vars, self.n_cluster, self.in_len, self.d_model, device=self.device)
            self.cluster_emb = self.Cluster_assigner.cluster_emb
        elif self.individual == "i":
            self.cluster_emb = torch.empty(self.n_cluster, self.d_model).to(self.device)

        if self.moe_mode == "multi_experts":
            self.experts = nn.ModuleList([
                    nn.Linear(self.d_model, self.out_len) for _ in range(self.n_experts)
                    ])
            
        self.encoder = Patch_backbone(args, device=self.device)
        self.decomposition = if_decomposition
        self.cluster_prob = None

        self.categorical = args.categorical
        if args.categorical:
            self.cat_embed = Cat_Embed(cat_indices=[16,17], cat_vocab_size=[3,3])
            self.mixed_proj = MixedProjector(cat_indices=[16,17], vocab_sizes=[3,3])
            # Add two learnable log-variance parameters for dynamic loss scaling
            self.log_sigma_num = nn.Parameter(torch.zeros(()))
            self.log_sigma_cat = nn.Parameter(torch.zeros(()))

            self.rev_in = RevIN(args.data_dim-2, affine=True, subtract_last=False)
        else:
            pass
            #self.rev_in = RevIN(args.data_dim, affine=True, subtract_last=False)
        
    def forward(self, x_seq, if_update=False):       #[bs, seq_len, n_vars]
        if (self.baseline):
            base = x_seq.mean(dim = 1, keepdim = True)
        else:
            base = 0
        if self.individual == "c":
            self.cluster_prob, cluster_emb_1 = self.Cluster_assigner(x_seq, self.cluster_emb)      #[n_vars, n_cluster]
        if self.decomposition:
            res_init, trend_init = self.decomp_module(x_seq)
            res_init, trend_init = res_init.permute(0,2,1), trend_init.permute(0,2,1)  # x: [Batch, Channel, Input length]
            res, cls_emb_res = self.encoder_res(res_init, self.cluster_emb, self.cluster_prob)
            trend, cls_emd_trend = self.encoder_trend(trend_init,  self.cluster_emb, self.cluster_prob)
            out = res + trend
            cluster_emb = (cls_emb_res + cls_emd_trend)/2
            if if_update and self.individual == "c":
                self.cluster_emb = nn.Parameter(cluster_emb_1, requires_grad=True)
            out = out.permute(0,2,1)    # x: [Batch, Input length, Channel]
        else:
            #x_seq is of shape [bs, in_len, n_vars]
            x_seq = x_seq.permute(0,2,1)

            if self.categorical:
                x = self.cat_embed(x_seq)  # x: [bs, n_vars, in_len], with categorical channels embedded
                x = x.transpose(1,2) # now x is [bs, in_seq, n_vars]
                x_num = x[:, :, :16]      # numerical features
                x_cat = x[:, :, 16:]      # categorical features
                x_num = self.rev_in(x_num, mode="norm")
                x = torch.cat([x_num, x_cat], dim=2)  #[bs, in_seq, n_vars]
                x = x.transpose(1,2)
            else:
                x = x_seq

            out, _, gate_score_lis = self.encoder(x, self.cluster_emb, self.cluster_prob) # out: [bs, n_vars, out_len]

            #print("if_update: ", if_update)
            if if_update and self.individual == "c":
                self.cluster_emb = nn.Parameter(cluster_emb_1, requires_grad=True)

            if self.categorical:
                #This is for Telecom
                out = out.transpose(1,2) # [bs, n_vars, out_len] -> [bs, out_len, n_vars]
                x_num_ = out[:, :, :16]      # numerical features
                x_cat_ = out[:, :, 16:]      # categorical features
                x_num_ = self.rev_in(x_num_, mode="denorm")
                out = torch.cat([x_num_, x_cat_], dim=2)  #[bs, in_seq, n_vars]

                out = self.mixed_proj(out.transpose(1,2))
                return out, gate_score_lis, self.cluster_prob  #[bs, out_len, n_vars] 
            else:
                #This line is normally used
                out = out.permute (0,2,1)
                return base + out[:, :self.out_len, :], gate_score_lis, self.cluster_prob  #[bs, out_len, n_vars] 
            

"""
lass PatchTSTC(nn.Module):
    def __init__(self, args, baseline = False, if_decomposition=False):
        super(PatchTSTC, self).__init__()
        self.n_vars = args.batch_size if args.data in ["M4", "stock"] else args.data_dim
        self.in_len = args.in_len
        self.out_len = args.out_len
        self.patch_len = args.patch_len
        self.n_cluster = args.n_cluster
        self.d_model = args.d_model
        self.d_ff = args.d_ff
        self.individual = args.individual
        self.if_moe = False
        self.baseline = baseline
        self.device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
        if if_decomposition:
            self.decomp_module = series_decomp(kernel_size=25)
            self.encoder_trend = Patch_backbone(args, device=self.device)
            self.encoder_res = Patch_backbone(args, device=self.device)
        if self.individual == "c":
            self.Cluster_assigner = Cluster_assigner(self.n_vars, self.n_cluster, self.in_len, self.d_model, device=self.device)
            self.cluster_emb = self.Cluster_assigner.cluster_emb
        else:
            if not self.if_moe:
                self.cluster_emb = torch.empty(self.n_cluster, self.d_model).to(self.device)
            else:
                self.cluster_emb = nn.Parameter(torch.empty(self.n_cluster, self.d_model).to(self.device), requires_grad=True)
                nn.init.kaiming_uniform_(self.cluster_emb, a=math.sqrt(5))

        self.encoder = Patch_backbone(args, device=self.device)
        self.decomposition = if_decomposition
        self.cluster_prob = None

        self.categorical = args.categorical
        if args.categorical:
            self.cat_embed = Cat_Embed(cat_indices=[16,17], cat_vocab_size=[3,3])
            self.mixed_proj = MixedProjector(cat_indices=[16,17], vocab_sizes=[3,3])
            # Add two learnable log-variance parameters for dynamic loss scaling
            self.log_sigma_num = nn.Parameter(torch.zeros(()))
            self.log_sigma_cat = nn.Parameter(torch.zeros(()))

            self.rev_in = RevIN(args.data_dim-2, affine=True, subtract_last=False)
        else:
            pass
            #self.rev_in = RevIN(args.data_dim, affine=True, subtract_last=False)
        
    def forward(self, x_seq, if_update=False):       #[bs, seq_len, n_vars]
        if (self.baseline):
            base = x_seq.mean(dim = 1, keepdim = True)
        else:
            base = 0
        if self.individual == "c":
            self.cluster_prob, cluster_emb_1 = self.Cluster_assigner(x_seq, self.cluster_emb)      #[n_vars, n_cluster]
        if self.decomposition:
            res_init, trend_init = self.decomp_module(x_seq)
            res_init, trend_init = res_init.permute(0,2,1), trend_init.permute(0,2,1)  # x: [Batch, Channel, Input length]
            res, cls_emb_res = self.encoder_res(res_init, self.cluster_emb, self.cluster_prob)
            trend, cls_emd_trend = self.encoder_trend(trend_init,  self.cluster_emb, self.cluster_prob)
            out = res + trend
            cluster_emb = (cls_emb_res + cls_emd_trend)/2
            if if_update and self.individual == "c":
                self.cluster_emb = nn.Parameter(cluster_emb_1, requires_grad=True)
            out = out.permute(0,2,1)    # x: [Batch, Input length, Channel]
        else:
            #x_seq is of shape [bs, in_len, n_vars]
            x_seq = x_seq.permute(0,2,1)

            if self.categorical:
                x = self.cat_embed(x_seq)  # x: [bs, n_vars, in_len], with categorical channels embedded
                x = x.transpose(1,2) # now x is [bs, in_seq, n_vars]
                x_num = x[:, :, :16]      # numerical features
                x_cat = x[:, :, 16:]      # categorical features
                x_num = self.rev_in(x_num, mode="norm")
                x = torch.cat([x_num, x_cat], dim=2)  #[bs, in_seq, n_vars]
                x = x.transpose(1,2)
            else:
                x = x_seq

            out, cls_emb, gate_score_lis = self.encoder(x, self.cluster_emb, self.cluster_prob) # out: [bs, n_vars, out_len]

            if if_update and self.individual == "c":
                self.cluster_emb = nn.Parameter(cluster_emb_1, requires_grad=True)

            if self.categorical:
                #This is for Telecom
                out = out.transpose(1,2) # [bs, n_vars, out_len] -> [bs, out_len, n_vars]
                x_num_ = out[:, :, :16]      # numerical features
                x_cat_ = out[:, :, 16:]      # categorical features
                x_num_ = self.rev_in(x_num_, mode="denorm")
                out = torch.cat([x_num_, x_cat_], dim=2)  #[bs, in_seq, n_vars]

                out = self.mixed_proj(out.transpose(1,2))
                return out, gate_score_lis, self.cluster_prob  #[bs, out_len, n_vars] 
            else:
                #This line is normally used
                out = out.permute (0,2,1)
                return base + out[:, :self.out_len, :], gate_score_lis, self.cluster_prob  #[bs, out_len, n_vars] 
"""