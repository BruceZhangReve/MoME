import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.layers import *
from models.patch_layer import *

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1) # x: [bs, in_len, n_vars]
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class DLinearC(nn.Module):
    #Decomposition-Linear

    def __init__(self, args):
        super(DLinearC, self).__init__()
        self.seq_len = args.in_len
        self.pred_len = args.out_len
        self.n_experts = args.n_experts
        #self.d_ff = args.d_ff
        self.moe_mode = args.moe_mode
        self.top_k = args.topk
        self.device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
        self.categorical = args.categorical

        # Decompsition Kernel Size
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        self.channels = args.batch_size if args.data == "M4" else args.data_dim

        if self.moe_mode == "multi_expert":
            self.Gating_sea = nn.Linear(self.seq_len, self.n_experts, bias=False)
            self.experts_sea = nn.ModuleList([nn.Linear(self.seq_len,self.pred_len) for _ in range(self.n_experts)])
            self.Gating_trend = nn.Linear(self.seq_len, self.n_experts, bias=False)
            self.experts_trend = nn.ModuleList([nn.Linear(self.seq_len,self.pred_len) for _ in range(self.n_experts)])
        else:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            
            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len,self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len,self.pred_len))   

        if args.categorical:
            self.cat_embed = Cat_Embed(cat_indices=[16,17], cat_vocab_size=[3,3])
            self.mixed_proj = MixedProjector(cat_indices=[16,17], vocab_sizes=[3,3])
            # Add two learnable log-variance parameters for dynamic loss scaling
            self.log_sigma_num = nn.Parameter(torch.zeros(()))
            self.log_sigma_cat = nn.Parameter(torch.zeros(()))
            

    def forward(self, x, if_update=False):
        # x: [Batch, Input length, Channel]
        B, in_len, n_vars = x.shape
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1) #[bs, n_vars, in_len]

        #gate_scores_trend = None; gate_scores_sea = None
        if self.moe_mode == "multi_expert":
            gate_logits_sea = self.Gating_sea(seasonal_init) #[B, N, E]
            gate_logits_sea = F.softmax(gate_logits_sea, dim=-1)
            weights_sea, selected_experts_sea = torch.topk(gate_logits_sea, self.top_k)  # [bs, n_vars, topk]
            #results_sea = torch.zeros_like(seasonal_init) #[bs, n_vars, in_len]
            results_sea = torch.zeros(B, n_vars, self.pred_len, device=x.device, dtype=x.dtype)
            for i, expert in enumerate(self.experts_sea):
                batch_idx, token_idx, kth = torch.where(selected_experts_sea == i)
                #print(expert(x[batch_idx, token_idx]).shape) #[N_i, out_len]
                results_sea[batch_idx, token_idx] += (
                    weights_sea[batch_idx, token_idx, kth][:, None] * expert(seasonal_init[batch_idx, token_idx])
                    )
            seasonal_output = results_sea 
                
            gate_logits_trend = self.Gating_trend(trend_init)
            gate_logits_trend = F.softmax(gate_logits_trend, dim=-1)
            weights_trend, selected_experts_trend = torch.topk(gate_logits_trend, self.top_k)  # [bs, n_vars, topk]
            #results_trend = torch.zeros_like(trend_init) #[bs, n_vars, d_model]
            results_trend = torch.zeros(B, n_vars, self.pred_len, device=x.device, dtype=x.dtype)
            for i, expert in enumerate(self.experts_trend):
                batch_idx, token_idx, kth = torch.where(selected_experts_trend == i)
                #print(expert(x[batch_idx, token_idx]).shape) #[N_i, d_model]
                results_trend[batch_idx, token_idx] += (
                    weights_trend[batch_idx, token_idx, kth][:, None] * expert(trend_init[batch_idx, token_idx])
                    )
                
            trend_output = results_trend

            x = (seasonal_output + trend_output).permute(0,2,1) #[B,N,Out_len] -> [B,out_len,N]

            if self.categorical:
                x_num_ = x[:, :, :16]      # numerical features
                x_cat_ = x[:, :, 16:]      # categorical features
                x = torch.cat([x_num_, x_cat_], dim=2)  #[bs, out_len, n_vars]
                output = self.mixed_proj(x.transpose(1,2)) 
            else: 
                output = x

            #return x.permute(0,2,1), [gate_logits_trend.mean(dim=0)], None#gate_scores_trend, self.cluster_prob # to [Batch, Output length, Channel]
            return output, [gate_logits_trend.mean(dim=0)], None#gate_scores_trend, self.cluster_prob # to [Batch, Output length, Channel]
        else:
            seasonal_output = torch.zeros([seasonal_init.size(0),seasonal_init.size(1),self.pred_len],dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0),trend_init.size(1),self.pred_len],dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:,i,:] = self.Linear_Seasonal[i](seasonal_init[:,i,:])
                trend_output[:,i,:] = self.Linear_Trend[i](trend_init[:,i,:])
        
            x = (seasonal_output + trend_output).permute(0,2,1) #[B,N,Out_len] -> [B,out_len,N]

            if self.categorical:
                x_num_ = x[:, :, :16]      # numerical features
                x_cat_ = x[:, :, 16:]      # categorical features
                x = torch.cat([x_num_, x_cat_], dim=2)  #[bs, out_len, n_vars]
                output = self.mixed_proj(x.transpose(1,2)) 
            else: 
                output = x

            return output, None, None#gate_scores_trend, self.cluster_prob # to [Batch, Output length, Channel]
            #return x.permute(0,2,1), None, None#gate_scores_trend, self.cluster_prob # to [Batch, Output length, Channel]


"""
class DLinearC(nn.Module):
    #Decomposition-Linear

    def __init__(self, args):
        super(DLinearC, self).__init__()
        self.seq_len = args.in_len
        self.pred_len = args.out_len
        self.n_cluster = args.n_cluster
        self.d_ff = args.d_ff
        self.if_moe = args.if_moe
        self.topk = args.topk
        self.device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

        # Decompsition Kernel Size
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        self.individual = args.individual
        self.channels = args.batch_size if args.data == "M4" else args.data_dim

        if self.individual == "i":
            if not self.if_moe:
                self.Linear_Seasonal = nn.ModuleList()
                self.Linear_Trend = nn.ModuleList()
            
                for i in range(self.channels):
                    self.Linear_Seasonal.append(nn.Linear(self.seq_len,self.pred_len))
                    self.Linear_Trend.append(nn.Linear(self.seq_len,self.pred_len))
            else:
                self.Gating_sea = nn.Linear(self.seq_len, self.n_cluster, bias=False)
                self.experts_sea = nn.ModuleList([nn.Linear(self.seq_len,self.pred_len) for _ in range(self.n_cluster)])
                self.Gating_trend = nn.Linear(self.seq_len, self.n_cluster, bias=False)
                self.experts_r_trend = nn.ModuleList([nn.Linear(self.seq_len,self.pred_len) for _ in range(self.n_cluster)])
                
        elif self.individual == "c":
            self.Linear_Seasonal = Cluster_wise_linear(self.n_cluster, self.channels, self.seq_len, self.pred_len, self.device)
            self.Linear_Trend = Cluster_wise_linear(self.n_cluster, self.channels,self.seq_len, self.pred_len, self.device)
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len,self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len,self.pred_len)
        if self.individual == "c":
            self.Cluster_assigner = Cluster_assigner(self.channels, self.n_cluster, self.seq_len, self.d_ff, device=self.device)
            self.cluster_emb = self.Cluster_assigner.cluster_emb
            

    def forward(self, x, if_update=False):
        # x: [Batch, Input length, Channel]
        if self.individual == "c":
            self.cluster_prob, cluster_emb = self.Cluster_assigner(x, self.cluster_emb)
        else:
            self.cluster_prob = None
        if if_update and self.individual == "c":
            self.cluster_emb = nn.Parameter(cluster_emb, requires_grad=True)
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1) #[bs, n_vars, in_len]

        gate_scores_trend = None; gate_scores_sea = None

        if self.individual == "i":
            if not self.if_moe:
                seasonal_output = torch.zeros([seasonal_init.size(0),seasonal_init.size(1),self.pred_len],dtype=seasonal_init.dtype).to(seasonal_init.device)
                trend_output = torch.zeros([trend_init.size(0),trend_init.size(1),self.pred_len],dtype=trend_init.dtype).to(trend_init.device)
                for i in range(self.channels):
                    seasonal_output[:,i,:] = self.Linear_Seasonal[i](seasonal_init[:,i,:])
                    trend_output[:,i,:] = self.Linear_Trend[i](trend_init[:,i,:])
            else:
                gate_scores_sea = self.Gating_sea(seasonal_init) # [nvars, n_clusters]
                gate_scores_trend = self.Gating_trend(trend_init) # [nvars, n_clusters]
                #gate_scores_trend = self.Gating_trend(x.permute(0,2,1)) # [nvars, n_clusters]
                expert_r_sea_outputs = torch.stack([expert_r_sea(seasonal_init) for expert_r_sea in self.experts_r_sea], dim=2) #[bs, nvars, n_clusters, pred_len]
                expert_r_trend_outputs = torch.stack([expert_r_trend(seasonal_init) for expert_r_trend in self.experts_r_trend], dim=2) #[bs, nvars, n_clusters, pred_len]
                seasonal_output = torch.einsum('nk,bnkd->bnd', gate_scores_sea, expert_r_sea_outputs)
                trend_output = torch.einsum('nk,bnkd->bnd', gate_scores_trend, expert_r_trend_outputs)
                #seasonal_output = torch.zeros([seasonal_init.size(0),seasonal_init.size(1),self.pred_len],dtype=seasonal_init.dtype).to(trend_output.device)

        elif self.individual == "c":
            seasonal_output = self.Linear_Seasonal(seasonal_init, self.cluster_prob)
            trend_output = self.Linear_Trend(trend_init, self.cluster_prob)
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)
        
        x = seasonal_output + trend_output
        return x.permute(0,2,1), gate_scores_trend, self.cluster_prob # to [Batch, Output length, Channel]
    
"""