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

            return output, [gate_logits_trend.mean(dim=0)], None#gate_scores_trend, self.cluster_prob # to [Batch, Output length, Channel]
        else:
            seasonal_output = torch.zeros([seasonal_init.size(0),seasonal_init.size(1),self.pred_len],dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0),trend_init.size(1),self.pred_len],dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:,i,:] = self.Linear_Seasonal[i](seasonal_init[:,i,:])
                trend_output[:,i,:] = self.Linear_Trend[i](trend_init[:,i,:])
        
            x = (seasonal_output + trend_output).permute(0,2,1) #[B,N,Out_len] -> [B,out_len,N]

            output = x

            return output, None, None#gate_scores_trend, self.cluster_prob # to [Batch, Output length, Channel]

