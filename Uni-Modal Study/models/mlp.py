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
        

    def forward(self, x, if_update=False):
        #x: [bs, in_seq, n_vars]
        B, in_len, n_vars = x.shape
        gate_score_lis = [] #x is of shape [bs, in_seq, n_vars]     
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


        if self.moe_mode == "No":
            return x, None, None
        elif self.moe_mode == "multi_expert":
            return x, [gate_logits.mean(dim=0)], None
        else:
            raise ValueError("Unknown moe_mode")
    
