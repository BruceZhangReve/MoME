import torch
import torch.nn as nn
import torch.nn.functional as F


class MMLinear(nn.Module):
    def __init__(self, in_len, out_len, top_k, n_experts, modulation=False):
        super().__init__()
        # Basic Components
        self.in_len = in_len
        self.out_len = out_len
        self.top_k = top_k
        self.num_experts = n_experts
        # MoE Linear out Proj
        self.Gate = nn.Linear(in_len, self.num_experts, bias=False)
        self.norm_topk_prob = False
        self.experts = nn.ModuleList(
            [nn.Linear(in_len, out_len) for _ in range(self.num_experts)]
        )
        # Externel Modulation
        self.modulation = modulation
        if self.modulation == True:
            self.EiLM = nn.ModuleList(
                [EiLM(out_len) for _ in range(self.num_experts)]
                )
        else:
            self.EiLM = None


    def forward(self, x, Ins_tk=None):
        """
        Input: 
        x - [B, C, L], batch_size, channel, input_len
        Ins_tk - [B, N_i, hidden_dim], N_i: number of instruction tokens
        Output: encoded - [B, C, P, D]
        """
        B, C, L = x.shape

        if L != self.in_len:
            raise ValueError(f"Input length must be {self.in_len}, but got {L}")

        x = x.reshape(-1, self.in_len) # [B, C, L] => [B*C, L]

        router_logits = self.Gate(x)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float32) #[B*C, E]

        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1) #[B*C, E]
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

        final_hidden_states = torch.zeros(
            (B * C, self.out_len), dtype=x.dtype, device=x.device
            )
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        if self.modulation == False:
            for expert_idx in expert_hit:
                expert_layer = self.experts[expert_idx]
                idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))

                current_state = x[None, top_x].reshape(-1, self.in_len) #[N_i, L]
                current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None] #[N_i, L']

                final_hidden_states.index_add_(0, top_x, current_hidden_states.to(x.dtype))

            final_hidden_states = final_hidden_states.reshape(B, C, self.out_len) #[B*C, L'] -> [B, C, L']

            out = final_hidden_states #[B, C, L']
        else:
            for expert_idx in expert_hit:
                expert_layer = self.experts[expert_idx]
                EiLM_layer = self.EiLM[expert_idx]

                idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))

                current_state = x[None, top_x].reshape(-1, self.in_len) #[N_i, L]

                # Expert-Level Modulation #
                expert_output = expert_layer(current_state) #[N_i, L']
                expert_output = EiLM_layer(expert_output, Ins_tk) #[N_i, L'], modulated
                 # Expert-Level Modulation #

                current_hidden_states = expert_output  * routing_weights[top_x, idx, None] #[N_i, d]

                final_hidden_states.index_add_(0, top_x, current_hidden_states.to(x.dtype))

            final_hidden_states = final_hidden_states.reshape(B, C, self.out_len) #[B*C, d] -> [B, C, d]

            out = final_hidden_states #[B, C, L']
        return out


class MMLinearP(nn.Module):
    def __init__(self, in_len, out_len, top_k, n_experts, modulation=False):
        super().__init__()
        # Basic Components
        self.in_len = in_len
        self.out_len = out_len
        self.top_k = top_k
        self.num_experts = n_experts
        # MoE Linear out Proj
        self.Gate = nn.Linear(in_len, self.num_experts, bias=False)
        self.norm_topk_prob = False
        self.experts = nn.ModuleList(
            [nn.Linear(in_len, out_len) for _ in range(self.num_experts)]
        )
        # Externel Modulation
        self.modulation = modulation
        if self.modulation == True:
            self.EiLM = nn.ModuleList(
                [EiLM(out_len) for _ in range(self.num_experts)]
                )
            self.router_modulator = nn.Linear(out_len, self.num_experts, bias=False)
        else:
            self.EiLM = None


    def forward(self, x, Ins_tk=None):
        """
        Input: 
        x - [B, C, L], batch_size, channel, input_len
        Ins_tk - [B, N_i, hidden_dim], N_i: number of instruction tokens
        Output: encoded - [B, C, P, D]
        """
        B, C, L = x.shape

        if L != self.in_len:
            raise ValueError(f"Input length must be {self.in_len}, but got {L}")

        x = x.reshape(-1, self.in_len) # [B, C, L] => [B*C, L]

        router_logits = self.Gate(x)
        router_gamma = torch.mean(self.router_modulator(Ins_tk), dim=1)[0] # [B, N_i, out_len] -> [B, N_i, num_experts] -> [B, num_experts] -> [num_experts]
        router_gamma.unsqueeze(0).expand_as(router_logits) # [B*C*P, num_experts]
        router_logits = router_gamma + router_logits # [B*C*P, num_experts]
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float32) #[B*C, E]

        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1) #[B*C, E]
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

        final_hidden_states = torch.zeros(
            (B * C, self.out_len), dtype=x.dtype, device=x.device
            )
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        if self.modulation == False:
            for expert_idx in expert_hit:
                expert_layer = self.experts[expert_idx]
                idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))

                current_state = x[None, top_x].reshape(-1, self.in_len) #[N_i, L]
                current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None] #[N_i, L']

                final_hidden_states.index_add_(0, top_x, current_hidden_states.to(x.dtype))

            final_hidden_states = final_hidden_states.reshape(B, C, self.out_len) #[B*C, L'] -> [B, C, L']

            out = final_hidden_states #[B, C, L']
        else:
            for expert_idx in expert_hit:
                expert_layer = self.experts[expert_idx]
                EiLM_layer = self.EiLM[expert_idx]

                idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))

                current_state = x[None, top_x].reshape(-1, self.in_len) #[N_i, L]

                # Expert-Level Modulation #
                expert_output = expert_layer(current_state) #[N_i, L']
                expert_output = EiLM_layer(expert_output, Ins_tk) #[N_i, L'], modulated
                 # Expert-Level Modulation #

                current_hidden_states = expert_output  * routing_weights[top_x, idx, None] #[N_i, d]

                final_hidden_states.index_add_(0, top_x, current_hidden_states.to(x.dtype))

            final_hidden_states = final_hidden_states.reshape(B, C, self.out_len) #[B*C, d] -> [B, C, d]

            out = final_hidden_states #[B, C, L']
        return out
    
    
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
