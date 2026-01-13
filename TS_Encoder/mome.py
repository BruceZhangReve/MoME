import torch
import torch.nn as nn
import torch.nn.functional as F

class MoMe(nn.Module):
    def __init__(self, in_len, patch_len, hidden_dim, top_k, num_experts, modulation=False):
        super().__init__()
        # Basic Components
        self.in_len = in_len
        self.patch_len = patch_len
        self.hidden_dim = hidden_dim
        self.top_k = top_k
        self.num_experts = num_experts
        self.moe_d_ff = hidden_dim
        self.proj = nn.Linear(self.patch_len, hidden_dim)
        # Externel Modulation
        self.modulation = modulation
        if self.modulation == True:
            self.EiLM = nn.ModuleList(
                [EiLM(hidden_dim) for _ in range(self.num_experts)]
                )
        else:
            self.EiLM = None
        
        # We calculate patch_num, P. Ensure L is divisible by patch_len
        self.patch_num = (in_len + patch_len - 1) // patch_len
        self.L_padded = self.patch_num * self.patch_len  # the actual length after padding

        self.Gate = nn.Linear(hidden_dim, self.num_experts, bias=False)

        self.norm_topk_prob = False

        self.experts = nn.ModuleList(
            [MoMeMLP(hidden_dim, intermediate_size=self.moe_d_ff) for _ in range(self.num_experts)]
        )


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
        if L < self.L_padded: # Ensure L is divisible by patch_len, otherwise padding
            pad_len = self.L_padded - L
            x = F.pad(x, (0, pad_len)) 

        #x: [B, C, L] => [B, C, P, patch_len]
        x = x.view(B, C, self.patch_num, self.patch_len)  # [B, C, P, patch_len]

        x = self.proj(x) #[B, C, P, patch_len] => [B, C, P, d]

        tokens = x.reshape(-1, self.hidden_dim) # [B*C*P, d]

        router_logits = self.Gate(tokens)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float32) #[B*C*P, E]

        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1) #[B*C*P, E]
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

        final_hidden_states = torch.zeros(
            (B * C * self.patch_num, self.hidden_dim), dtype=x.dtype, device=x.device
            )
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        if self.modulation == False:
            for expert_idx in expert_hit:
                expert_layer = self.experts[expert_idx]
                idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))

                current_state = tokens[None, top_x].reshape(-1, self.hidden_dim) #[N_i, d]
                current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None] #[N_i, d]

                final_hidden_states.index_add_(0, top_x, current_hidden_states.to(x.dtype))

            final_hidden_states = final_hidden_states.reshape(B, C, self.patch_num, self.hidden_dim) #[B*C*P, d] -> [B, C, P, d]

            out = final_hidden_states.reshape(B, -1, self.hidden_dim) #[B, CP, d]
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

            final_hidden_states = final_hidden_states.reshape(B, C, self.patch_num, self.hidden_dim) #[B*C*P, d] -> [B, C, P, d]

            out = final_hidden_states.reshape(B, -1, self.hidden_dim) #[B, CP, d]
        return out


# MoMe w/ RM
class MoMeP(nn.Module):
    def __init__(self, in_len, patch_len, hidden_dim, top_k, num_experts, modulation=False):
        super().__init__()
        # Basic Components
        self.in_len = in_len
        self.patch_len = patch_len
        self.hidden_dim = hidden_dim
        self.top_k = top_k
        self.num_experts = num_experts
        self.moe_d_ff = hidden_dim
        self.proj = nn.Linear(self.patch_len, hidden_dim)
        # Externel Modulation
        self.modulation = modulation
        if self.modulation == True:
            self.EiLM = nn.ModuleList(
                [EiLM(hidden_dim) for _ in range(self.num_experts)]
                )
            self.router_modulator = nn.Linear(hidden_dim, self.num_experts, bias=False)
        else:
            self.EiLM = None
        
        # We calculate patch_num, P. Ensure L is divisible by patch_len
        self.patch_num = (in_len + patch_len - 1) // patch_len
        self.L_padded = self.patch_num * self.patch_len  # the actual length after padding

        self.Gate = nn.Linear(hidden_dim, self.num_experts, bias=False)

        self.norm_topk_prob = False

        self.experts = nn.ModuleList(
            [MoMeMLP(hidden_dim, intermediate_size=self.moe_d_ff) for _ in range(self.num_experts)]
        )


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
        if L < self.L_padded: # Ensure L is divisible by patch_len, otherwise padding
            pad_len = self.L_padded - L
            x = F.pad(x, (0, pad_len)) 

        #x: [B, C, L] => [B, C, P, patch_len]
        x = x.view(B, C, self.patch_num, self.patch_len)  # [B, C, P, patch_len]

        x = self.proj(x) #[B, C, P, patch_len] => [B, C, P, d]

        tokens = x.reshape(-1, self.hidden_dim) # [B*C*P, d]

        router_logits = self.Gate(tokens) # [B*C*P, num_experts]
        router_gamma = torch.mean(self.router_modulator(Ins_tk), dim=1)[0] # [B, N_i, hidden_dim] -> [B, N_i, num_experts] -> [B, num_experts] -> [num_experts]
        router_gamma.unsqueeze(0).expand_as(router_logits) # [B*C*P, num_experts]
        router_logits = router_gamma + router_logits # [B*C*P, num_experts]

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float32) #[B*C*P, E]

        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1) #[B*C*P, E]
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

        final_hidden_states = torch.zeros(
            (B * C * self.patch_num, self.hidden_dim), dtype=x.dtype, device=x.device
            )
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        if self.modulation == False:
            for expert_idx in expert_hit:
                expert_layer = self.experts[expert_idx]
                idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))

                current_state = tokens[None, top_x].reshape(-1, self.hidden_dim) #[N_i, d]
                current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None] #[N_i, d]

                final_hidden_states.index_add_(0, top_x, current_hidden_states.to(x.dtype))

            final_hidden_states = final_hidden_states.reshape(B, C, self.patch_num, self.hidden_dim) #[B*C*P, d] -> [B, C, P, d]

            out = final_hidden_states.reshape(B, -1, self.hidden_dim) #[B, CP, d]
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

            final_hidden_states = final_hidden_states.reshape(B, C, self.patch_num, self.hidden_dim) #[B*C*P, d] -> [B, C, P, d]

            out = final_hidden_states.reshape(B, -1, self.hidden_dim) #[B, CP, d]
        return out
    

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

        #print(betas.shape, x.shape)

        gammas = gammas.unsqueeze(0).expand_as(x)
        betas = betas.unsqueeze(0).expand_as(x)

        return (gammas * x) + betas

