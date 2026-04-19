import torch
import torch.nn as nn
import torch.nn.functional as F


class MoMeMLP(nn.Module):
    """Modified from transformers.models.mistral.modeling_mistral.MistralMLP"""
    def __init__(self, hidden_dim, intermediate_size=None):
        super().__init__()
        self.hidden_size = hidden_dim
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = F.silu

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class EiLM(nn.Module):
    """
    An expert-wise Linear Modulation Layer.
    Current implementation requires B == 1, can be revised for future versions.
    Input: Instruction tokens: [B, N_i, hidden_dim]; Tokens in main tasks received by an expert.
    Output: Modulated expert output for main task: [N_i, hidden_dim]
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.gamma_generator = nn.Linear(hidden_dim, 1, bias=False)
        self.beta_generator = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x, Ins_tk):
        # x: [N_e, hidden_dim]
        B, _, _ = Ins_tk.shape
        assert B == 1, "Current (MoE) implementation requires B == 1 to enable expert-level modulation!"

        gammas = torch.mean(self.gamma_generator(Ins_tk), dim=1)[0]  # [B, N_i, hidden_dim] -> [B, 1] -> [1]
        betas = torch.mean(self.beta_generator(Ins_tk), dim=1)[0]  # [B, N_i, hidden_dim] -> [B, hidden_dim] -> [hidden_dim]

        gammas = gammas.unsqueeze(0).expand_as(x)
        betas = betas.unsqueeze(0).expand_as(x)

        return (gammas * x) + betas


class MoMe(nn.Module):
    """
    Unified MoME model with optional router modulation and visualization output.

    Args:
        router_modulation: bool - whether to use router_modulation
        return_expert_selection: bool - whether to return selected_experts for visualization/case study
    """
    def __init__(self, in_len, patch_len, hidden_dim, top_k, num_experts, modulation=False,
                 router_modulation=False, return_expert_selection=False, lambda_e=0.75):
        super().__init__()
        # Basic Components
        self.in_len = in_len
        self.patch_len = patch_len
        self.hidden_dim = hidden_dim
        self.top_k = top_k
        self.num_experts = num_experts
        self.return_expert_selection = return_expert_selection
        self.moe_d_ff = hidden_dim
        self.lambda_e = lambda_e
        self.proj = nn.Linear(self.patch_len, hidden_dim)

        self.router_modulation = router_modulation
        self.lambda_e = lambda_e
        self.modulation = modulation

         # Expert Modulation
        if self.modulation:
            print("EiLM enabled!")
            self.EiLM = nn.ModuleList(
                [EiLM(hidden_dim) for _ in range(self.num_experts)]
            )
        else:
            self.EiLM = None
        
        # Router modulation
        if self.router_modulation and self.modulation:
            print("RM enabled with lambda_e =", self.lambda_e)
            self.router_modulator = nn.Linear(hidden_dim, self.num_experts, bias=False)

        # Calculate patch_num, ensure L is divisible by patch_len
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
        Output:
            if return_expert_selection=False: encoded - [B, CP, d]
            if return_expert_selection=True: (encoded, selected_experts)
        """
        B, C, L = x.shape

        if L != self.in_len:
            raise ValueError(f"Input length must be {self.in_len}, but got {L}")
        if L < self.L_padded:  # Ensure L is divisible by patch_len, otherwise padding
            pad_len = self.L_padded - L
            x = F.pad(x, (0, pad_len))

        # x: [B, C, L] => [B, C, P, patch_len]
        x = x.view(B, C, self.patch_num, self.patch_len)  # [B, C, P, patch_len]

        x = self.proj(x)  # [B, C, P, patch_len] => [B, C, P, d]

        tokens = x.reshape(-1, self.hidden_dim)  # [B*C*P, d]

        # Apply router modulation if enabled
        if self.router_modulation and self.modulation and Ins_tk is not None:
            router_logits = self.Gate(tokens)  # [B*C*P, num_experts]
            routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float32)  # [B*C*P, E]

            router_gamma = torch.mean(self.router_modulator(Ins_tk), dim=1)[0]  # [B, N_i, hidden_dim] -> [num_experts]
            router_gamma = router_gamma.unsqueeze(0).expand_as(router_logits)  # [B*C*P, num_experts]

            routing_weights = routing_weights + self.lambda_e * F.sigmoid(router_gamma)
        else:
            router_logits = self.Gate(tokens)  # [B*C*P, num_experts]
            routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float32)  # [B*C*P, E]

        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)  # [B*C*P, top_k]
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

                current_state = tokens[None, top_x].reshape(-1, self.hidden_dim)  # [N_i, d]
                current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]  # [N_i, d]

                final_hidden_states.index_add_(0, top_x, current_hidden_states.to(x.dtype))

            final_hidden_states = final_hidden_states.reshape(B, C, self.patch_num, self.hidden_dim)
            out = final_hidden_states.reshape(B, -1, self.hidden_dim)
        else:
            for expert_idx in expert_hit:
                expert_layer = self.experts[expert_idx]
                EiLM_layer = self.EiLM[expert_idx]

                idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))

                current_state = tokens[None, top_x].reshape(-1, self.hidden_dim)  # [N_i, d]

                # Expert-Level Modulation
                expert_output = expert_layer(current_state)  # [N_i, d]
                expert_output = EiLM_layer(expert_output, Ins_tk)  # [N_i, d], modulated

                current_hidden_states = expert_output * routing_weights[top_x, idx, None]  # [N_i, d]

                final_hidden_states.index_add_(0, top_x, current_hidden_states.to(x.dtype))

            final_hidden_states = final_hidden_states.reshape(B, C, self.patch_num, self.hidden_dim)
            out = final_hidden_states.reshape(B, -1, self.hidden_dim)

        if self.return_expert_selection:
            return out, selected_experts
        else:
            return out
