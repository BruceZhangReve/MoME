import torch
import torch.nn as nn
import torch.nn.functional as F


class EiLM(nn.Module):
    """
    An expert-wise Linear Modulation Layer.
    Current (MoE) implementation requires B == 1, can be revised for future versions.
    Input: Instruction tokens: [B, N_i, hidden_dim]; Tokens in main tasks received by an expert.
    Output: Modulated expert output for main task: [N_e, hidden_dim]
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


class MMLinear(nn.Module):
    def __init__(self, in_len, out_len, top_k, n_experts, modulation=False,
                 router_modulation=False, lambda_e=0.75):
        super().__init__()
        # Basic Components
        self.in_len = in_len
        self.out_len = out_len
        self.top_k = top_k
        self.num_experts = n_experts
        self.router_modulation = router_modulation
        self.modulation = modulation
        self.lambda_e = lambda_e
        # MoE Linear out Proj
        self.Gate = nn.Linear(in_len, self.num_experts, bias=False)
        self.norm_topk_prob = False
        self.experts = nn.ModuleList(
            [nn.Linear(in_len, out_len) for _ in range(self.num_experts)]
        )
        # Expert-instruction Modulation (EiLM)
        if self.modulation:
            print("EiLM enabled!")
            self.EiLM = nn.ModuleList(
                [EiLM(out_len) for _ in range(self.num_experts)]
            )
            # Router modulation (RM)
            if self.router_modulation:
                print(f"RM enabled with lambda_e = {self.lambda_e}")
                self.router_modulator = nn.Linear(out_len, self.num_experts, bias=False)
        else:
            self.EiLM = None
            self.router_modulator = None

    def forward(self, x, Ins_tk=None):
        """
        Input:
        x - [B, C, L], batch_size, channel, input_len
        Ins_tk - [B, N_i, hidden_dim], N_i: number of instruction tokens
        Output: encoded - [B, C, out_len]
        """
        B, C, L = x.shape

        if L != self.in_len:
            raise ValueError(f"Input length must be {self.in_len}, but got {L}")

        x = x.reshape(-1, self.in_len)  # [B, C, L] => [B*C, L]

        # Apply router modulation if enabled
        if self.router_modulation and self.modulation and Ins_tk is not None:
            router_logits = self.Gate(x)  # [B*C, num_experts]
            routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float32)

            router_gamma = torch.mean(self.router_modulator(Ins_tk), dim=1)[0]  # [num_experts]
            router_gamma = router_gamma.unsqueeze(0).expand_as(router_logits)  # [B*C, num_experts]

            routing_weights = routing_weights + self.lambda_e * F.sigmoid(router_gamma)
        else:
            router_logits = self.Gate(x)  # [B*C, num_experts]
            routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float32)

        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)  # [B*C, top_k]
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

        final_hidden_states = torch.zeros(
            (B * C, self.out_len), dtype=x.dtype, device=x.device
        )
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        if self.modulation == False:
            for expert_idx in expert_hit:
                expert_idx = expert_idx.item()
                expert_layer = self.experts[expert_idx]
                idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))

                current_state = x[None, top_x].reshape(-1, self.in_len)  # [N_i, L]
                current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

                final_hidden_states.index_add_(0, top_x, current_hidden_states.to(x.dtype))

            final_hidden_states = final_hidden_states.reshape(B, C, self.out_len)
            out = final_hidden_states
        else:
            for expert_idx in expert_hit:
                expert_idx = expert_idx.item()
                expert_layer = self.experts[expert_idx]
                EiLM_layer = self.EiLM[expert_idx]

                idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))

                current_state = x[None, top_x].reshape(-1, self.in_len)  # [N_i, L]

                # Expert-Level Modulation
                expert_output = expert_layer(current_state)  # [N_i, out_len]
                expert_output = EiLM_layer(expert_output, Ins_tk)  # [N_i, out_len], modulated

                current_hidden_states = expert_output * routing_weights[top_x, idx, None]

                final_hidden_states.index_add_(0, top_x, current_hidden_states.to(x.dtype))

            final_hidden_states = final_hidden_states.reshape(B, C, self.out_len)
            out = final_hidden_states

        return out
