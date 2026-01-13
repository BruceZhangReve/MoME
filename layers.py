import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional



    
########################################################

class QueryPool(nn.Module):
    """
    From LLM's token hidden states (B, T, d_llm), by cross-attention,
    extract `n_queries` learnable prototype vectors, and project to low-dimensional space.
    """
    def __init__(
        self,
        d_model: int,
        n_queries: int = 3,
        n_heads: int = 1,
        d_proj: int = 512,
        dropout: float = 0.1,
        semantic_init: Optional[torch.Tensor] = None,  # [n_queries, d_proj] or None
    ):
        super().__init__()
        self.n_queries = n_queries
        self.d_model = d_model
        
        self.proj = nn.Linear(d_model, d_proj) # [B, T, d_llm] => [B, n_queries, d_proj]
        # init as [Q, d]
        self.queries = nn.Parameter(torch.empty(n_queries, d_proj))

        # semantic initialization
        if semantic_init is not None and semantic_init.shape == (n_queries, d_proj):
            with torch.no_grad():
                q = F.normalize(semantic_init, dim=-1)
                self.queries.copy_(q)
        else:
            # orthogonal initialization
            if n_queries> d_proj:
                self.queries.data.normal_(0, 1) # Standard Normal
                self.queries.data = F.normalize(self.queries.data, p=2, dim=-1) # L2 normalize the last dimension
                #raise ValueError(f"Please choose the number of instruction tokens to be smaller than {d_proj}")
            else:
                nn.init.orthogonal_(self.queries)

        self.attn = nn.MultiheadAttention(
            embed_dim=d_proj,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.out_norm = nn.LayerNorm(d_proj)

    @torch.no_grad()
    def set_semantic_queries(self, semantic_init: torch.Tensor):
        """
        semantic_init: [n_queries, d_model]
        """
        assert semantic_init.shape == (self.n_queries, self.d_proj)
        self.queries.copy_(F.normalize(semantic_init, dim=-1))

    def forward(
        self,
        hidden_states: torch.Tensor,         # [B, T, d_model]
        attention_mask: Optional[torch.Tensor] = None  # [B, T]，1=valid，0=PAD
    ) -> torch.Tensor:
        #Return textural instruction tokens: [B, n_queries, d_proj]
        B, T, D = hidden_states.shape
        assert D == self.d_model

        # construct queries
        Q = self.queries.unsqueeze(0).expand(B, -1, -1)  # [B, n_queries, d_proj]

        # key_padding_mask: True represents the neglected positions（PAD）
        key_padding_mask = None
        if attention_mask is not None:
            # attention_mask: 1 = keep, 0 = pad -> key_padding_mask: True for pad
            key_padding_mask = ~attention_mask.bool()    # [B, T], True=neglect

        hidden_states = self.proj(hidden_states)

        # Cross-Attention：Q attends to hidden_states (K=V)
        # attn_output: [B, n_queries, d_model]
        attn_output, _ = self.attn(
            query=Q,
            key=hidden_states,
            value=hidden_states,
            key_padding_mask=key_padding_mask,  # ignore PAD tokens
            need_weights=False
        )

        features = self.out_norm(attn_output)        # [B, n_queries, d_proj]
        return features


@torch.no_grad()
def build_semantic_init(
    tokenizer, llm, concepts: List[str], max_len: int = 64
) -> torch.Tensor:
    """
    Use LLM to freeze conceptual tokens as queries, and return an init matrix of shape: [n_queries, d_model].
    """

    reps = []
    for c in concepts:
        tok = tokenizer(c, return_tensors="pt", truncation=True, max_length=max_len).to(llm.device)
        out = llm(**tok, output_hidden_states=True, use_cache=False)
        h = out.hidden_states[-1].mean(dim=1)  # [1, d_model]
        reps.append(h.squeeze(0))
    E = torch.stack(reps, dim=0)                 # [n_queries, d_model]
    E = F.normalize(E, dim=-1)
    return E
