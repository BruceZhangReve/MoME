import torch
import torch.nn as nn
from safetensors.torch import load_file as safe_load_file

import os
from typing import List, Optional, Tuple, Union
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import (
    prepare_model_for_kbit_training,
    get_peft_model,
    PeftModel,
    LoraConfig,
    TaskType,
)

from .custom_llm_layers import CustomQwen2MoeDecoderLayer


# This function builds origional llm, and equip it with LoRA
def build_llm_and_lora(args) -> Tuple[AutoModelForCausalLM]:
    dtype = torch.bfloat16 if args.use_bfloat16 else torch.float32
    target_modules = args.lora_modules.split(",") #e.g. ['q_proj', 'k_proj', 'v_proj', 'o_proj']

    model = AutoModelForCausalLM.from_pretrained(
        args.llm_model,
        torch_dtype=dtype,
        device_map="balanced",
        low_cpu_mem_usage=True
    )

    if args.lora_trainable:

        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,#args.lora_modules.split(","),
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_cfg)
        model.train()
        
    else: #fully trainbale, certain layers
        for param in model.parameters():
            param.requires_grad = False
        for name, module in model.named_modules():
            module_short_name = name.split(".")[-1] #like "model.layers.0.self_attn.q_proj"
            if module_short_name in target_modules:
                for param in module.parameters():
                    param.requires_grad = True
        
        print("\n=== Fully Trainable Modules ===")
        for name, module in model.named_modules():
            if any(param.requires_grad for param in module.parameters(recurse=False)):
                print(f"{name} ({type(module).__name__})")
        print(f"Total {len([n for n, m in model.named_modules() if any(p.requires_grad for p in m.parameters(recurse=False))])} trainable modules.")

    return model

def load_llm_for_evaluation(args):
    dtype = torch.bfloat16 if args.use_bfloat16 else torch.float32

    if args.lora_trainable:
        model = AutoModelForCausalLM.from_pretrained(
            args.llm_model,
            torch_dtype=dtype,
            device_map="balanced",
            low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model, 
            args.lora_adapter_path, 
            device_map="balanced",
            torch_dtype=dtype,
            is_trainable=False 
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.lora_adapter_path,
            torch_dtype=dtype,
            device_map="balanced",
            low_cpu_mem_usage=True
        )

    model.eval()

    return model

# This function builds customized llm, and equip it with LoRA
def build_llm_with_custom_layers_and_lora(args, custom_layer_cls) -> AutoModelForCausalLM:
    dtype = torch.bfloat16 if args.use_bfloat16 else torch.float32
    target_modules = args.lora_modules.split(",") #e.g. ['q_proj', 'k_proj', 'v_proj', 'o_proj']

    model = AutoModelForCausalLM.from_pretrained(
        args.llm_model,
        torch_dtype=dtype,
        device_map="balanced",
        low_cpu_mem_usage=True
    )
    # make sure device and dtype matches
    ref_param = next(model.parameters())
    dev, dt = ref_param.device, ref_param.dtype
    # 2.2 substitute layers
    layers = model.model.layers if hasattr(model.model, 'layers') else model.model.h
    
    for i in range(len(layers)):
        new_layer = custom_layer_cls(model.config, i, args.in_len).to(dev, dtype=dt)
        old_layer = layers[i]
        
        missing, unexpected = new_layer.load_state_dict(old_layer.state_dict(), strict=False)
        if hasattr(new_layer.mlp, "ts_gate") and hasattr(new_layer.mlp, "gate"):
            print(f"Initialize ts_router in layer {i}")
            with torch.no_grad():
                assert new_layer.mlp.ts_gate.weight.shape == new_layer.mlp.gate.weight.shape, \
                    f"shape mismatch: ts_gate {new_layer.mlp.ts_gate.weight.shape} vs gate {new_layer.mlp.gate.weight.shape}"
                
                new_layer.mlp.ts_gate.weight.copy_(new_layer.mlp.gate.weight.to(dev, dtype=dt))
        
        layers[i] = new_layer
        del old_layer
    
    torch.cuda.empty_cache()

    if args.lora_trainable:

        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,#args.lora_modules.split(","),
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_cfg)
        model.train()
        model.print_trainable_parameters()
        
    else: #fully trainbale, certain layers
        for param in model.parameters():
            param.requires_grad = False
        for name, module in model.named_modules():
            module_short_name = name.split(".")[-1] #like "model.layers.0.self_attn.q_proj"
            if module_short_name in target_modules:
                for param in module.parameters():
                    param.requires_grad = True
        
        print("\n=== Fully Trainable Modules ===")
        for name, module in model.named_modules():
            if any(param.requires_grad for param in module.parameters(recurse=False)):
                print(f"{name} ({type(module).__name__})")
        print(f"Total {len([n for n, m in model.named_modules() if any(p.requires_grad for p in m.parameters(recurse=False))])} trainable modules.")

    return model


def load_customized_llm_for_evaluation(args, custom_layer_cls):
    dtype = torch.bfloat16 if args.use_bfloat16 else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        args.llm_model,
        torch_dtype=dtype,
        device_map="balanced",
        low_cpu_mem_usage=True,
        trust_remote_code=True, 
    )

    _replace_layers_with_custom(model, custom_layer_cls, ts_in_len=args.in_len)

    if getattr(args, "lora_trainable", False):
        model = PeftModel.from_pretrained(model, args.lora_adapter_path)
    else:
        sd = _load_safetensors_state_dict(args.lora_adapter_path)
   
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            print(f"[load] Missing keys: {len(missing)} (showing first 10) -> {missing[:10]}")
        if unexpected:
            print(f"[load] Unexpected keys: {len(unexpected)} (showing first 10) -> {unexpected[:10]}")

    model.eval()

    return model



# Other helpful functions
def check_nan_inf(tensor: torch.Tensor, name: str) -> bool:
    nan_count = torch.isnan(tensor).sum().item()
    inf_count = torch.isinf(tensor).sum().item()
    if nan_count > 0 or inf_count > 0:
        print(f"[Warn] {name} contains {nan_count} NaN and {inf_count} Inf.")
        return True
    return False

def count_parameters(model: nn.Module, only_trainable: bool = False) -> float:
    """Return parameter count in Millions."""
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    else:
        return sum(p.numel() for p in model.parameters()) / 1e6


# Copied from transformers.models.mixtral.modeling_mixtral.load_balancing_loss_func
def load_balancing_loss_func(
    gate_logits: Union[torch.Tensor, Tuple[torch.Tensor], None],
    num_experts: Optional[int] = None,
    top_k=2,
    attention_mask: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, int]:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        gate_logits:
            Logits from the `gate`, should be a tuple of model.config.num_hidden_layers tensors of
            shape [batch_size X sequence_length, num_experts].
        num_experts:
            Number of experts
        top_k:
            The number of experts to route per-token, can be also interpreted as the `top-k` routing
            parameter.
        attention_mask (`torch.Tensor`, *optional*):
            The attention_mask used in forward function
            shape [batch_size X sequence_length] if not None.

    Returns:
        The auxiliary loss.
    """
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    if isinstance(gate_logits, tuple):
        compute_device = gate_logits[0].device
        concatenated_gate_logits = torch.cat([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0)

    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)

    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

    if attention_mask is None:
        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.mean(routing_weights, dim=0)
    else:
        batch_size, sequence_length = attention_mask.shape
        num_hidden_layers = concatenated_gate_logits.shape[0] // (batch_size * sequence_length)

        # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask
        expert_attention_mask = (
            attention_mask[None, :, :, None, None]
            .expand((num_hidden_layers, batch_size, sequence_length, top_k, num_experts))
            .reshape(-1, top_k, num_experts)
            .to(compute_device)
        )

        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.sum(expert_mask.float() * expert_attention_mask, dim=0) / torch.sum(
            expert_attention_mask, dim=0
        )

        # Compute the mask that masks all padding tokens as 0 with the same shape of tokens_per_expert
        router_per_expert_attention_mask = (
            attention_mask[None, :, :, None]
            .expand((num_hidden_layers, batch_size, sequence_length, num_experts))
            .reshape(-1, num_experts)
            .to(compute_device)
        )

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.sum(routing_weights * router_per_expert_attention_mask, dim=0) / torch.sum(
            router_per_expert_attention_mask, dim=0
        )

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts



# retrive certain layers

def get_base_model(m):
    return getattr(m, "model", None) or getattr(m, "transformer", None) or m

def get_transformer_backbone(m):
    """
    Robustly unwrap PEFT/DS/FSDP wrappers to get the transformer backbone
    that returns BaseModelOutputWithPast(last_hidden_state=...).
    Works for PeftModelForCausalLM(LoraModel(Qwen2MoeForCausalLM(...))).
    """
    if hasattr(m, "get_base_model"):
        try:
            m = m.get_base_model()  # e.g., Qwen2MoeForCausalLM
        except Exception:
            pass

    # 2) some models has LoRA PeftModelForCausalLM(base_model=LoraModel(model=...))
    if hasattr(m, "base_model") and hasattr(m.base_model, "model"):
        m = m.base_model.model  # -> Qwen2MoeForCausalLM

    # 3) from *ForCausalLM take transformer backbone
    backbone = getattr(m, "model", None) or getattr(m, "transformer", None)
    if backbone is None:
        backbone = m

    return backbone



def _find_decoder_layers(model):
    """
    """
    m = getattr(model, "model", model)
    for attr in ["layers", "h", "decoder"]:
        if hasattr(m, attr):
            obj = getattr(m, attr)
            if attr == "decoder" and hasattr(obj, "layers"):
                return obj.layers
            if isinstance(obj, (list, torch.nn.ModuleList)):
                return obj
    raise RuntimeError("Can't identify decoder layers, please look at LLM structure")

def _replace_layers_with_custom(model, custom_layer_cls, ts_in_len):
    layers = _find_decoder_layers(model)
    ref_param = next(model.parameters())
    dev, dt = ref_param.device, ref_param.dtype

    for i in range(len(layers)):
        old_layer = layers[i]
        new_layer = custom_layer_cls(model.config, i, ts_in_len).to(dev).to(dt)

        with torch.no_grad():
            old_sd = old_layer.state_dict()
            new_sd = new_layer.state_dict()
            for k, v in old_sd.items():
                if k in new_sd and new_sd[k].shape == v.shape:
                    new_sd[k].copy_(v)

        layers[i] = new_layer
        del old_layer

    torch.cuda.empty_cache()

def _load_safetensors_state_dict(ckpt_dir):
    """
    collect all .safetensors files and turn the into a state_dict
    """
    # usual file names:
    # - pytorch_model.safetensors / model.safetensors / adapter_model.safetensors
    # - model-00001-of-000XX.safetensors
    files = []
    for fname in os.listdir(ckpt_dir):
        if fname.endswith(".safetensors"):
            files.append(os.path.join(ckpt_dir, fname))
    if not files:
        raise FileNotFoundError(f"No .safetensors found in {ckpt_dir}")

    merged = {}
    for f in sorted(files):
        part = safe_load_file(f)
        for k, v in part.items():
            merged[k] = v
    return merged

