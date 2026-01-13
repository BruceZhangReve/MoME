import os
os.environ.setdefault("WANDB_MODE", "offline")

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import math
import wandb
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from tqdm import tqdm 
from transformers import AutoTokenizer, AutoModelForCausalLM

from data.dataloader import build_loader_from_saved
from torch.utils.data import DataLoader
from data.dataset import (WeatherDataset, FinanceDataset, EnvironmentDataset, EnergyDataset,
                          HealthUSDataset, HealthAFRDataset, SocialGoodDataset)
from TS_Encoder.dlinear_patch import TimeSeriesPatchEncoder
from TS_Encoder.mome import MoMe
from llm.layers import AlignLayer, RegressionHead, ClassificationHead
from llm.utils import (
    check_nan_inf,
    count_parameters,
    load_balancing_loss_func,
    build_llm_and_lora,
)
from utils import (
    get_transformer_backbone,
    normalize_timeseries, 
)


def fuse_ts_and_text(
    args,
    model: AutoModelForCausalLM,
    ts_encoder: nn.Module,
    align_layer: nn.Module,
    batch,
    embed_device: torch.device,
    model_dtype: torch.dtype,
    mode: str = "train",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Return inputs_embeds, attention_mask, labels for LLM.

    Note: 
        ts: tensor [B, in_len]
        txt:  tensor [B, max_txt_len]
    """
    # Initial processing and move to embedding device
    normalized_ts, denorm_params = normalize_timeseries(batch['input_window'])
    batch['denorm_params'] = denorm_params
    ts = normalized_ts.to(embed_device)

    txt = batch['text_input_ids'].to(embed_device)
    txt_attn_mask = batch['text_attention_mask'].to(embed_device)
    
    # TS and Text encode
    #ts_embed, _, _ = ts_encoder(ts)  # [B, C, d_ts]
    if args.ts_encoder == 'time-moe':
        ts_encoder = get_transformer_backbone(ts_encoder)
        if args.new_architecture:
            out = ts_encoder(ts, use_cache=False) 
            ts_embed = out.last_hidden_state # [B, t, d_ts] or time-moe, a timestamp is a token. We look for the states at last hidden layer
        else:
            out = ts_encoder(ts, use_cache=False) 
            ts_embed = out.last_hidden_state # [B, t, d_ts] or time-moe, a timestamp is a token. We look for the states at last hidden layer
    elif args.ts_encoder == 'DLinearP':
        ts = ts.unsqueeze(-1) #[B, in_len] -> [B, in_len, 1], single channel for current model
        ts_embed = ts_encoder(ts) # [B, in_len, 1] -> [B, t, d]
    elif args.ts_encoder == 'MoMe':
        ts = ts.unsqueeze(1) #[B, in_len] -> [B, 1, in_len]
        assert ts_encoder.modulation == False
        ts_embed = ts_encoder(ts) # [B, 1, in_len] -> [B, 1*P, d]
    else:
        raise ValueError("Please specify a valid TS-Encoder!")

    #check_nan_inf(ts_embed, "Raw TS Embedding (pre-cat)")

    txt_embeds = model.get_input_embeddings()(txt)  # [B, t, d_ts] n proper device/dtype already

    if args.pre_align_clip and args.pre_align_clip > 0:
        ts_embed = ts_embed.clamp(min=-args.pre_align_clip, max=args.pre_align_clip)

    check_nan_inf(ts_embed, "right before align")

    # Align to LLM hidden dimension
    ts_embed = align_layer(ts_embed) #d_ts -> d
    
    #compare_tensor_difference(ts_embed, txt_embeds) # this is printing function

    # Ensure dtype consistency for concatenation
    ts_embed = ts_embed.to(model_dtype)

    check_nan_inf(ts_embed, "Aligned TS Embedding (pre-cat)")

    # Concat
    inputs_embeds = torch.cat([ts_embed, txt_embeds], dim=1)

    # Build masks/labels
    attn_mask = torch.cat([
        torch.ones(ts_embed.size()[:2], dtype=txt_attn_mask.dtype, device=embed_device),
        txt_attn_mask
    ], dim=1)

    return inputs_embeds, attn_mask, txt_attn_mask


def train_one_epoch(
    args,
    model,
    ts_encoder,
    align_layer,
    downstream_head,
    dataloader,
    criterion, # the main loss function to be passed in
    optimizer,
    global_step: int,
    loss_fp: Optional[str], # the file to log loss changes, fp means file_path
    embed_device: torch.device,
    model_dtype: torch.dtype,
    tokenizer=None
) -> Tuple[int, float, int]:
    # Switch to training mod (although some modules only partially trainable)
    model.train()
    ts_encoder.train()
    align_layer.train()
    downstream_head.train()

    ##### Label mapping for trend prediction #####
    RAW_TO_LABEL_5WAY = {
        "-2% ~ +2%": "Neutral",
        "+2% ~ +4%": "Growth-Oriented",
        ">+4%": "Bullish",
        "-2% ~ -4%": "Warning",
        "<-4%": "Bearish",
    }

    LABEL_5WAY_TO_ID = {
        "Bearish": 0,
        "Warning": 1,
        "Neutral": 2,
        "Growth-Oriented": 3,
        "Bullish": 4,
    }

    LABEL_5WAY_TO_3WAY = {
        "Bearish": "Negative",
        "Warning": "Negative",
        "Neutral": "Neutral",
        "Growth-Oriented": "Positive",
        "Bullish": "Positive",
    }

    LABEL_3WAY_TO_ID = {
        "Negative": 0,
        "Neutral": 1,
        "Positive": 2,
    }

    LABEL_WT_TO_ID = {
        "increasing": 0,
        "decreasing": 1,
        "stable": 2,
    }
    ##### End label mapping #####

    # Initialize loss related
    pbar = tqdm(dataloader, desc="train")
    epoch_total_loss = 0.0
    epoch_batches = 0
    epoch_correct = 0 # for accuracy
    epoch_total_samples = 0  # for accuracy


    num_experts = model.config.num_experts # for Qwen is (60)
    num_experts_per_tok = model.config.num_experts_per_tok  # for Qwen is (4), AKA topk
    router_aux_loss_coef = model.config.router_aux_loss_coef  # Router loss weight


    for batch in pbar:
        optimizer.zero_grad()
        
        if args.task == 'finance_trend_prediction':
            raw_labels = batch['output_trend']
            try:
                standardized_labels = [RAW_TO_LABEL_5WAY[label.strip()] for label in raw_labels]
                if args.finance_trend_choice == "5way":
                    label_ids = [LABEL_5WAY_TO_ID[label] for label in standardized_labels]
                elif args.finance_trend_choice == "3way":
                    standardized_labels = [LABEL_5WAY_TO_3WAY[label.strip()] for label in standardized_labels]
                    label_ids = [LABEL_3WAY_TO_ID[label] for label in standardized_labels]
                else:
                    raise ValueError("args.finance_trend_choice must be '3way' or '5way'")

                true_label = torch.tensor(label_ids, dtype=torch.long, device=embed_device)
            except KeyError as e:
                available = list(RAW_TO_LABEL_5WAY.keys())
                raise KeyError(f"Unknown label: {e}. Available raw labels: {available}")

            inputs_embeds, attn_mask, txt_attn_mask = fuse_ts_and_text(
                args, model, ts_encoder, align_layer, batch, embed_device, model_dtype, mode="train"
                )
            
            llm_outputs = model(inputs_embeds=inputs_embeds, 
                                attention_mask=attn_mask, 
                                output_router_logits=True, 
                                use_cache=False)
            final_hidden = llm_outputs.last_hidden_state  # let's try this
            final_hidden = final_hidden.to(embed_device, torch.float32)

            logits =  downstream_head(final_hidden) #[B, C]

            main_loss = criterion(logits, true_label)

            if llm_outputs.router_logits is not None:
                #print(llm_outputs.router_logits[-1].shape, num_experts, num_experts_per_tok, attn_mask.shape)
                if args.new_architecture:
                    # This is still not working !!!!!!!!!!!!
                    router_loss = load_balancing_loss_func(
                        gate_logits=llm_outputs.router_logits,
                        num_experts=num_experts,
                        top_k=num_experts_per_tok,
                        attention_mask=txt_attn_mask#attn_mask  # filter padding tokens, they are not included in the router loss
                    )
                else:
                    router_loss = load_balancing_loss_func(
                        gate_logits=llm_outputs.router_logits,
                        num_experts=num_experts,
                        top_k=num_experts_per_tok,
                        attention_mask=attn_mask  # filter padding tokens, they are not included in the router loss
                    )
            else:
                raise ValueError("model cannot output router_logits! Nee to set output_router_logits=True")
        
            total_loss = main_loss + router_aux_loss_coef * router_loss

            outputs = logits

            # === Compute accuracy ===
            preds = torch.argmax(logits, dim=-1)  # [B]
            correct = (preds == true_label).sum().item()
            epoch_correct += correct
            epoch_total_samples += true_label.size(0)
        
        elif args.task == 'weather_trend_prediction':
            #if args.weather_trend_choice == "past":
            #    raw_labels = compute_temperature_trend(batch['input_window'])
            #elif args.weather_trend_choice == "future":
            #    raw_labels = compute_temperature_trend(batch['input_window'], batch['output_window'])
            #else:
            #    raise KeyError("args.weather_trend_choice not supported!")
            if args.weather_trend_choice == "past":
                raw_labels = batch['input_trend']
            elif args.weather_trend_choice == "future":
                raw_labels = batch['output_trend']
            else:
                raise KeyError("args.weather_trend_choice not supported!")
            
            label_ids = [LABEL_WT_TO_ID[label] for label in raw_labels]
            true_label = torch.tensor(label_ids, dtype=torch.long, device=embed_device)


            inputs_embeds, attn_mask, txt_attn_mask = fuse_ts_and_text(
                args, model, ts_encoder, align_layer, batch, embed_device, model_dtype, mode="train"
                )
            
            llm_outputs = model(inputs_embeds=inputs_embeds, 
                                attention_mask=attn_mask, 
                                output_router_logits=True, 
                                use_cache=False)
            final_hidden = llm_outputs.last_hidden_state  # let's try this
            final_hidden = final_hidden.to(embed_device, torch.float32)

            logits =  downstream_head(final_hidden) #[B, C]

            main_loss = criterion(logits, true_label)

            if llm_outputs.router_logits is not None:
                #print(llm_outputs.router_logits[-1].shape, num_experts, num_experts_per_tok, attn_mask.shape)
                if args.new_architecture:
                    router_loss = load_balancing_loss_func(
                        gate_logits=llm_outputs.router_logits,
                        num_experts=num_experts,
                        top_k=num_experts_per_tok,
                        attention_mask=txt_attn_mask#attn_mask  # filter padding tokens, they are not included in the router loss
                    )
                else:
                    router_loss = load_balancing_loss_func(
                        gate_logits=llm_outputs.router_logits,
                        num_experts=num_experts,
                        top_k=num_experts_per_tok,
                        attention_mask=attn_mask  # filter padding tokens, they are not included in the router loss
                    )
            else:
                raise ValueError("model cannot output router_logits! Nee to set output_router_logits=True")
        
            total_loss = main_loss + router_aux_loss_coef * router_loss

            outputs = logits

            # === Compute accuracy ===
            preds = torch.argmax(logits, dim=-1)  # [B]
            correct = (preds == true_label).sum().item()
            epoch_correct += correct
            epoch_total_samples += true_label.size(0)

        elif args.task in ['finance_forecast', 'weather_forecast']:
            inputs_embeds, attn_mask, txt_attn_mask = fuse_ts_and_text(
                args, model, ts_encoder, align_layer, batch, embed_device, model_dtype, mode="train"
                )
            mean, std = batch['denorm_params']
            true_seq = (batch['output_window'] - mean) / std  # [B, out_len]
            true_seq = true_seq.to(embed_device) #[B, out_len]

            check_nan_inf(inputs_embeds, "input_embeds")

            llm_outputs = model(inputs_embeds=inputs_embeds, 
                                attention_mask=attn_mask, 
                                output_router_logits=True, 
                                use_cache=False)
            final_hidden = llm_outputs.last_hidden_state 
            final_hidden = final_hidden.to(embed_device, true_seq.dtype)

            check_nan_inf(final_hidden, "final_hidden")
            
            #print(final_hidden.dtype, downstream_head.proj.weight.dtype, model.dtype, true_seq.dtype)

            outputs =  downstream_head(final_hidden) #[B, out_len]

            main_loss = criterion(outputs, true_seq)

            if llm_outputs.router_logits is not None:
                #print(llm_outputs.router_logits[-1].shape, num_experts, num_experts_per_tok, attn_mask.shape)
                if args.new_architecture:
                    router_loss = load_balancing_loss_func(
                        gate_logits=llm_outputs.router_logits,
                        num_experts=num_experts,
                        top_k=num_experts_per_tok,
                        attention_mask=txt_attn_mask#attn_mask  # filter padding tokens, they are not included in the router loss
                    )
                else:
                    router_loss = load_balancing_loss_func(
                        gate_logits=llm_outputs.router_logits,
                        num_experts=num_experts,
                        top_k=num_experts_per_tok,
                        attention_mask=attn_mask  # filter padding tokens, they are not included in the router loss
                    )
                    #print(router_loss)
            else:
                raise ValueError("model cannot output router_logits! Nee to set output_router_logits=True")
        
            total_loss = main_loss + router_aux_loss_coef * router_loss.to(model_dtype)

        elif args.task in ['energy_forecast', 'environment_forecast', 'healthus_forecast', 'healthafr_forecast', 'socialgood_forecast']:
            inputs_embeds, attn_mask, txt_attn_mask = fuse_ts_and_text(
                args, model, ts_encoder, align_layer, batch, embed_device, model_dtype, mode="train"
                )
            mean, std = batch['denorm_params']
            true_seq = (batch['output_window'] - mean) / std  # [B, out_len]
            true_seq = true_seq.to(embed_device) #[B, out_len]

            check_nan_inf(inputs_embeds, "input_embeds")

            llm_outputs = model(inputs_embeds=inputs_embeds, 
                                attention_mask=attn_mask, 
                                output_router_logits=True, 
                                use_cache=False)
            final_hidden = llm_outputs.last_hidden_state 
            final_hidden = final_hidden.to(embed_device, true_seq.dtype)

            check_nan_inf(final_hidden, "final_hidden")
            
            #print(final_hidden.dtype, downstream_head.proj.weight.dtype, model.dtype, true_seq.dtype)

            outputs =  downstream_head(final_hidden) #[B, out_len]

            main_loss = criterion(outputs, true_seq)

            if llm_outputs.router_logits is not None:
                #print(llm_outputs.router_logits[-1].shape, num_experts, num_experts_per_tok, attn_mask.shape)
                if args.new_architecture:
                    router_loss = load_balancing_loss_func(
                        gate_logits=llm_outputs.router_logits,
                        num_experts=num_experts,
                        top_k=num_experts_per_tok,
                        attention_mask=txt_attn_mask#attn_mask  # filter padding tokens, they are not included in the router loss
                    )
                else:
                    router_loss = load_balancing_loss_func(
                        gate_logits=llm_outputs.router_logits,
                        num_experts=num_experts,
                        top_k=num_experts_per_tok,
                        attention_mask=attn_mask  # filter padding tokens, they are not included in the router loss
                    )
                    #print(router_loss)
            else:
                raise ValueError("model cannot output router_logits! Nee to set output_router_logits=True")
        
            total_loss = main_loss + router_aux_loss_coef * router_loss.to(model_dtype)
        else:
            raise ValueError(f"Unsupported task: {args.task}")


        ### Back Propagate and Parameter Update ###

        #print(total_loss.dtype, router_loss.dtype, main_loss.dtype)
        total_loss.backward()
        # grad clip for stability
        if args.grad_clip_norm and args.grad_clip_norm > 0:
            # Obtain all trainable parameters directly from the optimizer (it is the same as the one defined in the main())
            torch.nn.utils.clip_grad_norm_(
                parameters=optimizer.param_groups[0]['params'], 
                max_norm=args.grad_clip_norm
            )
        optimizer.step()

#######if this is really trained?#######
        #ts_gate = model.layers[0].mlp.ts_gate
        #print("ts_gate.1 gradient:", ts_gate.weight.requires_grad)  
        #print("ts_gate.1 gradient:", ts_gate.weight.grad)
        #ts_gate = model.layers[0].mlp.ts_gate.modules_to_save["default"]
        #print("ts_gate.1 gradient:", ts_gate.weight.requires_grad)  
        #print("ts_gate.1 gradient:", ts_gate.weight.grad)  
        #ts_gate = model.layers[-1].mlp.ts_gate.modules_to_save["default"]
        #print("ts_gate.-1 gradient:", ts_gate.weight.requires_grad)  
        #print("ts_gate.-1 gradient:", ts_gate.weight.grad)  
        #if args.new_architecture:
        #    ts_gate_loraA = model.layers[-1].mlp.ts_gate.lora_A.default
        #    print("ts_gate_loraA.-1 gradient:", ts_gate_loraA.weight.requires_grad)  
        #    print("ts_gate_loraA.-1 gradient:", ts_gate_loraA.weight.grad) 

        #shared_gate_loraA = model.layers[-1].mlp.shared_expert_gate.lora_A.default
        #print("shared_gate_loraA.-1 gradient:", shared_gate_loraA.weight.requires_grad)  
        #print("shared_gate_loraA.-1 gradient:", shared_gate_loraA.weight.grad)  

        #shared_gate = model.layers[5].mlp.shared_expert_gate
        #print("shared_gate.5 gradient:", shared_gate.weight.requires_grad)  
        #print("shared_gate.5 gradient:", shared_gate.weight.grad)  

        #ts_gate = model.layers[5].mlp.ts_gate
        #print("ts_gate.5 gradient:", ts_gate.weight.requires_grad)  
        #print("ts_gate.5 gradient:", ts_gate.weight.grad)  

        #print("align layer gradient:",align_layer.linear.weight.grad)
        #print("head_grad:", downstream_head.proj.weight.grad)
#######if this is really trained?#######

        ### logging ###
        global_step += 1
        loss_item = total_loss.item()
        epoch_total_loss += loss_item
        epoch_batches += 1

        if args.task in ['finance_trend_prediction', 'weather_trend_prediction']:
            pbar.set_postfix(total_loss=f"{loss_item:.4f}", acc=f"{correct / len(true_label):.4f}")
        else:
            pbar.set_postfix(total_loss=f"{loss_item:.4f}")

        wandb.log({"train/loss": loss_item, "step": global_step})

        # Save to file
        if loss_fp is not None:
            with open(loss_fp, "a") as f:
                f.write(f"train_step {global_step}\ttrain_loss {loss_item:.4f}")
                if args.task in ['finance_trend_prediction', 'weather_trend_prediction']:
                    f.write(f"\tCE_Loss {main_loss.item():.4f}\n")
                elif args.task in ['finance_forecast', 'weather_forecast', 'energy_forecast', 'environment_forecast', 'healthus_forecast', 'healthafr_forecast', 'socialgood_forecast']:
                    f.write(f"\tMSE {main_loss.item():.4f}\n")
                else:
                    raise KeyError("args.task not defined!")

    # === After loop: compute average accuracy for classification ===
    avg_accuracy = 0.0
    if args.task in ['finance_trend_prediction', 'weather_trend_prediction'] and epoch_total_samples > 0:
        avg_accuracy = epoch_correct / epoch_total_samples
        print(f"\n[Epoch Train] Accuracy: {avg_accuracy:.4f} ({epoch_correct}/{epoch_total_samples})")
        wandb.log({"train/accuracy": avg_accuracy})  # Log once per epoch
    else:
        pass

    return global_step, epoch_total_loss, epoch_batches


# -------------------- Build Small Modules --------------------
def build_ts_and_align(args, embed_device: torch.device, target_hidden: int):
    #ts_encoder = iTransformerEncoder(
    #    args.n_vars, args.seq_len_channel, args.hidden_dim,
    #    args.dropout, args.topk, args.moe_mode,
    #    args.n_experts, args.n_heads, args.n_layers
    #).to(embed_device)
    #ts_encoder = TimeSeriesPatchEncoder(patch_len=64, hidden_dim=args.hidden_dim).to(embed_device)

    if args.ts_encoder == 'time-moe':
        args.hidden_dim = 384 #the hidden dimension for pretrained time-moe
        ts_token_num = args.in_len
        ts_encoder_path = "./TS_Encoder/TimeMoE-50M"
        ts_encoder = AutoModelForCausalLM.from_pretrained(
            ts_encoder_path, trust_remote_code=True
            )
        ts_encoder = ts_encoder.to(embed_device)
        for param in ts_encoder.parameters():
            param.requires_grad = False
    elif args.ts_encoder == 'DLinearP':
        ts_token_num = args.in_len
        ts_encoder = TimeSeriesPatchEncoder(patch_len=args.patch_len, hidden_dim=args.hidden_dim).to(embed_device)
        for param in ts_encoder.parameters():
            param.requires_grad = True
    elif args.ts_encoder == 'MoMe':
        ts_encoder = MoMe(in_len=args.in_len, patch_len=args.patch_len, hidden_dim=args.hidden_dim, 
                          top_k=args.topk, num_experts=args.n_experts, modulation=False).to(embed_device)
        ts_token_num = ts_encoder.patch_num
        for param in ts_encoder.parameters():
            param.requires_grad = True
    else:
        raise ValueError("Invalid Time Series Encoder!")

    align_layer = AlignLayer(args.hidden_dim, target_hidden).to(embed_device)

    txt_token_num = args.max_text_length
    token_num = ts_token_num + txt_token_num

    if args.task in ['finance_trend_prediction', 'weather_trend_prediction']:
        if args.task == 'finance_trend_prediction':
            if args.finance_trend_choice == '3way':
                num_class = 3
            elif args.finance_trend_choice == '5way':
                num_class = 5
            else:
                raise ValueError("args.finance_trend_choice is either 3way or 5 way")
            
            downstream_head = ClassificationHead(target_hidden, ts_token_num, num_class).to(embed_device)

        elif args.task == 'weather_trend_prediction':
            num_class = 3
            downstream_head = ClassificationHead(target_hidden, ts_token_num, num_class).to(embed_device)
        else:
            raise KeyError(f"Please specify a valid args.task!")

    elif args.task in ['finance_forecast', 'weather_forecast', 'energy_forecast', 'environment_forecast', 'healthus_forecast', 'healthafr_forecast', 'socialgood_forecast']:
        downstream_head = RegressionHead(target_hidden, ts_token_num, args.out_len).to(embed_device)
    
    else:
        raise KeyError(f"Please specify a valid args.task!")


    return ts_encoder, align_layer, downstream_head



###########################################################################


# -------------------- Argparse --------------------
def parse_args():
    parser = argparse.ArgumentParser()

    # -------- Model & Data --------
    parser.add_argument("--llm_model", default="./llm/Qwen1.5-MoE-A2.7B")
    parser.add_argument("--dataset_path", default="./data/processed/weather/aligned_in14days_out3days")
    parser.add_argument("--output_dir", default="output/debug")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.0)

    parser.add_argument("--data_pkl_dir", type=str, default="./data/saved_datasets/weather_forecasting")
    parser.add_argument("--data_suffix", type=str, default="in14_out3")


    # -------- Train Hyperparameter --------
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)

    # -------- LoRA --------
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    #parser.add_argument("--lora_modules_to_save", default=["ts_gate"])
    #parser.add_argument("--lora_modules", default="shared_expert_gate,ts_gate") #trainable modules, can be lora/fully trainable dependingon agrs.lora_trainable
    parser.add_argument("--lora_modules", default="q_proj,k_proj,v_proj,o_proj")
    #parser.add_argument("--lora_expert_layer_indices", default=[0,1])

    # -------- TS-Encoder & MoE --------
    parser.add_argument("--patch_len", type=int, default=4)
    parser.add_argument("--ts_encoder", type=str, default="time-moe")
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--n_vars", type=int, default=19)
    parser.add_argument("--seq_len_channel", type=int, default=128) #This might not be used anymore
    parser.add_argument("--hidden_dim", type=int, default=32) #note 384 is the embedding dimensionf for time-moe
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--n_experts", type=int, default=3)
    parser.add_argument("--topk", type=int, default=1)
    parser.add_argument("--moe_mode", default="multi_expert")  # multi_expert / No
    parser.add_argument("--n_heads", type=int, default=4)

    parser.add_argument("--in_len", type=int, default=336)
    parser.add_argument("--out_len", type=int, default=72)
    parser.add_argument("--max_text_length", type=int, default=2048)

    # -------- Numerical Safety & Dtypes --------
    parser.add_argument("--pre_align_clip", type=float, default=10, #1e5
                        help="If > 0, clamp TS embedding to [-clip, clip] BEFORE AlignLayer (both train & test).")
    parser.add_argument("--use_bfloat16", action="store_true",
                        help="Load LLM in bfloat16 (if supported) instead of float16.")

    # -------- Others --------
    parser.add_argument("--project_name", default="Fusion")
    parser.add_argument("--scale", type=bool, default=False)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", type=bool, default=True)

    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--new_architecture", action="store_true")
    parser.add_argument("--task", type=str, default="weather_forecast")
    parser.add_argument("--finance_trend_choice", type=str, default="3way") #'3way' or '5way'
    parser.add_argument("--weather_trend_choice", type=str, default="future") #'past' or 'future'
    parser.add_argument("--lora_trainable", action="store_true")

    return parser.parse_args()


# -------------------- Main --------------------
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    wandb.init(project=args.project_name, config=vars(args))


    tokenizer = AutoTokenizer.from_pretrained(args.llm_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id  

    if args.task in ["weather_forecast", "weather_trend_prediction"]:
        dataset = WeatherDataset(
            data_dir=args.dataset_path,
            tokenizer=tokenizer,
            input_seq_len=args.in_len,
            output_seq_len=args.out_len,
            max_text_length=args.max_text_length
        )
    elif (args.task == "finance_forecast") or (args.task == "finance_trend_prediction"):
        dataset = FinanceDataset(
            data_dir=args.dataset_path,
            tokenizer=tokenizer,
            input_len=args.in_len,
            output_len=args.out_len, 
            max_text_length=2048
            )
    elif args.task == "finance_indicator_prediction":
        raise NotImplementedError("Unknow Evaluation Task")

    elif args.task == "environment_forecast":
        dataset = EnvironmentDataset(
            data_dir=args.dataset_path,
            tokenizer=tokenizer,
            input_len=7,
            output_len=1, 
            max_text_length=512
            )

    elif args.task == "energy_forecast":
        dataset = EnergyDataset(
            data_dir=args.dataset_path,
            tokenizer=tokenizer,
            input_len=args.in_len,
            output_len=args.out_len, 
            max_text_length=512
            )
    elif args.task == "healthus_forecast":
        dataset = HealthUSDataset(
            data_dir=args.dataset_path,
            tokenizer=tokenizer,
            input_len=args.in_len,
            output_len=args.out_len, 
            max_text_length=512
            )
    elif args.task == "healthafr_forecast":
        dataset = HealthAFRDataset(
            data_dir=args.dataset_path,
            tokenizer=tokenizer,
            input_len=args.in_len,
            output_len=args.out_len, 
            max_text_length=512
            )
    elif args.task == "socialgood_forecast":
        dataset = SocialGoodDataset(
            data_dir=args.dataset_path,
            tokenizer=tokenizer,
            input_len=args.in_len,
            output_len=args.out_len, 
            max_text_length=512
            )
    else:
        raise NotImplementedError("Unknow Evaluation Task")

    if args.task in ['environment_forecast', 'energy_forecast']:
        # Dataset from TimeMMD
        train_loader = DataLoader(dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers
                                  )
    else: 
        train_loader, _, _ = build_loader_from_saved(dataset=dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                                     train_ratio=args.train_ratio,val_ratio=args.val_ratio, seed=args.seed, 
                                                     save_dir=args.data_pkl_dir, suffix=args.data_suffix)

    # build LLM + LoRA
    model = build_llm_and_lora(args) #old

    model_dtype = model.dtype
    embed_device = model.get_input_embeddings().weight.device

    # TS encoder + align layer + downstream head
    ts_encoder, align_layer, downstream_head = build_ts_and_align(
        args, embed_device, model.config.hidden_size
    )
    
    # Loss function
    if args.task in ['finance_trend_prediction', 'weather_trend_prediction']:
        criterion = nn.CrossEntropyLoss() 
    elif args.task in ['finance_forecast', 'weather_forecast', 'environment_forecast', 'energy_forecast', 'healthus_forecast', 'healthafr_forecast', 'socialgood_forecast']:
        criterion = nn.MSELoss()
    else:
        raise KeyError(f"Please specify a valid args.task!")


    # optimizer (ts + align + Head + LoRA/llm params)
    if args.ts_encoder == 'time-moe':
        trainable_params = (
            list(align_layer.parameters()) +
            list(downstream_head.parameters()) +
            [p for p in model.parameters() if p.requires_grad]
        )
    else:
        trainable_params = (
            list(ts_encoder.parameters()) +
            list(align_layer.parameters()) +
            list(downstream_head.parameters()) +
            [p for p in model.parameters() if p.requires_grad]
        )

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    

    # Parameter stats
    llm_total = count_parameters(model, only_trainable=False)
    llm_trainable = count_parameters(model, only_trainable=True)
    ts_total = count_parameters(ts_encoder, only_trainable=False)
    ts_trainable = count_parameters(ts_encoder, only_trainable=True)
    align_total = count_parameters(align_layer, only_trainable=False)
    align_trainable = count_parameters(align_layer, only_trainable=True)
    head_total = count_parameters(downstream_head, only_trainable=False)
    head_trainable = count_parameters(downstream_head, only_trainable=True)
    total_params = llm_total + ts_total + align_total + head_total
    total_trainable = llm_trainable + ts_trainable + align_trainable + head_trainable

    print("\n===== Parameter Statistics =====")
    print(f"LLM total parameter: {llm_total:.2f} M")
    print(f"LLM trainable parameter (LoRA): {llm_trainable:.2f} M")
    print(f"TS-Encoder total parameter: {ts_total:.2f} M")
    print(f"Align-Layer total parameter: {align_total:.2f} M")
    print(f"Downstream Head total parameter: {head_trainable:.2f} M")
    print("---------------------")
    print(f"Total number of parameters: {total_params:.2f} M")
    print(f"Total number of trainable parameters: {total_trainable:.2f} M")
    print(f"Trainable parameter's proportion: {total_trainable / max(total_params, 1e-9) * 100:.4f}%")
    print("=====================\n")

    # loss file
    loss_fpath = os.path.join(args.output_dir, "loss.txt")
    with open(loss_fpath, "w") as f:
        f.write("")


    # training loop
    global_step = 0
    for epoch in range(args.epochs):
        global_step, epoch_total_loss, epoch_batches = train_one_epoch(
            args, model, ts_encoder, align_layer, downstream_head, train_loader, criterion,
            optimizer, global_step, loss_fpath, embed_device, model_dtype, tokenizer=tokenizer
        )
        epoch_avg_loss = epoch_total_loss / max(epoch_batches, 1)
        print(f"Epoch {epoch} Avg Loss: {epoch_avg_loss:.4f}")
        wandb.log({"train/epoch_avg_loss": epoch_avg_loss, "epoch": epoch})

        # save checkpoints (LoRA + TS/Align)
        model.save_pretrained(args.output_dir)
        torch.save({
            "ts_encoder": ts_encoder.state_dict(),
            "align_layer": align_layer.state_dict(),
            "downstream_head": downstream_head.state_dict(),
            "args": vars(args)
        }, os.path.join(args.output_dir, f"ts_encoder_epoch{epoch}.pt"))


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    main()