import os
os.environ.setdefault("WANDB_MODE", "offline")

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import json
import random
import argparse
import numpy as np
from tqdm import tqdm 
from typing import Tuple
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

from TS_Encoder.dlinear_patch import TimeSeriesPatchEncoder
from TS_Encoder.mome import MoMe

from torch.utils.data import DataLoader
from data.dataloader import build_loader_from_saved
from data.dataset import (WeatherDataset, FinanceDataset, EnvironmentDataset, EnergyDataset,
                          HealthUSDataset, HealthAFRDataset, SocialGoodDataset)

from llm.layers import AlignLayer, RegressionHead, ClassificationHead, RegressionHead_latefusion, ClassificationHead_latefusion
from llm.utils import load_llm_for_evaluation, check_nan_inf
from utils import (
    get_transformer_backbone,
    normalize_timeseries, 
    denormalize_timeseries, 
    calculate_mape,
    compute_temperature_trend
)

        
####################################################
# we directly copy those modules from train_classic.py
def get_ts_embed(
    args,
    ts_encoder: nn.Module,
    batch,
    embed_device: torch.device,
    model_dtype: torch.dtype,
) -> torch.Tensor:
    """Get time series embeddings without text fusion"""
    normalized_ts, denorm_params = normalize_timeseries(batch['input_window'])
    batch['denorm_params'] = denorm_params
    ts = normalized_ts.to(embed_device)

    if args.ts_encoder == 'time-moe':
        ts_encoder = get_transformer_backbone(ts_encoder)
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
    

    if args.pre_align_clip and args.pre_align_clip > 0:
        ts_embed = ts_embed.clamp(min=-args.pre_align_clip, max=args.pre_align_clip)
    
    #print("get_ts_embed", ts_embed.shape)
    
    return ts_embed.to(model_dtype)


def fuse_ts_and_text(
    args,
    model: AutoModelForCausalLM,
    ts_encoder: nn.Module,
    align_layer: nn.Module,
    batch,
    embed_device: torch.device,
    model_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Unified interface for fusion strategies.
    
    Returns:
        inputs_embeds: For early fusion, the fused embeddings; for late fusion, None
        attn_mask: For early fusion, the fused attention mask; for late fusion, None
        txt_embeds: Text embeddings (for late fusion)
        txt_attn_mask: Text attention mask
        ts_embed: Time series embeddings (for late fusion)

    Note: 
        ts: tensor [B, in_len]
        txt:  tensor [B, max_txt_len]
    """

    fusion_strategy = getattr(args, 'fusion_strategy', 'early_concat')

    # Get text embeddings
    txt = batch['text_input_ids'].to(embed_device)
    txt_attn_mask = batch['text_attention_mask'].to(embed_device)
    txt_embeds = model.get_input_embeddings()(txt)# [B, t, d_ts] n proper device/dtype already

    # Get time series embeddings
    ts_embed = get_ts_embed(args, ts_encoder, batch, embed_device, model_dtype)

    # Early fusion strategies
    if fusion_strategy.startswith('early_'):
        
        if fusion_strategy == 'early_concat':
            # Align TS to LLM hidden dimension
            ts_embed = align_layer(ts_embed)#d_ts -> d
            ts_embed = ts_embed.to(model_dtype)# Ensure dtype consistency for concatenation
            inputs_embeds = torch.cat([ts_embed, txt_embeds], dim=1)
            attn_mask = torch.cat([
                torch.ones(ts_embed.size()[:2], dtype=txt_attn_mask.dtype, device=embed_device),
                txt_attn_mask], dim=1)
            return inputs_embeds, attn_mask, None, None, ts_embed
        
        elif fusion_strategy == 'early_crossattn':
            #print(ts_embed.dtype, txt_embeds.dtype)

            txt_key_padding_mask = ~txt_attn_mask.bool() 
            ts_embed = align_layer[0](ts_embed) 

            attn_out, _ = align_layer[1](query=ts_embed,
                                         key=txt_embeds,
                                         value=txt_embeds,
                                         key_padding_mask=txt_key_padding_mask,
                                         need_weights=False
                                         )  # [B, L_ts, d_llm]

            ts_enhanced = ts_embed + attn_out
            #print("fuse_ts_and_txt:" ,ts_embed.shape, ts_enhanced.shape)
            
            inputs_embeds = torch.cat([ts_enhanced, txt_embeds], dim=1)
            attn_mask = torch.cat([
                torch.ones(ts_embed.size()[:2], dtype=txt_attn_mask.dtype, device=embed_device),
                txt_attn_mask], dim=1)
            
            return inputs_embeds, attn_mask, None, None, ts_embed
        else:
            raise ValueError(f"Unsupported early fusion strategy: {fusion_strategy}")
        
    # Late fusion strategies - return separate embeddings
    elif fusion_strategy.startswith('late_'):
        return None, None, txt_embeds, txt_attn_mask, ts_embed

    else:
        raise ValueError(f"Unsupported fusion strategy: {fusion_strategy}")
####################################################
def build_ts_and_align(args, embed_device: torch.device, target_hidden: int):
    """
    target_hidden: The hidden dimension for the pretrained LLM
    """

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

    
        
    if args.fusion_strategy == 'late_crossattn':
        align_layer = nn.ModuleList([nn.Linear(args.hidden_dim, target_hidden),
                                     nn.MultiheadAttention(embed_dim=target_hidden,
                                                           num_heads=args.n_heads,
                                                           batch_first=True,
                                                           dropout=0.1
                                                           )]).to(embed_device)
        if args.use_bfloat16:
            align_layer = align_layer.to(torch.bfloat16)

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
    
    elif args.fusion_strategy == 'early_crossattn':
        #print(ts_token_num) right
        align_layer = nn.ModuleList([nn.Linear(args.hidden_dim, target_hidden),
                                     nn.MultiheadAttention(embed_dim=target_hidden,
                                                           num_heads=args.n_heads,
                                                           batch_first=True,
                                                           dropout=0.1
                                                           )]).to(embed_device)
        if args.use_bfloat16:
            align_layer = align_layer.to(torch.bfloat16)

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
    
    elif args.fusion_strategy in ['late_add', 'late_concat']:
        #print(ts_token_num) right
        align_layer = AlignLayer(args.hidden_dim, target_hidden).to(embed_device)
        if args.use_bfloat16:
            align_layer = align_layer.to(torch.bfloat16)

        if args.task in ['finance_trend_prediction', 'weather_trend_prediction']:
            if args.task == 'finance_trend_prediction':
                if args.finance_trend_choice == '3way':
                    num_class = 3
                elif args.finance_trend_choice == '5way':
                    num_class = 5
                else:
                    raise ValueError("args.finance_trend_choice is either 3way or 5 way")
            
                #downstream_head = ClassificationHead(target_hidden, ts_token_num, num_class).to(embed_device)
                if args.fusion_strategy == 'late_add':
                    downstream_head = ClassificationHead_latefusion(target_hidden, num_class).to(embed_device)
                else:
                    downstream_head = ClassificationHead_latefusion(2*target_hidden, num_class).to(embed_device)


            elif args.task == 'weather_trend_prediction':
                num_class = 3
                #downstream_head = ClassificationHead(target_hidden, ts_token_num, num_class).to(embed_device)
                if args.fusion_strategy == 'late_add':
                    downstream_head = ClassificationHead_latefusion(target_hidden, num_class).to(embed_device)
                else:
                    downstream_head = ClassificationHead_latefusion(2*target_hidden, num_class).to(embed_device)
            else:
                raise KeyError(f"Please specify a valid args.task!")

        elif args.task in ['finance_forecast', 'weather_forecast', 'energy_forecast', 'environment_forecast', 'healthus_forecast', 'healthafr_forecast', 'socialgood_forecast']:
            #downstream_head = RegressionHead(target_hidden, ts_token_num, args.out_len).to(embed_device)
            if args.fusion_strategy == 'late_add':
                downstream_head = RegressionHead_latefusion(target_hidden, num_class).to(embed_device)
            else:
                downstream_head = RegressionHead_latefusion(2*target_hidden, num_class).to(embed_device)
        else:
            raise KeyError(f"Please specify a valid args.task!")
        
        return ts_encoder, align_layer, downstream_head
    
    else:
        align_layer = AlignLayer(args.hidden_dim, target_hidden).to(embed_device)

        if args.use_bfloat16:
            align_layer = align_layer.to(torch.bfloat16)

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


def load_model_components(args):
    dtype = torch.bfloat16 if args.use_bfloat16 else torch.float32

    model = load_llm_for_evaluation(args)
    print(f"Finetuned Base-LLM Loaded")
    
    target_hidden = model.config.hidden_size
    embed_device = model.get_input_embeddings().weight.device

    print(f"Loading TS-Encoder: {args.ts_encoder}")
    ts_encoder, align_layer, downstream_head = build_ts_and_align(
            args, embed_device, model.config.hidden_size
        )
    ts_encoder.eval(); align_layer.eval(); downstream_head.eval()
    for param in ts_encoder.parameters():
        param.requires_grad = False
    for param in align_layer.parameters():
        param.requires_grad = False
    for param in downstream_head.parameters():
        param.requires_grad = False

    
    print(f"Load Pre-trained Parameters: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=embed_device)

    ts_encoder.load_state_dict(checkpoint["ts_encoder"])
    align_layer.load_state_dict(checkpoint["align_layer"])
    downstream_head.load_state_dict(checkpoint["downstream_head"])
    
    if args.use_checkpoint_args:
        print("Using the configurations in the checkpoint file.")
        checkpoint_args = checkpoint["args"]
        for key, value in checkpoint_args.items():
            if hasattr(args, key):
                setattr(args, key, value)
    
    return model, ts_encoder, align_layer, downstream_head, embed_device


    
@torch.no_grad()
def evaluate_full_test_set(
    args,
    model,
    ts_encoder,
    align_layer,
    downstream_head,
    test_loader,
    embed_device: torch.device,
    model_dtype: torch.dtype,
    tokenizer=None
) -> tuple[float, float, dict]:
    
    model.eval()
    ts_encoder.eval()
    align_layer.eval()
    downstream_head.eval()
    

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

    total_metrics = {
        'mse': 0.0,
        'mae': 0.0,
        'mape': 0.0,
        'acc': 0.0
    }
    total_samples = 0
    
    results = {
        "predictions": [],
        "ground_truth": [],
        "sample_indices": []
    }
    
    mse_criterion = torch.nn.MSELoss()
    mae_criterion = torch.nn.L1Loss()
    
    pbar = tqdm(test_loader, desc="Full Test Set Evaluation")

    model = get_transformer_backbone(model)

    if args.task == 'weather_forecast':

        for batch_idx, batch in enumerate(pbar):

            fusion_strategy = getattr(args, 'fusion_strategy', 'early_concat')
            inputs_embeds, attn_mask, txt_embeds, txt_attn_mask, ts_embed = fuse_ts_and_text(
                args, model, ts_encoder, align_layer, batch, embed_device, model_dtype
            )

            true_seq = batch['output_window'].to(embed_device)

            mean, std = batch['denorm_params'] #this is dynanically saved to batch in the fuse_ts_and_text function
            mean = mean.to(embed_device); std = std.to(embed_device)

            if fusion_strategy.startswith('early_'):
                # Early fusion: process fused embeddings through LLM
                llm_outputs = model(inputs_embeds=inputs_embeds, 
                                    attention_mask=attn_mask, 
                                    output_router_logits=False, 
                                    use_cache=False)
                final_hidden = llm_outputs.last_hidden_state  # let's try this
                final_hidden = final_hidden.to(embed_device, torch.float32)

            elif fusion_strategy.startswith('late_'):
                # Late fusion: process text and TS separately
                llm_outputs = model(
                    inputs_embeds=txt_embeds,
                    attention_mask=txt_attn_mask,
                    output_router_logits=False,
                    use_cache=False
                )
                text_features = llm_outputs.last_hidden_state  # [B, L_txt, d]
                text_features = text_features.to(embed_device, torch.float32)

                if fusion_strategy == 'late_concat':
                    text_pooled = text_features.mean(dim=1) # [B, d]
                    ts_pooled   = ts_embed.mean(dim=1)# [B, d_ts]
                    ts_pooled = align_layer(ts_pooled)  # [B, d] -> align to llm hidden dimension 
                    final_hidden = torch.cat([ts_pooled, text_pooled], dim=1)  # [B, 2d]
                elif fusion_strategy == 'late_add':
                    text_pooled = text_features.mean(dim=1) # [B, d]
                    ts_pooled   = ts_embed.mean(dim=1)# [B, d_ts]
                    ts_pooled = align_layer(ts_pooled)  # [B, d] -> align to LLM dim
                    final_hidden = ts_pooled + text_pooled # [B, d]
                elif fusion_strategy == 'late_crossattn':
                    txt_key_padding_mask = ~txt_attn_mask.bool() 
                    ts_embed = align_layer[0](ts_embed) 

                    attn_out, _ = align_layer[1](query=ts_embed,
                                                 key=txt_embeds,
                                                 value=txt_embeds,
                                                 key_padding_mask=txt_key_padding_mask,
                                                 need_weights=False
                                                 )  # [B, L_ts, d_llm]

                    final_hidden = ts_embed + attn_out #[B, L_ts, d_ts] poetntially bf16
                    final_hidden = final_hidden.to(torch.float32)
                    #print(final_hidden.shape, final_hidden.dtype)
                else:
                    raise ValueError(f"Unsupported late fusion strategy: {fusion_strategy}")
            
            outputs = downstream_head(final_hidden) #[B, out_len]

            pred_seq = outputs.to(embed_device, true_seq.dtype) # [B, out_len]

            pred_seq = denormalize_timeseries(pred_seq, (mean, std))

            batch_size = pred_seq.size(0)
            total_samples += batch_size

            results["predictions"].append(pred_seq.to(torch.float32).cpu().numpy())
            results["ground_truth"].append(true_seq.to(torch.float32).cpu().numpy())
            results["sample_indices"].extend([f"{batch_idx}_{i}" for i in range(batch_size)])

            batch_mse = mse_criterion(pred_seq, true_seq)
            batch_mae = mae_criterion(pred_seq, true_seq)
            
            total_metrics['mse'] += batch_mse.item() * batch_size
            total_metrics['mae'] += batch_mae.item() * batch_size
            
            pbar.set_postfix({
                "batch_MSE": f"{batch_mse.item():.4f}",
                "batch_MAE": f"{batch_mae.item():.4f}",
            })

        avg_metrics = {key: total / max(total_samples, 1) 
                       for key, total in total_metrics.items()
                       }
    
        results["predictions"] = np.concatenate(results["predictions"], axis=0)
        results["ground_truth"] = np.concatenate(results["ground_truth"], axis=0)
    
        print("\n" + "="*40)
        print(f"Full Test Set Evaluation Results for {args.task}")
        print(f"Total Samples: {total_samples}")

        print(f"Average MSE: {avg_metrics['mse']:.6f}")
        print(f"Average MAE: {avg_metrics['mae']:.6f}")
        return avg_metrics['mse'], avg_metrics['mae'], results
    
    elif args.task == 'finance_forecast':

        for batch_idx, batch in enumerate(pbar):

            fusion_strategy = getattr(args, 'fusion_strategy', 'early_concat')
            inputs_embeds, attn_mask, txt_embeds, txt_attn_mask, ts_embed = fuse_ts_and_text(
                args, model, ts_encoder, align_layer, batch, embed_device, model_dtype
            )
            
            true_seq = batch['output_window'].to(embed_device)
            mean, std = batch['denorm_params'] #this is dynanically saved to batch in the fuse_ts_and_text function
            mean = mean.to(embed_device); std = std.to(embed_device)

            if fusion_strategy.startswith('early_'):
                # Early fusion: process fused embeddings through LLM
                llm_outputs = model(inputs_embeds=inputs_embeds, 
                                    attention_mask=attn_mask, 
                                    output_router_logits=False, 
                                    use_cache=False)
                final_hidden = llm_outputs.last_hidden_state  # let's try this
                final_hidden = final_hidden.to(embed_device, torch.float32)

            elif fusion_strategy.startswith('late_'):
                # Late fusion: process text and TS separately
                llm_outputs = model(
                    inputs_embeds=txt_embeds,
                    attention_mask=txt_attn_mask,
                    output_router_logits=False,
                    use_cache=False
                )
                text_features = llm_outputs.last_hidden_state  # [B, L_txt, d]
                text_features = text_features.to(embed_device, torch.float32)

                if fusion_strategy == 'late_concat':
                    text_pooled = text_features.mean(dim=1) # [B, d]
                    ts_pooled   = ts_embed.mean(dim=1)# [B, d_ts]
                    ts_pooled = align_layer(ts_pooled)  # [B, d] -> align to llm hidden dimension 
                    final_hidden = torch.cat([ts_pooled, text_pooled], dim=1)  # [B, 2d]
                elif fusion_strategy == 'late_add':
                    text_pooled = text_features.mean(dim=1) # [B, d]
                    ts_pooled   = ts_embed.mean(dim=1)# [B, d_ts]
                    ts_pooled = align_layer(ts_pooled)  # [B, d] -> align to LLM dim
                    final_hidden = ts_pooled + text_pooled # [B, d]
                elif fusion_strategy == 'late_crossattn':
                    txt_key_padding_mask = ~txt_attn_mask.bool() 
                    ts_embed = align_layer[0](ts_embed) 

                    attn_out, _ = align_layer[1](query=ts_embed,
                                                 key=txt_embeds,
                                                 value=txt_embeds,
                                                 key_padding_mask=txt_key_padding_mask,
                                                 need_weights=False
                                                 )  # [B, L_ts, d_llm]

                    final_hidden = ts_embed + attn_out #[B, L_ts, d_ts] poetntially bf16
                    final_hidden = final_hidden.to(torch.float32)
                    #print(final_hidden.shape, final_hidden.dtype)
                else:
                    raise ValueError(f"Unsupported late fusion strategy: {fusion_strategy}")
            
            outputs = downstream_head(final_hidden) #[B, out_len]

            pred_seq = outputs.to(embed_device, pred_seq.dtype) # [B, out_len]
            pred_seq = denormalize_timeseries(pred_seq, (mean, std))

            batch_size = pred_seq.size(0)
            total_samples += batch_size

            results["predictions"].append(pred_seq.to(torch.float32).cpu().numpy())
            results["ground_truth"].append(true_seq.to(torch.float32).cpu().numpy())
            results["sample_indices"].extend([f"{batch_idx}_{i}" for i in range(batch_size)])

            batch_mape = calculate_mape(pred_seq, true_seq)#calculate_mape(pred_seq, true_seq)
            batch_mae = mae_criterion(pred_seq, true_seq)
            
            total_metrics['mape'] += batch_mape.item() * batch_size
            total_metrics['mae'] += batch_mae.item() * batch_size
            
            pbar.set_postfix({
                "batch_MAPE": f"{batch_mape.item():.4f}",
                "batch_MAE": f"{batch_mae.item():.4f}",
            })

        avg_metrics = {key: total / max(total_samples, 1) 
                       for key, total in total_metrics.items()
                       }
    
        results["predictions"] = np.concatenate(results["predictions"], axis=0)
        results["ground_truth"] = np.concatenate(results["ground_truth"], axis=0)
    
        print("\n" + "="*40)
        print(f"Full Test Set Evaluation Results for {args.task}")
        print(f"Total Samples: {total_samples}")

        print(f"Average MAPE: {avg_metrics['mape']:.6f}")
        print(f"Average MAE: {avg_metrics['mae']:.6f}")
        return avg_metrics['mape'], avg_metrics['mae'], results

    elif args.task == 'finance_trend_prediction':

        for batch_idx, batch in enumerate(pbar):
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

            fusion_strategy = getattr(args, 'fusion_strategy', 'early_concat')
            inputs_embeds, attn_mask, txt_embeds, txt_attn_mask, ts_embed = fuse_ts_and_text(
                args, model, ts_encoder, align_layer, batch, embed_device, model_dtype
            )
            
            if fusion_strategy.startswith('early_'):
                # Early fusion: process fused embeddings through LLM
                llm_outputs = model(inputs_embeds=inputs_embeds, 
                                    attention_mask=attn_mask, 
                                    output_router_logits=False, 
                                    use_cache=False)
                final_hidden = llm_outputs.last_hidden_state  # let's try this
                final_hidden = final_hidden.to(embed_device, torch.float32)

            elif fusion_strategy.startswith('late_'):
                # Late fusion: process text and TS separately
                llm_outputs = model(
                    inputs_embeds=txt_embeds,
                    attention_mask=txt_attn_mask,
                    output_router_logits=False,
                    use_cache=False
                )
                text_features = llm_outputs.last_hidden_state  # [B, L_txt, d]
                text_features = text_features.to(embed_device, torch.float32)

                if fusion_strategy == 'late_concat':
                    text_pooled = text_features.mean(dim=1) # [B, d]
                    ts_pooled   = ts_embed.mean(dim=1)# [B, d_ts]
                    ts_pooled = align_layer(ts_pooled)  # [B, d] -> align to llm hidden dimension 
                    final_hidden = torch.cat([ts_pooled, text_pooled], dim=1)  # [B, 2d]
                elif fusion_strategy == 'late_add':
                    text_pooled = text_features.mean(dim=1) # [B, d]
                    ts_pooled   = ts_embed.mean(dim=1)# [B, d_ts]
                    ts_pooled = align_layer(ts_pooled)  # [B, d] -> align to LLM dim
                    final_hidden = ts_pooled + text_pooled # [B, d]
                elif fusion_strategy == 'late_crossattn':
                    txt_key_padding_mask = ~txt_attn_mask.bool() 
                    ts_embed = align_layer[0](ts_embed) 

                    attn_out, _ = align_layer[1](query=ts_embed,
                                                 key=txt_embeds,
                                                 value=txt_embeds,
                                                 key_padding_mask=txt_key_padding_mask,
                                                 need_weights=False
                                                 )  # [B, L_ts, d_llm]

                    final_hidden = ts_embed + attn_out #[B, L_ts, d_ts] poetntially bf16
                    final_hidden = final_hidden.to(torch.float32)
                    #print(final_hidden.shape, final_hidden.dtype)
                else:
                    raise ValueError(f"Unsupported late fusion strategy: {fusion_strategy}")

            logits = downstream_head(final_hidden)  # [B, C]
            
            batch_size = true_label.size(0)

            preds = torch.argmax(logits, dim=-1)  # [B]
            batch_correct = (preds == true_label).sum().item()

            results["sample_indices"].extend([f"{batch_idx}_{i}" for i in range(batch_size)])

            batch_acc =  batch_correct / batch_size

            total_samples += batch_size
            total_metrics['acc'] += batch_acc * batch_size
            
            pbar.set_postfix({
                "batch_Acc": f"{batch_acc:.4f}"
            })

        avg_metrics = {key: total / max(total_samples, 1) 
                       for key, total in total_metrics.items()
                       }
    
    
        print("\n" + "="*40)
        print(f"Full Test Set Evaluation Results for {args.task}")
        print(f"Total Samples: {total_samples}")

        print(f"Average Accuracy: {avg_metrics['acc']:.6f}")
        return avg_metrics['acc'], avg_metrics['acc'], results
    
    elif args.task == "weather_trend_prediction":
        for batch_idx, batch in enumerate(pbar):
            if args.weather_trend_choice == "past":
                raw_labels = compute_temperature_trend(batch['input_window'])
            elif args.weather_trend_choice == "future":
                raw_labels = compute_temperature_trend(batch['input_window'], batch['output_window'])
            else:
                raise KeyError("args.weather_trend_choice not supported!")
            label_ids = [LABEL_WT_TO_ID[label] for label in raw_labels]
            true_label = torch.tensor(label_ids, dtype=torch.long, device=embed_device)


            fusion_strategy = getattr(args, 'fusion_strategy', 'early_concat')
            inputs_embeds, attn_mask, txt_embeds, txt_attn_mask, ts_embed = fuse_ts_and_text(
                args, model, ts_encoder, align_layer, batch, embed_device, model_dtype
            )
            
            if fusion_strategy.startswith('early_'):
                # Early fusion: process fused embeddings through LLM
                llm_outputs = model(inputs_embeds=inputs_embeds, 
                                    attention_mask=attn_mask, 
                                    output_router_logits=False, 
                                    use_cache=False)
                final_hidden = llm_outputs.last_hidden_state  # let's try this
                final_hidden = final_hidden.to(embed_device, torch.float32)

            elif fusion_strategy.startswith('late_'):
                # Late fusion: process text and TS separately
                llm_outputs = model(
                    inputs_embeds=txt_embeds,
                    attention_mask=txt_attn_mask,
                    output_router_logits=False,
                    use_cache=False
                )
                text_features = llm_outputs.last_hidden_state  # [B, L_txt, d]
                text_features = text_features.to(embed_device, torch.float32)

                if fusion_strategy == 'late_concat':
                    text_pooled = text_features.mean(dim=1) # [B, d]
                    ts_pooled   = ts_embed.mean(dim=1)# [B, d_ts]
                    ts_pooled = align_layer(ts_pooled)  # [B, d] -> align to llm hidden dimension 
                    final_hidden = torch.cat([ts_pooled, text_pooled], dim=1)  # [B, 2d]
                elif fusion_strategy == 'late_add':
                    text_pooled = text_features.mean(dim=1) # [B, d]
                    ts_pooled   = ts_embed.mean(dim=1)# [B, d_ts]
                    ts_pooled = align_layer(ts_pooled)  # [B, d] -> align to LLM dim
                    final_hidden = ts_pooled + text_pooled # [B, d]
                elif fusion_strategy == 'late_crossattn':
                    txt_key_padding_mask = ~txt_attn_mask.bool() 
                    ts_embed = align_layer[0](ts_embed) 

                    attn_out, _ = align_layer[1](query=ts_embed,
                                                 key=txt_embeds,
                                                 value=txt_embeds,
                                                 key_padding_mask=txt_key_padding_mask,
                                                 need_weights=False
                                                 )  # [B, L_ts, d_llm]

                    final_hidden = ts_embed + attn_out #[B, L_ts, d_ts] poetntially bf16
                    final_hidden = final_hidden.to(torch.float32)
                    #print(final_hidden.shape, final_hidden.dtype)
                else:
                    raise ValueError(f"Unsupported late fusion strategy: {fusion_strategy}")

            logits = downstream_head(final_hidden)  # [B, C]
            
            batch_size = true_label.size(0)

            preds = torch.argmax(logits, dim=-1)  # [B]
            batch_correct = (preds == true_label).sum().item()

            results["sample_indices"].extend([f"{batch_idx}_{i}" for i in range(batch_size)])

            batch_acc =  batch_correct / batch_size

            total_samples += batch_size
            total_metrics['acc'] += batch_acc * batch_size
            
            pbar.set_postfix({
                "batch_Acc": f"{batch_acc:.4f}"
            })

        avg_metrics = {key: total / max(total_samples, 1) 
                       for key, total in total_metrics.items()
                       }
    
    
        print("\n" + "="*40)
        print(f"Full Test Set Evaluation Results for {args.task}")
        print(f"Total Samples: {total_samples}")

        print(f"Average Accuracy: {avg_metrics['acc']:.6f}")
        return avg_metrics['acc'], avg_metrics['acc'], results
    
    elif args.task == 'environment_forecast':

        for batch_idx, batch in enumerate(pbar):

            fusion_strategy = getattr(args, 'fusion_strategy', 'early_concat')
            inputs_embeds, attn_mask, txt_embeds, txt_attn_mask, ts_embed = fuse_ts_and_text(
                args, model, ts_encoder, align_layer, batch, embed_device, model_dtype
            )
            
            true_seq = batch['output_window'].to(embed_device)
            mean, std = batch['denorm_params'] #this is dynanically saved to batch in the fuse_ts_and_text function
            mean = mean.to(embed_device); std = std.to(embed_device)

            if fusion_strategy.startswith('early_'):
                # Early fusion: process fused embeddings through LLM
                llm_outputs = model(inputs_embeds=inputs_embeds, 
                                    attention_mask=attn_mask, 
                                    output_router_logits=False, 
                                    use_cache=False)
                final_hidden = llm_outputs.last_hidden_state  # let's try this
                final_hidden = final_hidden.to(embed_device, torch.float32)

            elif fusion_strategy.startswith('late_'):
                # Late fusion: process text and TS separately
                llm_outputs = model(
                    inputs_embeds=txt_embeds,
                    attention_mask=txt_attn_mask,
                    output_router_logits=False,
                    use_cache=False
                )
                text_features = llm_outputs.last_hidden_state  # [B, L_txt, d]
                text_features = text_features.to(embed_device, torch.float32)

                if fusion_strategy == 'late_concat':
                    text_pooled = text_features.mean(dim=1) # [B, d]
                    ts_pooled   = ts_embed.mean(dim=1)# [B, d_ts]
                    ts_pooled = align_layer(ts_pooled)  # [B, d] -> align to llm hidden dimension 
                    final_hidden = torch.cat([ts_pooled, text_pooled], dim=1)  # [B, 2d]
                elif fusion_strategy == 'late_add':
                    text_pooled = text_features.mean(dim=1) # [B, d]
                    ts_pooled   = ts_embed.mean(dim=1)# [B, d_ts]
                    ts_pooled = align_layer(ts_pooled)  # [B, d] -> align to LLM dim
                    final_hidden = ts_pooled + text_pooled # [B, d]
                elif fusion_strategy == 'late_crossattn':
                    txt_key_padding_mask = ~txt_attn_mask.bool() 
                    ts_embed = align_layer[0](ts_embed) 

                    attn_out, _ = align_layer[1](query=ts_embed,
                                                 key=txt_embeds,
                                                 value=txt_embeds,
                                                 key_padding_mask=txt_key_padding_mask,
                                                 need_weights=False
                                                 )  # [B, L_ts, d_llm]

                    final_hidden = ts_embed + attn_out #[B, L_ts, d_ts] poetntially bf16
                    final_hidden = final_hidden.to(torch.float32)
                    #print(final_hidden.shape, final_hidden.dtype)
                else:
                    raise ValueError(f"Unsupported late fusion strategy: {fusion_strategy}")

            #print(final_hidden.dtype, downstream_head.proj.weight.dtype)
            pred_seq = downstream_head(final_hidden) # [B, out_len]
            pred_seq = denormalize_timeseries(pred_seq, (mean, std))

            batch_size = pred_seq.size(0)
            total_samples += batch_size

            results["predictions"].append(pred_seq.to(torch.float32).cpu().numpy())
            results["ground_truth"].append(true_seq.to(torch.float32).cpu().numpy())
            results["sample_indices"].extend([f"{batch_idx}_{i}" for i in range(batch_size)])

            batch_mape = calculate_mape(pred_seq, true_seq)#calculate_mape(pred_seq, true_seq)
            batch_mae = mae_criterion(pred_seq, true_seq)
            
            total_metrics['mape'] += batch_mape.item() * batch_size
            total_metrics['mae'] += batch_mae.item() * batch_size
            
            pbar.set_postfix({
                "batch_MAPE": f"{batch_mape.item():.4f}",
                "batch_MAE": f"{batch_mae.item():.4f}",
            })

        avg_metrics = {key: total / max(total_samples, 1) 
                       for key, total in total_metrics.items()
                       }
    
        results["predictions"] = np.concatenate(results["predictions"], axis=0)
        results["ground_truth"] = np.concatenate(results["ground_truth"], axis=0)
    
        print("\n" + "="*40)
        print(f"Full Test Set Evaluation Results for {args.task}")
        print(f"Total Samples: {total_samples}")

        print(f"Average MAPE: {avg_metrics['mape']:.6f}")
        print(f"Average MAE: {avg_metrics['mae']:.6f}")
        return avg_metrics['mape'], avg_metrics['mae'], results
    
    if args.task in ['energy_forecast', 'healthus_forecast', 'healthafr_forecast', 'socialgood_forecast']:

        for batch_idx, batch in enumerate(pbar):

            fusion_strategy = getattr(args, 'fusion_strategy', 'early_concat')
            inputs_embeds, attn_mask, txt_embeds, txt_attn_mask, ts_embed = fuse_ts_and_text(
                args, model, ts_encoder, align_layer, batch, embed_device, model_dtype
            )
            
            true_seq = batch['output_window'].to(embed_device)

            mean, std = batch['denorm_params'] #this is dynanically saved to batch in the fuse_ts_and_text function
            mean = mean.to(embed_device); std = std.to(embed_device)


            if fusion_strategy.startswith('early_'):
                # Early fusion: process fused embeddings through LLM
                llm_outputs = model(inputs_embeds=inputs_embeds, 
                                    attention_mask=attn_mask, 
                                    output_router_logits=False, 
                                    use_cache=False)
                final_hidden = llm_outputs.last_hidden_state  # let's try this
                final_hidden = final_hidden.to(embed_device, torch.float32)

            elif fusion_strategy.startswith('late_'):
                # Late fusion: process text and TS separately
                llm_outputs = model(
                    inputs_embeds=txt_embeds,
                    attention_mask=txt_attn_mask,
                    output_router_logits=False,
                    use_cache=False
                )
                text_features = llm_outputs.last_hidden_state  # [B, L_txt, d]
                text_features = text_features.to(embed_device, torch.float32)

                if fusion_strategy == 'late_concat':
                    text_pooled = text_features.mean(dim=1) # [B, d]
                    ts_pooled   = ts_embed.mean(dim=1)# [B, d_ts]
                    ts_pooled = align_layer(ts_pooled)  # [B, d] -> align to llm hidden dimension 
                    final_hidden = torch.cat([ts_pooled, text_pooled], dim=1)  # [B, 2d]
                elif fusion_strategy == 'late_add':
                    text_pooled = text_features.mean(dim=1) # [B, d]
                    ts_pooled   = ts_embed.mean(dim=1)# [B, d_ts]
                    ts_pooled = align_layer(ts_pooled)  # [B, d] -> align to LLM dim
                    final_hidden = ts_pooled + text_pooled # [B, d]
                elif fusion_strategy == 'late_crossattn':
                    txt_key_padding_mask = ~txt_attn_mask.bool() 
                    ts_embed = align_layer[0](ts_embed) 

                    attn_out, _ = align_layer[1](query=ts_embed,
                                                 key=txt_embeds,
                                                 value=txt_embeds,
                                                 key_padding_mask=txt_key_padding_mask,
                                                 need_weights=False
                                                 )  # [B, L_ts, d_llm]

                    final_hidden = ts_embed + attn_out #[B, L_ts, d_ts] poetntially bf16
                    final_hidden = final_hidden.to(torch.float32)
                    #print(final_hidden.shape, final_hidden.dtype)
                else:
                    raise ValueError(f"Unsupported late fusion strategy: {fusion_strategy}")

            pred_seq = downstream_head(final_hidden) # [B, out_len]
            pred_seq = denormalize_timeseries(pred_seq, (mean, std))

            batch_size = pred_seq.size(0)
            total_samples += batch_size

            results["predictions"].append(pred_seq.to(torch.float32).cpu().numpy())
            results["ground_truth"].append(true_seq.to(torch.float32).cpu().numpy())
            results["sample_indices"].extend([f"{batch_idx}_{i}" for i in range(batch_size)])

            batch_mse = mse_criterion(pred_seq, true_seq)
            batch_mae = mae_criterion(pred_seq, true_seq)
            
            total_metrics['mse'] += batch_mse.item() * batch_size
            total_metrics['mae'] += batch_mae.item() * batch_size
            
            pbar.set_postfix({
                "batch_MSE": f"{batch_mse.item():.4f}",
                "batch_MAE": f"{batch_mae.item():.4f}",
            })

        avg_metrics = {key: total / max(total_samples, 1) 
                       for key, total in total_metrics.items()
                       }
    
        results["predictions"] = np.concatenate(results["predictions"], axis=0)
        results["ground_truth"] = np.concatenate(results["ground_truth"], axis=0)
    
        print("\n" + "="*40)
        print(f"Full Test Set Evaluation Results for {args.task}")
        print(f"Total Samples: {total_samples}")

        print(f"Average MSE: {avg_metrics['mse']:.6f}")
        print(f"Average MAE: {avg_metrics['mae']:.6f}")
        return avg_metrics['mse'], avg_metrics['mae'], results
    
    else:
        raise KeyError("agrs.task not supproted")




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
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)

    # -------- LoRA --------
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_modules_to_save", default=["ts_gate"])
    parser.add_argument("--lora_modules", default="shared_expert_gate")

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
    parser.add_argument("--pre_align_clip", type=float, default=1e5,
                        help="If > 0, clamp TS embedding to [-clip, clip] BEFORE AlignLayer (both train & test).")
    parser.add_argument("--use_bfloat16", action="store_true",
                        help="Load LLM in bfloat16 (if supported) instead of float16.")

    # -------- Others --------
    parser.add_argument("--fusion_strategy", type=str, default="early_concat",
                        choices=["early_concat", "early_crossattn", 
                                 "late_concat", "late_add", "late_crossattn"],
                        help="Fusion strategy: early (input-level) or late (feature-level)")
    
    parser.add_argument("--project_name", default="Fusion")
    parser.add_argument("--scale", type=bool, default=False)
    parser.add_argument("--upsampling_pad_direction", type=str, default="forward")
    parser.add_argument("--upsampling_type", type=str, default="pad")
    parser.add_argument("--downsampling_type", type=str, default="average")
    parser.add_argument("--pad_mode", type=str, default="constant")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", type=bool, default=True)

    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--new_architecture", action="store_true")
    parser.add_argument("--task", type=str, default="weather_forecast")
    parser.add_argument("--finance_trend_choice", type=str, default="3way") #'3way' or '5way'
    parser.add_argument("--weather_trend_choice", type=str, default="future") #'past' or 'future'
    parser.add_argument("--lora_trainable", action="store_true")
    parser.add_argument("--use_answer_token", action="store_true")

    # -------- Eval --------
    parser.add_argument("--eval_mode", type=str, default="full_test", help="full_test or random_sample")
    parser.add_argument("--save_full_predictions", action="store_true", help="whether or not save predicted series")
    parser.add_argument("--use_checkpoint_args", action="store_true", help="whether or not use identical configurations as checkpoint")
    parser.add_argument("--checkpoint_path", type=str, default="./output/8-28-base2/ts_encoder_epoch3.pt")
    parser.add_argument("--lora_adapter_path", type=str, default="./output/8-28-base2")
    parser.add_argument("--sample_seed", type=int, default=7)

    return parser.parse_args()



def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    random.seed(args.sample_seed)

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
    elif args.task == "finance_forecast":
        dataset = FinanceDataset(
            data_dir=args.dataset_path,
            tokenizer=tokenizer,
            input_len=args.in_len,
            output_len=args.out_len, 
            max_text_length=2048
            )
    
    elif args.task == "finance_trend_prediction":
        dataset = FinanceDataset(
            data_dir=args.dataset_path,
            tokenizer=tokenizer,
            input_len=args.in_len,
            output_len=args.out_len, 
            max_text_length=2048
            )
        
    elif args.task == "environment_forecast":
        dataset = EnvironmentDataset(
            data_dir=args.dataset_path,
            tokenizer=tokenizer,
            input_len=args.in_len, # 7
            output_len=args.out_len, # 1
            max_text_length=512
            )
    elif args.task == "energy_forecast":
        dataset = EnergyDataset(
            data_dir=args.dataset_path,
            tokenizer=tokenizer,
            input_len=args.in_len, # 14
            output_len=args.out_len, # 3
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
    
    if args.ts_encoder == 'time-moe':
        args.hidden_dim = 384 #the hidden dimension for pretrained time-moe
    else:
        pass

    # build the models
    model, ts_encoder, align_layer, downstream_head, embed_device = load_model_components(args)
    model_dtype =  model.dtype
    
    if args.task in ['environment_forecast', 'energy_forecast', 'healthus_forecast', 'healthafr_forecast', 'socialgood_forecast']:
        # Dataset from TimeMMD
        test_loader = DataLoader(dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=args.num_workers
                                 )
    else:                        
        _, _, test_loader = build_loader_from_saved(dataset=dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                                    train_ratio=args.train_ratio,val_ratio=args.val_ratio, seed=args.seed, 
                                                    save_dir=args.data_pkl_dir, suffix=args.data_suffix)
    #test_loader = get_test_dataloader(args)
    

    if args.eval_mode == "random_sample":

        raise NotImplementedError("`args.eval_mode == random_sample` is not implemented!")
        
    elif args.eval_mode == "full_test":
        if args.task in ["weather_forecast", "energy_forecast", "finance_forecast", 'healthus_forecast', 'healthafr_forecast', 'socialgood_forecast']:
            avg_metric1, avg_metric2, results = evaluate_full_test_set(
            args, model, ts_encoder, align_layer, downstream_head,
            test_loader, embed_device, model_dtype
        )
            metric_names = ("MSE", "MAE")
        elif args.task in ["environment_forecast", "finance_forecast"]:
            avg_metric1, avg_metric2, results = evaluate_full_test_set(
            args, model, ts_encoder, align_layer, downstream_head,
            test_loader, embed_device, model_dtype
        )
            metric_names = ("MAPE", "MAE")
        elif args.task in ["finance_trend_prediction", "weather_trend_prediction"]:
            avg_metric1, avg_metric2, results = evaluate_full_test_set(
            args, model, ts_encoder, align_layer, downstream_head,
            test_loader, embed_device, model_dtype
        )
            metric_names = ("ACC", "ACC")
        elif args.task in ["weather_MCQA", "finance_MCQA"]:
            avg_metric1, avg_metric2, results = evaluate_full_test_set(
            args, model, ts_encoder, align_layer, downstream_head,
            test_loader, embed_device, model_dtype, tokenizer
        )
            metric_names = ("ACC", "ACC")
        else:
            raise NotImplementedError()

        summary_path = os.path.join(args.output_dir, "test_set_evaluation_summary.json")

        result_summary = {
            f"average_{metric_names[0].lower()}": avg_metric1,
            f"average_{metric_names[1].lower()}": avg_metric2,
            "total_samples": len(results["predictions"]),
            "task": args.task,
        }

        with open(summary_path, "w") as f:
            json.dump(result_summary, f, indent=2)

        print(f"Evaluation results is saved in test_set_evaluation_summary.json")

        if args.save_full_predictions:
            pred_path = os.path.join(args.output_dir, "test_predictions.npz")
            np.savez(
                pred_path,
                predictions=results["predictions"],
                ground_truth=results["ground_truth"],
                sample_indices=results["sample_indices"]
            )
            print(f"Complete results are saved at: {pred_path}")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    main()