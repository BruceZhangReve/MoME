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

from torch.utils.data import DataLoader
from data.dataloader import build_loader_from_saved
from data.dataset import (WeatherDataset, FinanceDataset, EnvironmentDataset, EnergyDataset,
                          HealthUSDataset, HealthAFRDataset, SocialGoodDataset)
from utils import (
    normalize_timeseries, 
    denormalize_timeseries, 
    calculate_mape,
    get_transformer_backbone
)
from TS_Encoder.mome import MoMe, MoMeP
from TS_Encoder.iTransformer import iTransformer
from TS_Encoder.MiTransformer import MiTransformer
from TS_Encoder.mmlinear import MMLinear, MMLinearP
from TS_Encoder.patchTST import PatchTST
from TS_Encoder.dlinear import DLinear
from TS_Encoder.GPT4TS import GPT4TS
from TS_Encoder.TSMix import TSMixer
from TS_Encoder.timellm import Model as TimeLLM
from layers import QueryPool

def get_ts_embed(
    args,
    llm,
    ts_encoder: nn.Module,
    batch,
    language_instructor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Return inputs_embeds.
    Note: 
        ts: tensor [B, in_len]
    """
    # Initial processing and move to embedding device
    ts = batch['input_window'].to(args.device)
    normalized_ts, denorm_params = normalize_timeseries(batch['input_window'])
    batch['denorm_params'] = denorm_params
    ts = normalized_ts.to(args.device)

    if args.ts_encoder in ["MoMe", "MoMeP", "mmlinear", "mmlinearp", "MiTransformer"]:
        if ts_encoder.modulation:
            txt_embeds = llm.get_input_embeddings()(batch['text_input_ids'].to(args.device))  # [B, t, d_llm]
            txt_attn_mask = batch['text_attention_mask'].to(args.device)
            with torch.no_grad():
                outputs = llm(inputs_embeds=txt_embeds, 
                              attention_mask=txt_attn_mask, 
                              output_hidden_states=True,
                              use_cache=False
                              )
            last_hidden = outputs.hidden_states[-1]  # [B, T, d_llm]

            Ins_tk = language_instructor(last_hidden.to(torch.float32), txt_attn_mask) # ... => [B, N_i, d]
        else:
            pass
    else:
        pass
    
    # TS and Text encode
    if args.ts_encoder == 'iTransformer':
        ts = ts.unsqueeze(2) #[B, in_len] -> [B, in_len, 1]
        ts_embed = ts_encoder(ts) # [B, in_len, 1] -> [B,t,d]
    elif args.ts_encoder == 'MiTransformer':
        ts = ts.unsqueeze(2) #[B, in_len] -> [B, in_len, 1], since it's single channel
        if ts_encoder.modulation == False:
            ts_embed = ts_encoder(ts) # [B, 1, in_len] -> [B, 1*P, d]
        else:
            ts_embed = ts_encoder(ts, Ins_tk) # [B, 1, in_len] -> [B, 1*P, d]
    elif args.ts_encoder == 'PatchTST':
        ts = ts.unsqueeze(2) #[B, in_len] -> [B, in_len, 1], since it's single channel
        ts_embed = ts_encoder(ts) # [B, in_len, 1] -> [B, t, d] in single channel
    elif args.ts_encoder == 'GPT4TS':
        ts = ts.unsqueeze(2) #[B, in_len] -> [B, in_len, 1], since it's single channel
        ts_embed = ts_encoder(ts).permute(0, 2, 1) # [B, 1, out_len]
    elif args.ts_encoder == 'TimeLLM':
        ts = ts.unsqueeze(2) #[B, in_len] -> [B, in_len, 1], since it's single channel
        ts_embed = ts_encoder(ts).permute(0, 2, 1) # [B, 1, out_len]
    elif args.ts_encoder == 'DLinear':
        ts = ts.unsqueeze(-1) #[B, in_len] -> [B, in_len, 1]
        ts_embed = ts_encoder(ts) # [B, in_len, 1] -> [B, 1, out_len]
    elif args.ts_encoder == 'TSMixer':
        ts = ts.unsqueeze(2) #[B, in_len] -> [B, in_len, 1], since it's single channel
        #print(ts.shape)
        ts_embed = ts_encoder(ts).permute(0, 2, 1) # [B, 1, out_len]
    elif args.ts_encoder in ['mmlinear', 'mmlinearp']:
        ts = ts.unsqueeze(1) #[B, in_len] -> [B, 1, in_len]
        if ts_encoder.modulation == False:
            ts_embed = ts_encoder(ts) # [B, 1, in_len] -> [B, 1, out_len]
        else:
            ts_embed = ts_encoder(ts, Ins_tk) 
    elif args.ts_encoder in ['MoMe', 'MoMeP']:
        ts = ts.unsqueeze(1) #[B, in_len] -> [B, 1, in_len]
        if ts_encoder.modulation == False:
            ts_embed = ts_encoder(ts) # [B, 1, in_len] -> [B, 1*P, d]
        else:
            ts_embed = ts_encoder(ts, Ins_tk) # [B, 1, in_len] -> [B, 1*P, d]
    elif args.ts_encoder == 'time-moe':
        ts_encoder = get_transformer_backbone(ts_encoder)
        out = ts_encoder(ts, use_cache=False) #[B, in_len]
        ts_embed = out.last_hidden_state # [B, t, d_ts] or time-moe, a timestamp is a token. We look for the states at last hidden layer
    else:
        raise ValueError("Please specify a valid TS-Encoder!")

    return ts_embed



def load_model_components(args):
    dtype = torch.bfloat16 if args.use_bfloat16 else torch.float32

    if args.modulation:
        print(f"Finetuned Base-LLM Loaded")
        llm = AutoModelForCausalLM.from_pretrained(
            args.llm_model,
            torch_dtype=dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True
            )
        llm.eval() 
        for param in llm.parameters():
            param.requires_grad = False
        print(f"LLM Loaded")
    else:
        llm = None

    print(f"Loading TS-Encoder: {args.ts_encoder}")
    if args.ts_encoder == 'iTransformer':
        ts_token_num = args.in_len
        ts_encoder = iTransformer(
            n_vars=1, seq_len=ts_token_num , d_model=args.hidden_dim, dropout=0.1,
            topk=args.topk, moe_mode=args.moe_mode,n_experts=args.n_experts, 
            n_heads=2, n_layers=args.n_layers
            ).to(args.device)
        for param in ts_encoder.parameters():
            param.requires_grad = False
        Instructor = None

    elif args.ts_encoder == 'MiTransformer':
        ts_token_num = args.in_len
        ts_encoder = MiTransformer(
            n_vars=1, seq_len=ts_token_num , d_model=args.hidden_dim, dropout=0.1,
            topk=args.topk, router_modulation=args.router_modulation, n_experts=args.n_experts, 
            n_heads=2, n_layers=args.n_layers
            ).to(args.device)
        
        Instructor = None
        if ts_encoder.modulation:
            Instructor = QueryPool(d_model=llm.config.hidden_size,
                                   n_queries=args.instructor_query,
                                   n_heads=1,
                                   d_proj=args.hidden_dim,
                                   dropout=args.dropout).to(args.device)
        else:
            pass

        for param in ts_encoder.parameters():
            param.requires_grad = False

    elif args.ts_encoder == 'PatchTST':
        ts_encoder = PatchTST(args, args.in_len, args.hidden_dim, args.patch_len).to(args.device)
        ts_token_num = ts_encoder.patch_num
        
        Instructor = None
        for param in ts_encoder.parameters():
            param.requires_grad = False

    elif args.ts_encoder == 'GPT4TS':
        #raise ValueError("Invalid Time Series Encoder!")
        ts_encoder = GPT4TS(args).to(args.device)
        Instructor = None
        for param in ts_encoder.parameters():
            param.requires_grad = False

    elif args.ts_encoder == 'TimeLLM':
        ts_encoder = TimeLLM(args).to(args.device)
        Instructor = None
        for param in ts_encoder.parameters():
            param.requires_grad = False

    elif args.ts_encoder == 'DLinear':
        ts_encoder = DLinear(args, in_len=args.in_len, out_len=args.out_len).to(args.device)
        
        Instructor = None
        for param in ts_encoder.parameters():
            param.requires_grad = False

    elif args.ts_encoder == 'TSMixer':
        ts_encoder = TSMixer(args).to(args.device)
        
        Instructor = None
        for param in ts_encoder.parameters():
            param.requires_grad = False

    elif args.ts_encoder == 'time-moe':
        ts_token_num = args.in_len
        ts_encoder_path = "./TS_Encoder/TimeMoE-50M"
        ts_encoder = AutoModelForCausalLM.from_pretrained(
            ts_encoder_path, trust_remote_code=True
            ).to(args.device)
        ts_encoder.eval()

        Instructor = None

    elif args.ts_encoder == 'mmlinearp':
        ts_encoder = MMLinearP(in_len=args.in_len, out_len=args.out_len,
                              top_k=args.topk, n_experts=args.n_experts,
                              modulation=args.modulation).to(args.device)
        
        Instructor = None
        if ts_encoder.modulation:
            Instructor = QueryPool(d_model=llm.config.hidden_size,
                                   n_queries=args.instructor_query,
                                   n_heads=1,
                                   d_proj=args.out_len,
                                   dropout=args.dropout).to(args.device)
        for param in ts_encoder.parameters():
            param.requires_grad = False

    elif args.ts_encoder == 'mmlinear':
        ts_encoder = MMLinear(in_len=args.in_len, out_len=args.out_len,
                              top_k=args.topk, n_experts=args.n_experts,
                              modulation=args.modulation).to(args.device)
        
        Instructor = None
        if ts_encoder.modulation:
            Instructor = QueryPool(d_model=llm.config.hidden_size,
                                   n_queries=args.instructor_query,
                                   n_heads=1,
                                   d_proj=args.out_len,
                                   dropout=args.dropout).to(args.device)
        for param in ts_encoder.parameters():
            param.requires_grad = False

    elif args.ts_encoder == 'MoMe':
        ts_encoder = MoMe(in_len=args.in_len, patch_len=args.patch_len, hidden_dim=args.hidden_dim, 
                          top_k=args.topk, num_experts=args.n_experts, modulation=args.modulation).to(args.device)
        ts_token_num = ts_encoder.patch_num
        for param in ts_encoder.parameters():
            param.requires_grad = False
        
        Instructor = None
        if ts_encoder.modulation:
            Instructor = QueryPool(d_model=llm.config.hidden_size,
                                   n_queries=args.instructor_query,
                                   n_heads=1,
                                   d_proj=args.hidden_dim,
                                   dropout=args.dropout).to(args.device)
    elif args.ts_encoder == 'MoMeP':
        ts_encoder = MoMeP(in_len=args.in_len, patch_len=args.patch_len, hidden_dim=args.hidden_dim, 
                           top_k=args.topk, num_experts=args.n_experts, modulation=args.modulation).to(args.device)
        ts_token_num = ts_encoder.patch_num
        for param in ts_encoder.parameters():
            param.requires_grad = False
        
        Instructor = None
        if ts_encoder.modulation:
            Instructor = QueryPool(d_model=llm.config.hidden_size,
                                   n_queries=args.instructor_query,
                                   n_heads=1,
                                   d_proj=args.hidden_dim,
                                   dropout=args.dropout).to(args.device)
    else:
        raise ValueError("Invalid Time Series Encoder!")


    if args.task in ['finance_trend_prediction', 'weather_trend_prediction']:
        if args.task == 'finance_trend_prediction':
            if args.finance_trend_choice == '3way':
                num_class = 3
            elif args.finance_trend_choice == '5way':
                num_class = 5
            else:
                raise ValueError("args.finance_trend_choice is either 3way or 5 way")

            if args.ts_encoder in ['mmlinear', 'mmlinearp', 'DLinear', 'GPT4TS', 'TimeLLM', 'TSMixer']:
                downstream_head = None
            else:
                downstream_head = nn.Sequential(nn.Flatten(start_dim=-2),
                                                nn.Linear(ts_token_num * args.hidden_dim, num_class)).to(args.device)

        elif args.task == 'weather_trend_prediction':
            num_class = 3
            if args.ts_encoder in ['mmlinear', 'mmlinearp', 'DLinear', 'GPT4TS', 'TimeLLM', 'TSMixer']:
                downstream_head = None
            else:
                downstream_head = nn.Sequential(nn.Flatten(start_dim=-2),
                                                nn.Linear(ts_token_num * args.hidden_dim, num_class)).to(args.device)

        else:
            raise KeyError(f"Please specify a valid args.task!")
        
    elif args.task in ['finance_forecast', 'weather_forecast', 'environment_forecast', 'energy_forecast', 'healthus_forecast', 'healthafr_forecast', 'socialgood_forecast']:
        # [B, T, d] => [B, T*d] => [B, out_len]
        if args.ts_encoder in ['mmlinear', 'mmlinearp', 'DLinear', 'GPT4TS', 'TimeLLM', 'TSMixer']:
            downstream_head = None
        else:
            downstream_head = nn.Sequential(nn.Flatten(start_dim=-2),
                                            nn.Linear(ts_token_num * args.hidden_dim, args.out_len)).to(args.device)

    else:
         raise KeyError(f"Please specify a valid args.task!")


    print(f"Load Pre-trained Parameters: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=args.device)

    #print(ts_encoder)
    ts_encoder.load_state_dict(checkpoint["ts_encoder"])

    if downstream_head is not None:
        downstream_head.load_state_dict(checkpoint["downstream_head"])

    if Instructor is not None:
        Instructor.load_state_dict(checkpoint["Instructor"])
    
    if args.use_checkpoint_args:
        print("Using the configurations in the checkpoint file.")
        checkpoint_args = checkpoint["args"]
        for key, value in checkpoint_args.items():
            if hasattr(args, key):
                setattr(args, key, value)
    
    return ts_encoder, downstream_head, llm, Instructor


@torch.no_grad()
def evaluate_test_sample(
    args,
    llm,
    tokenizer,
    ts_encoder,
    downstream_head,
    language_instructor,
    test_loader,
):
    assert args.batch_size == 1, "please set batch_size = 1 to do sample-level evaluation!"
    ts_encoder.eval()
    if downstream_head is not None:
        downstream_head.eval()

    results = {
        "predictions": [],
        "ground_truth": [],
        "input_sequence": [],
        "timestamps": [],
        "text": ""
    }

    mse_criterion = torch.nn.MSELoss()
    mae_criterion = torch.nn.L1Loss()

    random_idx = random.randint(0, len(test_loader) - 1)
    print(f"random_idx: {random_idx}")
    random_batch = None
    for idx, batch in enumerate(test_loader):
        if idx == random_idx:
            random_batch = batch
            break
    if random_batch is None:
        raise ValueError("Fail to retrive a random batch from test_loader, check if test_loader is empty!")
    
    text_input_ids = random_batch['text_input_ids'][0].cpu()  # [max_text_len]
    decoded_text = tokenizer.decode(
        text_input_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )
    results["text"] = decoded_text

    if args.task == "weather_forecast":
        true_seq = random_batch['output_window'].to(args.device)
        inputs_embeds = get_ts_embed(args, llm, ts_encoder, batch, language_instructor)

        mean, std = batch['denorm_params'] #this is dynanically saved to batch in the fuse_ts_and_text function
        mean = mean.to(args.device); std = std.to(args.device)  

        if downstream_head is not None:
            pred_seq = downstream_head(inputs_embeds)  # [B, out_len]
        else:
            pred_seq = inputs_embeds  # [B, out_len]

        pred_seq = denormalize_timeseries(pred_seq, (mean, std))

        metric1 = mse_criterion(pred_seq, true_seq)
        metric2 = mae_criterion(pred_seq, true_seq)
        metric_names = ("MSE", "MAE")

        pred_seq = pred_seq.to(torch.float32).cpu().numpy()
        ground_truth = true_seq.to(torch.float32).cpu().numpy()  # the appended value should be of shape [B, out_len], bu usually B=1
        input_seq = random_batch['input_window'].numpy()
    elif args.task == "finance_forecast":
        true_seq = random_batch['output_window'].to(args.device)
        inputs_embeds = get_ts_embed(args, llm, ts_encoder, batch, language_instructor)  

        mean, std = batch['denorm_params'] #this is dynanically saved to batch in the fuse_ts_and_text function
        mean = mean.to(args.device); std = std.to(args.device)  

        if downstream_head is not None:
            pred_seq = downstream_head(inputs_embeds)  # [B, out_len]
        else:
            pred_seq = inputs_embeds  # [B, out_len]

        pred_seq = denormalize_timeseries(pred_seq, (mean, std))

        metric1 = calculate_mape(pred_seq, true_seq)
        metric2 = mae_criterion(pred_seq, true_seq)
        metric_names = ("MAPE", "MAE")

        pred_seq = pred_seq.to(torch.float32).cpu().numpy()
        ground_truth = true_seq.to(torch.float32).cpu().numpy()  # the appended value should be of shape [B, out_len], bu usually B=1
        input_seq = random_batch['input_window'].numpy()

        print(f"Input trend: {random_batch['input_trend']}, Output Trend: {random_batch['output_trend']}")
    elif args.task in ["environment_forecast", "energy_forecast",  'healthus_forecast', 'healthafr_forecast', 'socialgood_forecast']:
        true_seq = random_batch['output_window'].to(args.device)
        inputs_embeds = get_ts_embed(args, llm, ts_encoder, batch, language_instructor)

        mean, std = batch['denorm_params'] #this is dynanically saved to batch in the fuse_ts_and_text function
        mean = mean.to(args.device); std = std.to(args.device)  

        pred_seq = downstream_head(inputs_embeds) # [B, out_len]
        pred_seq = denormalize_timeseries(pred_seq, (mean, std))

        metric1 = mse_criterion(pred_seq, true_seq)
        metric2 = mae_criterion(pred_seq, true_seq)
        metric_names = ("MSE", "MAE")

        pred_seq = pred_seq.to(torch.float32).cpu().numpy()
        ground_truth = true_seq.to(torch.float32).cpu().numpy()  # the appended value should be of shape [B, out_len], bu usually B=1
        input_seq = random_batch['input_window'].numpy()
        
    elif args.task in ['finance_trend_prediction', 'weather_trend_prediction']:
        raise NotImplementedError()
    else:
        raise ValueError(f"Not supported: {args.task}")


    results["predictions"].append(pred_seq)  
    results["ground_truth"].append(ground_truth)
    results["input_sequence"].append(input_seq)
    
    print("\n" + "="*40)
    print(f"Evaluation on a Single Batch Results")
    print(f"{metric_names[0]}: {metric1:.6f}")
    print(f"{metric_names[1]}: {metric2:.6f}")
    print("="*40 + "\n")

    # finally, we do some visualization and logging if it's forecasting
    if args.task in ['weather_forecast','finance_forecast']:
        input_length = input_seq.shape[-1]; output_length = ground_truth.shape[-1]
        total_timestamp = np.array(range(input_length + output_length)) #this is for now, we can fix it
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir, exist_ok=True)

        plt.figure(figsize=(12, 6)) 
        sns.set_style("whitegrid") 
        sns.lineplot(
            x=total_timestamp[:input_length],  # change dataset, so that in includes timestamps information, then we can do label on x axis
            y=input_seq.squeeze(),
            label="Input Sequence",
            color="#2ecc71",
            linewidth=6,
            #marker="o",
            #markersize=5
        )
        sns.lineplot(
            x=total_timestamp[input_length:],  
            y=ground_truth.squeeze(),
            label="Ground Truth Future",
            color="#3498db",  
            linewidth=6,
            #marker="s",  
            #markersize=5
        )
        sns.lineplot(
            x=total_timestamp[input_length:],  
            y=pred_seq.squeeze(),
            label="Predicted Future",
            color="#e74c3c",  
            linewidth=6,
            #linestyle="--",  
            #marker="^",  
            #markersize=5
        )
        plt.title(f"Time Series Prediction Sample\n {metric_names[0]}: {metric1:.6f} | {metric_names[1]}: {metric2:.6f}", fontsize=16)
        plt.xlabel("Time", fontsize=14)
        plt.ylabel("Value", fontsize=14)
        plt.xticks(rotation=45, ha="right")  
        plt.legend(fontsize=12)
        plt.tight_layout()  
        plot_save_path = os.path.join(args.output_dir, "sample_prediction_plot.pdf")
        plt.savefig(plot_save_path, dpi=300, bbox_inches="tight") 
        plt.close()  

        text_save_path = os.path.join(args.output_dir, "sample_text.txt")
        with open(text_save_path, "w", encoding="utf-8") as f:
            f.write("Sample Text Information:\n")
            f.write("="*50 + "\n")
            f.write(f"Decoded Text: {decoded_text}\n")
            f.write(f"{metric_names[0]}: {metric1:.6f}\n")
            f.write(f"{metric_names[1]}: {metric2:.6f}\n")

    elif args.task in ['environment_forecast', 'energy_forecast',  'healthus_forecast', 'healthafr_forecast', 'socialgood_forecast']:
        input_length = input_seq.shape[-1]; output_length = ground_truth.shape[-1]
        total_timestamp = np.array(range(input_length + output_length)) #this is for now, we can fix it
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir, exist_ok=True)

        input_seq_1d = input_seq.squeeze()
        ground_truth_1d = np.atleast_1d(ground_truth.squeeze())
        pred_seq_1d = np.atleast_1d(pred_seq.squeeze())

        x_in = list(range(input_length))
        x_out = list(range(input_length, input_length + output_length))

        plt.figure(figsize=(12, 6))
        plt.plot(x_in, input_seq_1d, marker='o', label='Input Window', color='C0', linewidth=2)
        plt.plot(x_out, ground_truth_1d, marker='o', label='Ground Truth', color='C1', linewidth=2)
        plt.plot(x_out, pred_seq_1d, marker='s', label='Prediction', color='C2', linewidth=2, alpha=0.9)
        plt.plot([x_in[-1], x_out[0]],
                 [input_seq_1d[-1], ground_truth_1d[0]],
                 color='gray', linestyle='--', linewidth=1, alpha=0.7)
        plt.plot([x_in[-1], x_out[0]],
                 [input_seq_1d[-1], pred_seq_1d[0]],
                 color='C2', linestyle=':', linewidth=1, alpha=0.6)
        plt.axvline(x=input_length - 1, color='gray', linestyle=':', alpha=0.5, linewidth=1)


        plt.title(f"Time Series Prediction Sample\n {metric_names[0]}: {metric1:.6f} | {metric_names[1]}: {metric2:.6f}", fontsize=14)
        plt.xlabel("Time", fontsize=12)
        plt.ylabel("Value", fontsize=12)
        plt.xticks(rotation=45, ha="right")  
        plt.legend(fontsize=10)
        plt.tight_layout()  
        plot_save_path = os.path.join(args.output_dir, "sample_prediction_plot.pdf")
        plt.savefig(plot_save_path, dpi=300, bbox_inches="tight") 
        plt.close()  

        text_save_path = os.path.join(args.output_dir, "sample_text.txt")
        with open(text_save_path, "w", encoding="utf-8") as f:
            f.write("Sample Text Information:\n")
            f.write("="*50 + "\n")
            f.write(f"Decoded Text: {decoded_text}\n")
            f.write(f"{metric_names[0]}: {metric1:.6f}\n")
            f.write(f"{metric_names[1]}: {metric2:.6f}\n")


    return metric1, metric2, results
    


@torch.no_grad()
def evaluate_full_test_set(
    args,
    llm,
    ts_encoder,
    downstream_head,
    language_instructor,
    test_loader,
) -> tuple[float, float, dict]:
    ts_encoder.eval()
    if downstream_head is not None:
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

    if args.task == 'weather_forecast':

        for batch_idx, batch in enumerate(pbar):
            true_seq = batch['output_window'].to(args.device)

            inputs_embeds = get_ts_embed(args, llm, ts_encoder, batch, language_instructor)
            mean, std = batch['denorm_params'] #this is dynanically saved to batch in the fuse_ts_and_text function
            mean = mean.to(args.device); std = std.to(args.device)

            if downstream_head is not None:
                pred_seq = downstream_head(inputs_embeds)  # [B, out_len]
            else:
                pred_seq = inputs_embeds  # [B, out_len]
                
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
            true_seq = batch['output_window'].to(args.device)

            inputs_embeds = get_ts_embed(args, llm, ts_encoder, batch, language_instructor)
            mean, std = batch['denorm_params'] #this is dynanically saved to batch in the fuse_ts_and_text function
            mean = mean.to(args.device); std = std.to(args.device)

            if downstream_head is not None:
                pred_seq = downstream_head(inputs_embeds)  # [B, out_len]
            else:
                pred_seq = inputs_embeds  # [B, out_len]

            pred_seq = denormalize_timeseries(pred_seq, (mean, std))

            batch_size = pred_seq.size(0)
            total_samples += batch_size

            results["predictions"].append(pred_seq.to(torch.float32).cpu().numpy())
            results["ground_truth"].append(true_seq.to(torch.float32).cpu().numpy())
            results["sample_indices"].extend([f"{batch_idx}_{i}" for i in range(batch_size)])

            batch_mape = calculate_mape(pred_seq, true_seq)
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

                true_label = torch.tensor(label_ids, dtype=torch.long, device=args.device)
            except KeyError as e:
                available = list(RAW_TO_LABEL_5WAY.keys())
                raise KeyError(f"Unknown label: {e}. Available raw labels: {available}")

            inputs_embeds = get_ts_embed(args, llm, ts_encoder, batch, language_instructor)

            if downstream_head is not None:
                logits = downstream_head(inputs_embeds)  # [B, num_class]
            else:
                logits = inputs_embeds  # [B, num_class]

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

        print(f"Average Acc: {avg_metrics['acc']:.6f}")
        return avg_metrics['acc'], avg_metrics['acc'], results
    
    elif args.task == "weather_trend_prediction":
        for batch_idx, batch in enumerate(pbar):
            if args.weather_trend_choice == "past":
                raw_labels = batch['input_trend']
            elif args.weather_trend_choice == "future":
                raw_labels = batch['output_trend']
            else:
                raise KeyError("args.weather_trend_choice not supported!")
            
            label_ids = [LABEL_WT_TO_ID[label] for label in raw_labels]
            true_label = torch.tensor(label_ids, dtype=torch.long, device=args.device)
            
            inputs_embeds = get_ts_embed(args, llm, ts_encoder, batch, language_instructor)

            if downstream_head is not None:
                logits = downstream_head(inputs_embeds)  # [B, num_class]
            else:
                logits = inputs_embeds  # [B, num_class]

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
    
    elif args.task in ['environment_forecast']:

        for batch_idx, batch in enumerate(pbar):
            true_seq = batch['output_window'].to(args.device)

            inputs_embeds = get_ts_embed(args, llm, ts_encoder, batch, language_instructor)
            mean, std = batch['denorm_params'] #this is dynanically saved to batch in the fuse_ts_and_text function
            mean = mean.to(args.device); std = std.to(args.device)

            if downstream_head is not None:
                pred_seq = downstream_head(inputs_embeds)  # [B, out_len]
            else:
                pred_seq = inputs_embeds  # [B, out_len]
                
            pred_seq = denormalize_timeseries(pred_seq, (mean, std))

            batch_size = pred_seq.size(0)
            total_samples += batch_size

            results["predictions"].append(pred_seq.to(torch.float32).cpu().numpy())
            results["ground_truth"].append(true_seq.to(torch.float32).cpu().numpy())
            results["sample_indices"].extend([f"{batch_idx}_{i}" for i in range(batch_size)])

            batch_mape = calculate_mape(pred_seq, true_seq)
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

    elif args.task in ['energy_forecast',  'healthus_forecast', 'healthafr_forecast', 'socialgood_forecast']:

        for batch_idx, batch in enumerate(pbar):
            true_seq = batch['output_window'].to(args.device)

            inputs_embeds = get_ts_embed(args, llm, ts_encoder, batch, language_instructor)
            mean, std = batch['denorm_params'] #this is dynanically saved to batch in the fuse_ts_and_text function
            mean = mean.to(args.device); std = std.to(args.device)

            if downstream_head is not None:
                pred_seq = downstream_head(inputs_embeds)  # [B, out_len]
            else:
                pred_seq = inputs_embeds  # [B, out_len]
                
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
    parser.add_argument("--llm_model", default="./llm/Qwen1.5-MoE-A2.7B") # we still need this for tokenizer to build/load dataset
    parser.add_argument("--dataset_path", default="./data/processed/weather/aligned_in14days_out3days")
    parser.add_argument("--output_dir", default="output/debug")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.0)

    parser.add_argument("--data_pkl_dir", type=str, default="./data/saved_datasets/weather_forecasting")
    parser.add_argument("--data_suffix", type=str, default="in14_out3")


    # -------- Train Hyperparameter --------
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)


    # -------- TS-Encoder & MoE --------
    parser.add_argument("--instructor_query", type=int, default=1)
    parser.add_argument("--modulation", action="store_true", help="MoMe")
    parser.add_argument("--patch_len", type=int, default=4)
    parser.add_argument("--ts_encoder", type=str, default="time-moe")
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--n_vars", type=int, default=1)
    parser.add_argument("--seq_len_channel", type=int, default=128) #This might not be used anymore
    parser.add_argument("--hidden_dim", type=int, default=32) #note 384 is the embedding dimensionf for time-moe
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--n_experts", type=int, default=3)
    parser.add_argument("--topk", type=int, default=1)
    parser.add_argument("--moe_mode", default="No")  # multi_expert / No
    parser.add_argument("--n_heads", type=int, default=4)

    parser.add_argument("--in_len", type=int, default=336)
    parser.add_argument("--out_len", type=int, default=72)
    parser.add_argument("--max_text_length", type=int, default=2048)
    parser.add_argument("--router_modulation", action="store_true", help="MoMe")

    parser.add_argument("--stride", type=int, default=4) #some model like time-llm needs it

    # -------- Numerical Safety & Dtypes --------
    parser.add_argument("--pre_align_clip", type=float, default=1e5,
                        help="If > 0, clamp TS embedding to [-clip, clip] BEFORE AlignLayer (both train & test).")
    parser.add_argument("--use_bfloat16", action="store_true",
                        help="Load LLM in bfloat16 (if supported) instead of float16.")

    # -------- Others --------
    parser.add_argument("--project_name", default="Fusion")
    parser.add_argument("--scale", type=bool, default=False)
    parser.add_argument("--upsampling_pad_direction", type=str, default="forward")
    parser.add_argument("--upsampling_type", type=str, default="pad")
    parser.add_argument("--downsampling_type", type=str, default="average")
    parser.add_argument("--pad_mode", type=str, default="constant")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", type=bool, default=True)
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--task", type=str, default="weather_forecast")
    parser.add_argument("--finance_trend_choice", type=str, default="3way") #'3way' or '5way'
    parser.add_argument("--weather_trend_choice", type=str, default="future") #'past' or 'future'

    # -------- Eval --------
    parser.add_argument("--eval_mode", type=str, default="full_test", help="full_test or random_sample")
    parser.add_argument("--save_full_predictions", action="store_true", help="whether or not save predicted series")
    parser.add_argument("--use_checkpoint_args", action="store_true", help="whether or not use identical configurations as checkpoint")
    parser.add_argument("--checkpoint_path", type=str, default="./output/8-28-base/ts_encoder_epoch3.pt")
    parser.add_argument("--sample_seed", type=int, default=7)

    return parser.parse_args()



def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.sample_seed)

    llm_path = args.llm_model

    tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True)
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
    ts_encoder, downstream_head, llm, Instructor = load_model_components(args)
    
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
                                                    save_dir=args.data_pkl_dir, suffix=args.data_suffix, train_shuffle=False)

    if args.eval_mode == "random_sample":
        evaluate_test_sample(args, llm, tokenizer, ts_encoder, downstream_head, Instructor, test_loader)
        
    elif args.eval_mode == "full_test":

        if args.task == "weather_forecast":
            avg_metric1, avg_metric2, results = evaluate_full_test_set(args, llm, ts_encoder, downstream_head, Instructor, test_loader)
            metric_names = ("MSE", "MAE")
        elif args.task == "finance_forecast":
            avg_metric1, avg_metric2, results = evaluate_full_test_set(args, llm, ts_encoder, downstream_head, Instructor, test_loader)
            metric_names = ("MAPE", "MAE")
        elif args.task == "finance_trend_prediction":
            avg_metric1, avg_metric2, results = evaluate_full_test_set(args, llm, ts_encoder, downstream_head, Instructor, test_loader)
            metric_names = ("ACC", "ACC")
        elif args.task == "weather_trend_prediction":
            avg_metric1, avg_metric2, results = evaluate_full_test_set(args, llm, ts_encoder, downstream_head, Instructor, test_loader)
            metric_names = ("ACC", "ACC")
        elif args.task in ['energy_forecast']:
            avg_metric1, avg_metric2, results = evaluate_full_test_set(args, llm, ts_encoder, downstream_head, Instructor, test_loader)
            metric_names = ("MSE", "MAE")
        elif args.task in ['environment_forecast']:
            avg_metric1, avg_metric2, results = evaluate_full_test_set(args, llm, ts_encoder, downstream_head, Instructor, test_loader)
            metric_names = ("MAPE", "MAE")
        elif args.task in ['energy_forecast', 'healthus_forecast', 'healthafr_forecast', 'socialgood_forecast']:
            avg_metric1, avg_metric2, results = evaluate_full_test_set(args, llm, ts_encoder, downstream_head, Instructor, test_loader)
            metric_names = ("MSE", "MAE")
        else:
            raise KeyError("agrs.task not defined")
        
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

