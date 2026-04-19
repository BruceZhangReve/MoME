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

from torch.utils.data import DataLoader
from data.dataloader import build_loader_from_saved
from data.dataset import (WeatherDataset, FinanceDataset, EnvironmentDataset, EnergyDataset,
                          HealthUSDataset, HealthAFRDataset, SocialGoodDataset)
from llm.utils import (
    count_parameters,
)
from utils import (
    normalize_timeseries, 
    get_transformer_backbone
)

from TS_Encoder.mome import MoMe, MoMeP
from TS_Encoder.mmlinear import MMLinear, MMLinearP
from TS_Encoder.dlinear import DLinear
from TS_Encoder.iTransformer import iTransformer
from TS_Encoder.MiTransformer import MiTransformer
from TS_Encoder.patchTST import PatchTST
from TS_Encoder.TSMix import TSMixer
from TS_Encoder.GPT4TS import GPT4TS
from TS_Encoder.timellm import Model as TimeLLM
from layers import QueryPool


def get_ts_embed(
    args,
    ts_encoder: nn.Module,
    batch,
    llm,
    language_instructor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Return inputs_embeds, attention_mask, labels for LLM.
    Note: 
        ts: tensor [B, in_len]
    """
    # Initial processing and move to embedding device
    ts = batch['input_window'].to(args.device)
    normalized_ts, denorm_params = normalize_timeseries(batch['input_window'])
    batch['denorm_params'] = denorm_params
    ts = normalized_ts.to(args.device)

    if args.ts_encoder in ["MoMe","MoMeP", "mmlinear","mmlinearp", "MiTransformer"]:
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
        #print(ts.shape)
        ts_embed = ts_encoder(ts) # [B, in_len, 1] -> [B, 1, d] in single channel
    elif args.ts_encoder == 'TSMixer':
        ts = ts.unsqueeze(2) #[B, in_len] -> [B, in_len, 1], since it's single channel
        #print(ts.shape)
        ts_embed = ts_encoder(ts).permute(0, 2, 1) # [B, 1, out_len] 
    elif args.ts_encoder == 'GPT4TS':
        ts = ts.unsqueeze(2) #[B, in_len] -> [B, in_len, 1], since it's single channel
        ts_embed = ts_encoder(ts).permute(0, 2, 1) # [B, 1, out_len]
    elif args.ts_encoder == 'TimeLLM':
        ts = ts.unsqueeze(2) #[B, in_len] -> [B, in_len, 1], since it's single channel
        ts_embed = ts_encoder(ts).permute(0, 2, 1) # [B, 1, out_len]
    elif args.ts_encoder == 'DLinear':
        ts = ts.unsqueeze(-1) #[B, in_len] -> [B, in_len, 1]
        ts_embed = ts_encoder(ts) # [B, in_len, 1] -> [B, 1, out_len]
    elif args.ts_encoder in ['mmlinear', 'mmlinearp']:
        ts = ts.unsqueeze(1) #[B, in_len] -> [B, 1, in_len]
        if ts_encoder.modulation == False:
            ts_embed = ts_encoder(ts) # [B, 1, in_len] -> [B, 1, out_len]
        else:
            ts_embed = ts_encoder(ts, Ins_tk)
    elif args.ts_encoder in ['MoMe','MoMeP']:
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


def train_one_epoch(
    args,
    ts_encoder,
    downstream_head,
    dataloader,
    criterion,
    optimizer,
    global_step: int,
    loss_fp: Optional[str],
    llm,
    instructor=None
) -> Tuple[int, float, int]:
    # Switch to training mode
    ts_encoder.train()
    if not downstream_head == None:
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

    pbar = tqdm(dataloader, desc="train")
    epoch_total_loss = 0.0
    epoch_batches = 0
    epoch_correct = 0
    epoch_total_samples = 0  # for accuracy

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

                true_label = torch.tensor(label_ids, dtype=torch.long, device=args.device)
            except KeyError as e:
                available = list(RAW_TO_LABEL_5WAY.keys())
                raise KeyError(f"Unknown label: {e}. Available raw labels: {available}")

            inputs_embeds = get_ts_embed(args, ts_encoder, batch, llm, instructor)

            if args.ts_encoder in ['mmlinear', 'mmlinearp', 'DLinear', 'GPT4TS', 'TimeLLM', 'TSMixer']:
                logits = inputs_embeds.squeeze(1) # you need to set out_len = num_class
            else:
                logits = downstream_head(inputs_embeds)  # [B, C]


            main_loss = criterion(logits, true_label)
            total_loss = main_loss

            # === Compute accuracy ===
            preds = torch.argmax(logits, dim=-1)  # [B]
            correct = (preds == true_label).sum().item()
            epoch_correct += correct
            epoch_total_samples += true_label.size(0)

        elif args.task == 'weather_trend_prediction':

            if args.weather_trend_choice == "past":
                raw_labels = batch['input_trend']
            elif args.weather_trend_choice == "future":
                raw_labels = batch['output_trend']
            else:
                raise KeyError("args.weather_trend_choice not supported!")
            label_ids = [LABEL_WT_TO_ID[label] for label in raw_labels]
            true_label = torch.tensor(label_ids, dtype=torch.long, device=args.device)

            inputs_embeds = get_ts_embed(args, ts_encoder, batch, llm, instructor)
            
            if args.ts_encoder in ['mmlinear', 'mmlinearp', 'DLinear', 'GPT4TS', 'TimeLLM', 'TSMixer']:
                logits = inputs_embeds.squeeze(1) # you need to set out_len = num_class
            else:
                logits = downstream_head(inputs_embeds)  # [B, C]


            main_loss = criterion(logits, true_label)

            total_loss = main_loss

            # === Compute accuracy ===
            preds = torch.argmax(logits, dim=-1)  # [B]
            correct = (preds == true_label).sum().item()
            epoch_correct += correct
            epoch_total_samples += true_label.size(0)

        elif args.task in ['finance_forecast', 'weather_forecast']:
            ##### THIS IS A SEVERE DATA LEAKAGE PROBLEM!!! Data Leakage#####
            #true_seq, _ = normalize_timeseries(batch['output_window'])
            #true_seq = true_seq.to(args.device) #[B, out_len]
            ##### THIS IS A SEVERE DATA LEAKAGE PROBLEM!!! Data Leakage#####

            inputs_embeds = get_ts_embed(args, ts_encoder, batch, llm, instructor)
            mean, std = batch['denorm_params']
            true_seq = (batch['output_window'] - mean) / std  # [B, out_len]
            true_seq = true_seq.to(args.device) #[B, out_len]

            if args.ts_encoder in ['mmlinear', 'mmlinearp', 'DLinear', 'GPT4TS', 'TimeLLM', 'TSMixer']:
                pred_seq = inputs_embeds
            else:
                pred_seq = downstream_head(inputs_embeds)

            main_loss = criterion(pred_seq, true_seq)

            total_loss = main_loss

        elif args.task in ['environment_forecast', 'energy_forecast', 'healthus_forecast', 'healthafr_forecast', 'socialgood_forecast']:

            inputs_embeds = get_ts_embed(args, ts_encoder, batch, llm, instructor)
            mean, std = batch['denorm_params']
            true_seq = (batch['output_window'] - mean) / std  # [B, out_len]
            true_seq = true_seq.to(args.device) #[B, out_len]

            if args.ts_encoder in ['mmlinear', 'mmlinearp', 'DLinear', 'GPT4TS', 'TimeLLM', 'TSMixer']:
                pred_seq = inputs_embeds
            else:
                pred_seq = downstream_head(inputs_embeds)

            main_loss = criterion(pred_seq, true_seq)

            total_loss = main_loss

        else:
            raise ValueError(f"Unsupported task: {args.task}")

        # Backward pass
        total_loss.backward()
        if args.grad_clip_norm and args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                parameters=optimizer.param_groups[0]['params'],
                max_norm=args.grad_clip_norm
            )
        optimizer.step()

##########
        #print("ts_gate.0 gradient:", ts_encoder.Gate.weight.grad)  
        #print("head_grad:", downstream_head[-1].weight.grad)
        #print("beta_geterator:", ts_encoder.EiLM.beta_generator.weight.grad)
##########

        # Logging
        global_step += 1
        loss_item = total_loss.item()
        epoch_total_loss += loss_item
        epoch_batches += 1

        # Update progress bar
        if args.task in ['finance_trend_prediction', 'weather_trend_prediction']:
            pbar.set_postfix(total_loss=f"{loss_item:.4f}", acc=f"{correct / len(true_label):.4f}")
        elif args.task in ['finance_forecast', 'weather_forecast']:
            pbar.set_postfix(total_loss=f"{loss_item:.4f}")
        elif args.task in ['environment_forecast', 'energy_forecast', 'healthus_forecast', 'healthafr_forecast', 'socialgood_forecast']:
            pbar.set_postfix(total_loss=f"{loss_item:.4f}")
        else:
            raise KeyError("args.task not defined!")

        # Log to wandb
        wandb.log({"train/loss": loss_item, "step": global_step})

        # Save to file
        if loss_fp is not None:
            with open(loss_fp, "a") as f:
                f.write(f"train_step {global_step}\ttrain_loss {loss_item:.4f}")
                if args.task in ['finance_trend_prediction', 'weather_trend_prediction']:
                    f.write(f"\tCE_Loss {main_loss.item():.4f}\n")
                elif args.task in ['finance_forecast', 'weather_forecast']:
                    f.write(f"\tMSE {main_loss.item():.4f}\n")
                elif args.task in ['environment_forecast', 'energy_forecast', 'healthus_forecast', 'healthafr_forecast', 'socialgood_forecast']:
                    f.write(f"\tMSE {main_loss.item():.4f}\n")
                else:
                    raise KeyError("args.task not defined!")

    # === After loop: compute average accuracy for classification ===
    avg_accuracy = 0.0
    if args.task in ['finance_trend_prediction', 'weather_trend_prediction'] and epoch_total_samples > 0:
        avg_accuracy = epoch_correct / epoch_total_samples
        print(f"\n[Epoch Train] Accuracy: {avg_accuracy:.4f} ({epoch_correct}/{epoch_total_samples})")
        wandb.log({"train/accuracy": avg_accuracy})  # Log once per epoch
    elif args.task in ['finance_MCQA', 'weather_MCQA'] and epoch_total_samples > 0:
        avg_accuracy = epoch_correct / epoch_total_samples
        print(f"\n[Epoch Train] Accuracy: {avg_accuracy:.4f} ({epoch_correct}/{epoch_total_samples})")
        wandb.log({"train/accuracy": avg_accuracy})  # Log once per epoch
    else:
        pass

    return global_step, epoch_total_loss, epoch_batches


# -------------------- Argparse --------------------
def parse_args():
    parser = argparse.ArgumentParser()

    # -------- Model & Data --------
    parser.add_argument("--llm_model", default="./llm/Qwen1.5-MoE-A2.7B")
    parser.add_argument("--dataset_path", default="./data/processed/finance/pair_in_7days_5minutes_out_1days_5minutes")
    parser.add_argument("--output_dir", default="output/debug")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.0)

    parser.add_argument("--data_pkl_dir", type=str, default="./data/saved_datasets/finance_forecasting")
    parser.add_argument("--data_suffix", type=str, default="in7_out1")


    # -------- Train Hyperparameter --------
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)


    # -------- TS-Encoder & MoE --------
    parser.add_argument("--instructor_query", type=int, default=3)
    parser.add_argument("--modulation", action="store_true", help="MoMe")
    parser.add_argument("--patch_len", type=int, default=4)
    parser.add_argument("--ts_encoder", type=str, default="time-moe")
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--n_vars", type=int, default=1)
    parser.add_argument("--hidden_dim", type=int, default=32) #note 384 is the embedding dimensionf for time-moe
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--n_experts", type=int, default=4)
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--moe_mode", default="No")  # multi_expert / No
    parser.add_argument("--n_heads", type=int, default=4)

    parser.add_argument("--in_len", type=int, default=336)
    parser.add_argument("--out_len", type=int, default=72)
    parser.add_argument("--max_text_length", type=int, default=2048)
    parser.add_argument("--router_modulation", action="store_true", help="MoMe")

    parser.add_argument("--llm_dim", type=int, default=2048) #2048 is the hidden dim for Qwen-MoE
    parser.add_argument("--stride", type=int, default=4) #some model like time-llm needs it

    # -------- Numerical Safety & Dtypes --------
    parser.add_argument("--use_bfloat16", action="store_true",
                        help="Load LLM in bfloat16 (if supported) instead of float16.")

    # -------- Others --------
    parser.add_argument("--project_name", default="Fusion")
    parser.add_argument("--pad_mode", type=str, default="constant")
    parser.add_argument("--device", type=str, default="cuda:0") 
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", type=bool, default=True)

    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--task", type=str, default="weather_forecast")
    parser.add_argument("--finance_trend_choice", type=str, default="3way") #'3way' or '5way'
    parser.add_argument("--weather_trend_choice", type=str, default="future") #'past' or 'future'


    return parser.parse_args()

# -------------------- Main --------------------
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    wandb.init(project=args.project_name, config=vars(args))

    device = args.device
    dtype = torch.bfloat16 if args.use_bfloat16 and torch.cuda.is_bf16_supported() else torch.float32
    
    llm_path = args.llm_model #"./llm/Qwen-7B-Chat"
    tokenizer = AutoTokenizer.from_pretrained(args.llm_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id  
    
    llm = AutoModelForCausalLM.from_pretrained(
        llm_path,
        torch_dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True
        )
    llm.eval() 
    for param in llm.parameters():
        param.requires_grad = False

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

    if args.task in ['environment_forecast', 'energy_forecast', 'healthus_forecast', 'healthafr_forecast', 'socialgood_forecast']:
        # Dataset from TimeMMD
        train_loader = DataLoader(dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers
                                  )
    else:
        # Dataset from MTBench
        train_loader, _, _ = build_loader_from_saved(dataset=dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                                     train_ratio=args.train_ratio,val_ratio=args.val_ratio, seed=args.seed, 
                                                     save_dir=args.data_pkl_dir, suffix=args.data_suffix)
    
    ### Loss function ###
    if args.task in ['finance_trend_prediction', 'weather_trend_prediction']:
        criterion = nn.CrossEntropyLoss() 
    elif args.task in ['finance_forecast', 'weather_forecast']:
        criterion = nn.MSELoss()
    elif args.task in ['environment_forecast', 'energy_forecast', 'healthus_forecast', 'healthafr_forecast', 'socialgood_forecast']:
        criterion = nn.MSELoss()
    else:
        raise KeyError(f"Please specify a valid args.task!")
    ### Loss function ###


    ### Build time-series model and downstream head ###
    if args.ts_encoder == 'iTransformer':
        ts_token_num = args.in_len
        ts_encoder = iTransformer(
            n_vars=1, seq_len=ts_token_num , d_model=args.hidden_dim, dropout=0.1,
            topk=args.topk, moe_mode=args.moe_mode, n_experts=args.n_experts, 
            n_heads=2, n_layers=args.n_layers
            ).to(device)
        
        Instructor = None
        for param in ts_encoder.parameters():
            param.requires_grad = True

    elif args.ts_encoder == 'MiTransformer':
        ts_token_num = args.in_len
        ts_encoder = MiTransformer(
            n_vars=1, seq_len=ts_token_num , d_model=args.hidden_dim, dropout=0.1,
            topk=args.topk, router_modulation=args.router_modulation, n_experts=args.n_experts, 
            n_heads=2, n_layers=args.n_layers
            ).to(device)
        
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
            param.requires_grad = True

    elif args.ts_encoder == 'PatchTST':
        ts_encoder = PatchTST(args, args.in_len, args.hidden_dim, args.patch_len).to(device)
        ts_token_num = ts_encoder.patch_num
        
        Instructor = None
        for param in ts_encoder.parameters():
            param.requires_grad = True

    elif args.ts_encoder == 'TSMixer':
        ts_encoder = TSMixer(args).to(device)
        
        Instructor = None
        for param in ts_encoder.parameters():
            param.requires_grad = True

    elif args.ts_encoder == 'GPT4TS':
        ts_encoder = GPT4TS(args).to(device)
        
        Instructor = None
    
    elif args.ts_encoder == 'TimeLLM':
        ts_encoder = TimeLLM(args).to(device)
        
        Instructor = None


    elif args.ts_encoder == 'DLinear':
        ts_encoder = DLinear(args, in_len=args.in_len, out_len=args.out_len).to(device)
        
        Instructor = None
        for param in ts_encoder.parameters():
            param.requires_grad = True

    elif args.ts_encoder == 'mmlinear':
        ts_encoder = MMLinear(in_len=args.in_len, out_len=args.out_len,
                              top_k=args.topk, n_experts=args.n_experts,
                              modulation=args.modulation).to(device)
        
        Instructor = None
        if ts_encoder.modulation:
            Instructor = QueryPool(d_model=llm.config.hidden_size,
                                   n_queries=args.instructor_query,
                                   n_heads=1,
                                   d_proj=args.out_len,
                                   dropout=args.dropout).to(args.device)
        else:
            pass
        for param in ts_encoder.parameters():
            param.requires_grad = True

    elif args.ts_encoder == 'mmlinearp':
        ts_encoder = MMLinearP(in_len=args.in_len, out_len=args.out_len,
                              top_k=args.topk, n_experts=args.n_experts,
                              modulation=args.modulation).to(device)
        
        Instructor = None
        if ts_encoder.modulation:
            Instructor = QueryPool(d_model=llm.config.hidden_size,
                                   n_queries=args.instructor_query,
                                   n_heads=1,
                                   d_proj=args.out_len,
                                   dropout=args.dropout).to(args.device)
        else:
            pass
        for param in ts_encoder.parameters():
            param.requires_grad = True

    elif args.ts_encoder == 'time-moe':
        ts_token_num = args.in_len
        ts_encoder_path = "./TS_Encoder/TimeMoE-50M"
        ts_encoder = AutoModelForCausalLM.from_pretrained(
            ts_encoder_path, trust_remote_code=True
            )
        ts_encoder = ts_encoder.to(device)
        for param in ts_encoder.parameters():
            param.requires_grad = False

        Instructor = None
        for param in ts_encoder.parameters():
            param.requires_grad = False
    elif args.ts_encoder == 'MoMe':
        ts_encoder = MoMe(in_len=args.in_len, patch_len=args.patch_len, hidden_dim=args.hidden_dim, 
                          top_k=args.topk, num_experts=args.n_experts, modulation=args.modulation).to(device)
        ts_token_num = ts_encoder.patch_num
        for param in ts_encoder.parameters():
            param.requires_grad = True
        Instructor = None
        if ts_encoder.modulation:
            Instructor = QueryPool(d_model=llm.config.hidden_size,
                                   n_queries=args.instructor_query,
                                   n_heads=1,
                                   d_proj=args.hidden_dim,
                                   dropout=args.dropout).to(args.device)
    elif args.ts_encoder == 'MoMeP':
        ts_encoder = MoMeP(in_len=args.in_len, patch_len=args.patch_len, hidden_dim=args.hidden_dim, 
                          top_k=args.topk, num_experts=args.n_experts, modulation=args.modulation).to(device)
        ts_token_num = ts_encoder.patch_num
        for param in ts_encoder.parameters():
            param.requires_grad = True
        Instructor = None
        if ts_encoder.modulation:
            Instructor = QueryPool(d_model=llm.config.hidden_size,
                                   n_queries=args.instructor_query,
                                   n_heads=1,
                                   d_proj=args.hidden_dim,
                                   dropout=args.dropout).to(args.device)
        else:
            pass
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

            if args.ts_encoder == 'time-moe':
                hidden_dim = 384 #the hidden dimension for pretrained time-moe
            else:
                hidden_dim = args.hidden_dim
            
            if args.ts_encoder in ['mmlinear', 'mmlinearp', 'DLinear', 'GPT4TS', 'TimeLLM', 'TSMixer']:
                assert ts_encoder.out_len == num_class, "please set ts_encoder.out_len = num_class to do classification!"
                downstream_head = None
            else:
                downstream_head = nn.Sequential(nn.Flatten(start_dim=-2),
                                                nn.Linear(ts_token_num * hidden_dim, num_class)
                                                ).to(args.device)

            
        elif args.task == 'weather_trend_prediction':
            # haven't modified this one yet
            num_class = 3
            if args.ts_encoder == 'time-moe':
                hidden_dim = 384 #the hidden dimension for pretrained time-moe
            else:
                hidden_dim = args.hidden_dim


            if args.ts_encoder in ['mmlinear', 'mmlinearp', 'DLinear', 'GPT4TS', 'TimeLLM', 'TSMixer']:
                # ought to be [B, C] already
                assert ts_encoder.out_len == num_class, "please set ts_encoder.out_len = num_class to do classification!"
                downstream_head = None
            else:
                # [B, T, d] => [B, T*d] => [B, C]
                downstream_head = nn.Sequential(nn.Flatten(start_dim=-2),
                                                nn.Linear(ts_token_num * hidden_dim, num_class)
                                                ).to(args.device)

        else:
            raise KeyError(f"Please specify a valid args.task!")

    elif args.task in ['finance_forecast', 'weather_forecast', 'energy_forecast', 'environment_forecast', 'healthus_forecast', 'healthafr_forecast', 'socialgood_forecast']:
        # [B, T, d] => [B, T*d] => [B, out_len]
        if args.ts_encoder == 'time-moe':
            hidden_dim = 384 #the hidden dimension for pretrained time-moe
        else:
            hidden_dim = args.hidden_dim

        #print(args.ts_encoder)
        if args.ts_encoder in ['mmlinear', 'mmlinearp', 'DLinear', 'GPT4TS', 'TimeLLM', 'TSMixer']:
            downstream_head = None
        else:
            # [B, T, d] => [B, T*d] => [B, L']
            downstream_head = nn.Sequential(nn.Flatten(start_dim=-2),
                                            nn.Linear(ts_token_num * hidden_dim, args.out_len)
                                            ).to(args.device)
    else:
        raise KeyError(f"Please specify a valid args.task!")
    
    ### Build time-series model and downstream head ###
    

    # optimizer config
    if args.ts_encoder == 'time-moe':
        trainable_params = (
            list(downstream_head.parameters())
        )
    elif (args.ts_encoder in ['mmlinear', 'mmlinearp']) or (args.ts_encoder in ['DLinear', 'TSMixer']):
        trainable_params = (
            list(ts_encoder.parameters())
        )
    elif args.ts_encoder in ['GPT4TS', 'TimeLLM']:
        trainable_params = (
            list(ts_encoder.parameters())
        )
    else:
        trainable_params = (
            list(ts_encoder.parameters()) +
            list(downstream_head.parameters())
        )
    if Instructor is not None:
        trainable_params += list(Instructor.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    

    # Parameter stats
    llm_total = count_parameters(llm, only_trainable=False)
    llm_trainable = count_parameters(llm, only_trainable=True)
    ts_total = count_parameters(ts_encoder, only_trainable=False)
    ts_trainable = count_parameters(ts_encoder, only_trainable=True)

    head_total  = 0
    head_trainable = 0
    if downstream_head is not None:
        head_total = count_parameters(downstream_head, only_trainable=False)
        head_trainable = count_parameters(downstream_head, only_trainable=True)

    instructor_total = 0
    instructor_trainable = 0
    if Instructor is not None:
        instructor_total = count_parameters(Instructor, only_trainable=False)
        instructor_trainable = count_parameters(Instructor, only_trainable=True)

    total_params = llm_total + ts_total + head_total + instructor_total
    total_trainable = llm_trainable + ts_trainable + head_trainable + instructor_trainable

    print("\n===== Parameter Statistics =====")
    print(f"LLM total parameter: {llm_total:.2f} M")
    print(f"LLM trainable parameter (LoRA): {llm_trainable:.2f} M")
    print(f"TS-Encoder total parameter: {ts_total:.2f} M")
    if downstream_head is not None:
        print(f"Downstream Head total parameter: {head_trainable:.2f} M")
    if Instructor is not None:
        print(f"Instructor total parameter: {instructor_total:.2f} M")
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
            args, ts_encoder, downstream_head, train_loader, criterion,
            optimizer, global_step, loss_fpath, llm, Instructor
        )
        epoch_avg_loss = epoch_total_loss / max(epoch_batches, 1)
        print(f"Epoch {epoch} Avg Loss: {epoch_avg_loss:.4f}")
        wandb.log({"train/epoch_avg_loss": epoch_avg_loss, "epoch": epoch})

        # save checkpoints
        checkpoint = {
            "ts_encoder": ts_encoder.state_dict(),
            "args": vars(args)
            }
        if downstream_head is not None:
            checkpoint["downstream_head"] = downstream_head.state_dict()
        if Instructor is not None:
            checkpoint["Instructor"] = Instructor.state_dict()
        
        torch.save(checkpoint, os.path.join(args.output_dir, f"ts_encoder_epoch{epoch}.pt"))


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    main()